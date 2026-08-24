from functools import partial
import torch
import torch.nn as nn
import torchinfo
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from spikingjelly.clock_driven import layer
import copy
from torchvision import transforms
import matplotlib.pyplot as plt
import encoder
import torch
from torch.distributions import Normal
import math

#timestep
T=4

# === A2SG: Bayesian Optimization-based Gradient Adjustment ===
# All operations avoid boolean mask indexing (IndexKernel.cu) to prevent CUDA device-side asserts.
# Instead, torch.where + element-wise ops are used throughout.

def _similarity_masked(beta, v_m, i_m, g_m):
    """Cosine-like similarity using pre-masked (zero-filled) tensors."""
    g_new = ((beta / 2) * i_m + 0.75) * g_m
    return (v_m * g_new).sum() / (v_m.norm() * g_new.norm() + 1e-8)

def _cv_masked(beta, i_m, g_m, n):
    """Coefficient of variation using pre-masked (zero-filled) tensors."""
    grad_m = ((beta / 2) * i_m + 0.75) * g_m
    mean_abs = grad_m.abs().sum() / n
    mean = grad_m.sum() / n
    var = (grad_m ** 2).sum() / n - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return std / (mean_abs + 1e-8)


def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    X1_sq = (X1 ** 2).sum(dim=1, keepdim=True)
    X2_sq = (X2 ** 2).sum(dim=1, keepdim=True)
    sq_dist = X1_sq - 2.0 * X1.matmul(X2.t()) + X2_sq.t()
    return sigma_f ** 2 * torch.exp(-0.5 / length_scale ** 2 * sq_dist)


def gp_posterior(X_train, Y_train, X_test,
                 length_scale=0.05, sigma_f=1.0, noise=1e-2):
    # Run on CPU to avoid CUDA device-side assert from cholesky on non-PD matrix
    orig_device = X_train.device
    X_train = X_train.float().cpu()
    Y_train = Y_train.float().cpu()
    X_test = X_test.float().cpu()

    N = X_train.size(0)
    M = X_test.size(0)

    K = rbf_kernel(X_train, X_train, length_scale, sigma_f)
    jitter = noise * torch.eye(N, dtype=torch.float32)
    K = K + jitter

    K_s = rbf_kernel(X_train, X_test, length_scale, sigma_f)
    K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f)
    K_ss = K_ss + noise * torch.eye(M, dtype=torch.float32)

    L = torch.linalg.cholesky(K)

    Yh = Y_train.view(N, 1)
    alpha = torch.cholesky_solve(Yh, L, upper=False)
    mu_s = K_s.t().matmul(alpha).view(M)

    v = torch.cholesky_solve(K_s, L, upper=False)
    cov_s = K_ss - K_s.t().matmul(v)
    diag = torch.clamp(torch.diagonal(cov_s, 0), min=1e-6)
    std_s = torch.sqrt(diag)

    return mu_s.to(orig_device).half(), std_s.to(orig_device).half()


def expected_improvement(mu, sigma, f_best, xi=0.0):
    mu     = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma  = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
    f_best = f_best.clone().detach().nan_to_num(nan=0.0) if isinstance(f_best, torch.Tensor) else torch.tensor(f_best, device=mu.device)

    imp = mu - f_best - xi
    Z   = imp / (sigma + 1e-8)

    dist  = Normal(0.0, 1.0)
    cdf_Z = dist.cdf(Z)
    pdf_Z = torch.exp(-0.5 * Z * Z) / math.sqrt(2 * math.pi)

    ei = imp * cdf_Z + sigma * pdf_Z
    return ei


def _bo_find_beta(X_obs, Y_obs, objective_fn, low_init, high_init, device):
    """Shared BO logic: GP posterior -> EI -> best beta. All small-tensor ops."""
    Y_obs = torch.nan_to_num(Y_obs, nan=0.0, posinf=0.0, neginf=0.0)

    best_idx = torch.argmax(Y_obs)
    y_best_obs = Y_obs[best_idx].item()
    best_beta = X_obs[best_idx].item()

    delta = 0.05
    left = max(best_beta - delta, low_init)
    right = min(best_beta + delta, high_init)
    X_test = torch.linspace(left, right, steps=15, device=device)

    mu_s, std_s = gp_posterior(
        X_obs.unsqueeze(-1), Y_obs,
        X_test.unsqueeze(-1)
    )

    f_best = Y_obs.max()
    ei = expected_improvement(mu_s, std_s, f_best)

    ei_best_idx = ei.argmax().item()
    beta_candidate = X_test[ei_best_idx].item()
    y_candidate = objective_fn(beta_candidate)

    return best_beta if y_best_obs > y_candidate else beta_candidate


def adjust_consistency(vec1, mask, grad_input, i, low_init=0.2, high_init=1.0):
    device = i.device
    dtype = grad_input.dtype
    zero = torch.zeros_like(grad_input, dtype=torch.float32)

    n_masked = mask.sum().item()
    if n_masked < 2:
        return torch.where(mask, grad_input, torch.zeros_like(grad_input))

    # Pre-mask tensors: non-masked positions become 0 (no boolean indexing)
    v_m = torch.where(mask, vec1.float(), zero)
    g_m = torch.where(mask, grad_input.float(), zero)
    i_m = torch.where(mask, i.float(), zero)

    X_obs = low_init + (high_init - low_init) * torch.rand(10, device=device)
    Y_obs = torch.stack([_similarity_masked(b, v_m, i_m, g_m) for b in X_obs])

    best_beta = _bo_find_beta(
        X_obs, Y_obs,
        objective_fn=lambda b: _similarity_masked(b, v_m, i_m, g_m).item(),
        low_init=low_init, high_init=high_init, device=device
    )

    adjusted = ((best_beta / 2) * i.float() + 0.75) * grad_input.float()
    return torch.where(mask, adjusted.to(dtype), torch.zeros_like(grad_input))


def adjust_consistency_t4(mask, grad_input, i, low_init=0.2, high_init=1.0):
    device = i.device
    dtype = grad_input.dtype
    zero = torch.zeros_like(grad_input, dtype=torch.float32)

    n_masked = mask.sum().item()
    if n_masked < 2:
        return torch.where(mask, grad_input, torch.zeros_like(grad_input))

    # Pre-mask tensors: non-masked positions become 0 (no boolean indexing)
    g_m = torch.where(mask, grad_input.float(), zero)
    i_m = torch.where(mask, i.float(), zero)
    n_t = torch.tensor(n_masked, dtype=torch.float32, device=device)

    X_obs = low_init + (high_init - low_init) * torch.rand(10, device=device)
    Y_obs = torch.stack([_cv_masked(b, i_m, g_m, n_t) for b in X_obs])

    best_beta = _bo_find_beta(
        X_obs, Y_obs,
        objective_fn=lambda b: _cv_masked(b, i_m, g_m, n_t).item(),
        low_init=low_init, high_init=high_init, device=device
    )

    adjusted = ((best_beta / 2) * i.float() + 0.75) * grad_input.float()
    return torch.where(mask, adjusted.to(dtype), torch.zeros_like(grad_input))


class Quant(torch.autograd.Function):
    grad_sum = {}
    grad_count = {}
    window_start = {}

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value, train_counter, layer_ref):
        ctx.min = min_value
        ctx.max = max_value
        ctx.train_counter = train_counter
        ctx.layer_ref = layer_ref
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        epoch = ctx.train_counter
        layer = ctx.layer_ref

        # shape guard: under AMP+DDP, shapes may rarely mismatch
        if i.shape != grad_input.shape:
            grad_input[grad_input.abs() > 0] *= 1.0  # STE fallback
            return grad_input, None, None, None, None

        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0

        if layer not in Quant.window_start:
            Quant.window_start[layer] = epoch
            Quant.grad_sum[layer] = torch.zeros_like(grad_input)
            Quant.grad_count[layer] = 0

        if Quant.grad_sum[layer].shape == grad_input.shape:
            Quant.grad_sum[layer].add_(grad_input.detach())
        else:
            Quant.grad_sum[layer] = grad_input.detach().clone()
        Quant.grad_count[layer] += 1

        if epoch >= Quant.window_start[layer] + 1:
            try:
                mask_t1 = (i > 0) & (i <= 1)
                mask_t2 = (i > 1) & (i <= 2)
                mask_t3 = (i > 2) & (i <= 3)
                mask_t4 = (i > 3) & (i <= 4)

                grad_t4 = adjust_consistency_t4(mask_t4, grad_input, i, 0.2, 1.0)
                grad_t3 = adjust_consistency(grad_t4, mask_t3, grad_input, i, 0.2, 1.0)
                grad_t2 = adjust_consistency(grad_t3, mask_t2, grad_input, i, 0.2, 1.0)
                grad_t1 = adjust_consistency(grad_t2, mask_t1, grad_input, i, 0.2, 1.0)

                # Use torch.where instead of boolean mask assignment to avoid IndexKernel.cu assert
                grad_input = torch.where(mask_t4, grad_t4, grad_input)
                grad_input = torch.where(mask_t3, grad_t3, grad_input)
                grad_input = torch.where(mask_t2, grad_t2, grad_input)
                grad_input = torch.where(mask_t1, grad_t1, grad_input)
            except Exception:
                pass  # STE fallback: keep grad_input as-is

            Quant.window_start[layer] = epoch
            Quant.grad_sum[layer].zero_()
            Quant.grad_count[layer] = 0

        return grad_input, None, None, None, None


class Multispike(nn.Module):
    def __init__(self, norm=T):
        super().__init__()
        self.norm = norm
        self.min_value = 0
        self.max_value = T
        self.train_counter = 0
        self.register_buffer("spike_count_int", torch.tensor(0.0))
        self.register_buffer("total_count_int", torch.tensor(0.0))

    def forward(self, inputs):
        out = Quant.apply(inputs, self.min_value, self.max_value, self.train_counter, self) / self.norm
        if not self.training:
            spike_sum = out.sum()
            self.spike_count_int += spike_sum.detach()
            self.total_count_int += out.numel()
        return out




def MS_conv_unit(in_channels, out_channels,kernel_size=1,padding=0,groups=1):
    return nn.Sequential(
        layer.SeqToANNContainer(
           encoder.SparseConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups,bias=True),
           encoder.SparseBatchNorm2d(out_channels)
        )
    )
class MS_ConvBlock(nn.Module):
    def __init__(self, dim,
        mlp_ratio=4.0):
        super().__init__()

        self.neuron1 = Multispike()
        self.conv1 = MS_conv_unit(dim, dim * mlp_ratio, 3, 1)

        self.neuron2 = Multispike()
        self.conv2 = MS_conv_unit(dim*mlp_ratio, dim, 3, 1)


    def forward(self, x, mask=None):
        short_cut = x
        x = self.neuron1(x)
        x = self.conv1(x)
        x = self.neuron2(x)
        x = self.conv2(x)
        x = x +short_cut
        return x

class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif =  Multispike()


        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = Multispike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N= x.shape

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()

        return x

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # TODO in_channel-> 2*in_channel->in_channel
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel*1.5), kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(int(in_channel*1.5)))
        self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel*1.5), out_channel, kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(out_channel))
    def forward(self, x):
        return self.conv2(self.conv1(x))
class RepConv2(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # TODO in_channel-> 2*in_channel->in_channel
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, int(in_channel), kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(int(in_channel)))
        self.conv2 = nn.Sequential(nn.Conv1d(int(in_channel), out_channel, kernel_size=1, stride=1,bias=False), nn.BatchNorm1d(out_channel))
    def forward(self, x):
        return self.conv2(self.conv1(x))

class MS_Attention_Conv_qkv_id(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.sr_ratio=sr_ratio

        self.head_lif = Multispike()

        # track 1: split convs
        self.q_conv = nn.Sequential(RepConv(dim,dim), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(RepConv(dim,dim), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(RepConv(dim,dim*sr_ratio), nn.BatchNorm1d(dim*sr_ratio))

        # track 2: merge (prefer) NOTE: need `chunk` in forward
        # self.qkv_conv = nn.Sequential(RepConv(dim,dim * 3), nn.BatchNorm2d(dim * 3))

        self.q_lif = Multispike()

        self.k_lif = Multispike()

        self.v_lif = Multispike()

        self.attn_lif = Multispike()

        self.proj_conv = nn.Sequential(RepConv(sr_ratio*dim,dim), nn.BatchNorm1d(dim))

    def forward(self, x):
        T, B, C, N = x.shape

        x = self.head_lif(x)

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv).reshape(T, B, C, N)

        q_conv_out = self.q_lif(q_conv_out)

        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4)

        k_conv_out = self.k_conv(x_for_qkv).reshape(T, B, C, N)

        k_conv_out = self.k_lif(k_conv_out)

        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4)

        v_conv_out = self.v_conv(x_for_qkv).reshape(T, B, self.sr_ratio*C, N)

        v_conv_out = self.v_lif(v_conv_out)

        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.sr_ratio*C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4)

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T, B, self.sr_ratio*C, N)
        x = self.attn_lif(x)

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, N)
        return x




class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,

    ):
        super().__init__()

        self.encode_conv = encoder.SparseConv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = encoder.SparseBatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = Multispike()

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W)

        return x

class MS_Block(nn.Module):
    def __init__(
            self,
            dim,
            choice,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,init_values=1e-6,finetune=False,
    ):
        super().__init__()
        self.model=choice
        if self.model=="base":
            self.rep_conv=RepConv2(dim,dim) #if have param==83M
        self.lif = Multispike()
        self.attn = MS_Attention_Conv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.finetune = finetune
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if self.finetune:
            self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        T, B, C, N = x.shape
        if self.model=="base":
            x= x + self.rep_conv(self.lif(x).flatten(0, 1)).reshape(T, B, C, N)
        # TODO: need channel-wise layer scale, init as 1e-6
        if self.finetune:
            x = x + self.drop_path(self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
            x = x + self.drop_path(self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x

class Spikmae(nn.Module):
    def __init__(self, T=1,choice=None,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128, 256, 512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), #norm_layer=nn.LayerNorm shaokun
        depths=8,
        sr_ratios=1,
        decoder_embed_dim=768,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_pix_loss=False, nb_classes=1000):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = 1

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,

        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[2],
                    choice=choice,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    finetune=False,
                )
                for j in range(depths)
            ]
        )

        self.norm = nn.BatchNorm1d(embed_dim[-1])
        self.downsample_raito =16

        num_patches = 196

        self.pos_embed = nn.Parameter(torch.zeros(1,  embed_dim[-1],num_patches), requires_grad=False)

        ## MAE decoder vit
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim,bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Try  larned decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=False, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels,bias=True)  # decoder to patch
        self.initialize_weights()

    def initialize_weights(self):
        num_patches=196
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[1], int(num_patches ** .5),
                                            cls_token=False)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed.transpose(1,0)).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(num_patches** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        num_patches=196
        T, N, _, _, _ = x.shape  # batch, length, dim
        L = num_patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # active is inverse mask
        active = torch.ones([N, L], device=x.device)
        active[:, len_keep:] = 0
        active = torch.gather(active, dim=1, index=ids_restore)

        return ids_keep, active, ids_restore

    def forward_encoder(self, x , mask_ratio=1.0):
        x  = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        # step1. Mask
        ids_keep, active, ids_restore = self.random_masking(x , mask_ratio)
        B,N=active.shape
        active_b1ff=active.reshape(B,1,14,14)

        encoder._cur_active = active_b1ff
        active_hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)
        active_hw = active_hw.unsqueeze(0)
        masked_bchw = x * active_hw
        x = masked_bchw
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        x = x.flatten(3)
        for blk in self.block3:
            x = blk(x)

        x = x.mean(0)
        x = self.norm(x).transpose(-1, -2).contiguous()
        return x, active,ids_restore,active_hw

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        B, N, C = x.shape
        x = self.decoder_embed(x)  # B, N, C
        # append mask tokens to sequence
        # ids_restore#1,196
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_
#
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        inp, rec = self.patchify(imgs), pred # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)
        non_active = mask.logical_not().int().view(mask.shape[0], -1)  # (B, 1, f, f) => (B, L)
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
        return recon_loss,mean,var

    def forward(self, imgs, mask_ratio=0.5,vis=False):

        latent, active, ids_restore,active_hw = self.forward_encoder(imgs, mask_ratio)
        rec = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        recon_loss,mean,var = self.forward_loss(imgs, rec, active)
        if vis:
            masked_bchw = imgs * active_hw.flatten(0,1)
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_hw.flatten(0,1).bool(), imgs, rec_bchw)
            return imgs, masked_bchw, rec_or_inp
        else:
            return recon_loss


def spikmae_12_512(**kwargs):
    model = Spikmae(
        T=1,
        choice="base",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[128,256,512],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        **kwargs)
    return model
def spikmae_12_768(**kwargs):
    model = Spikmae(
        T=1,
        choice="large",
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[192,384,768],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=12,
        sr_ratios=1, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        **kwargs)
    return model




if __name__ == "__main__":
    model = spikmae_12_768()
    x=torch.randn(1,3,224,224)
    loss = model(x,mask_ratio=0.50)
    print('loss',loss)
    torchinfo.summary(model, (1, 3, 224, 224))
    print(f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
