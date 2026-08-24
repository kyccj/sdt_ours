"""Spike regularization with loss-ratio lambda control (port of the TF lib_snn method).

Three parts, all of them here:

  1. the regularization term itself -- ||spike * sc_rate||_2 per neuron layer, with a
     custom backward that sends gradient to neurons that did not fire.  Standard L2 has
     d||x||/dx = x/||x||, which is zero wherever the neuron was silent, so a silent
     neuron never learns to stay silent.  The modified backward uses sc_rate/||x||
     instead, which is non-zero everywhere.  (TF: lib_snn/layers.py l2_norm_wta_rev)

  2. loss-ratio control of lambda -- hold the regularization term at a fixed fraction
     rho of the task loss.  reg = lambda * R, so lambda = rho * L_task / R solves it in
     one step, once per epoch.  No gain, no target spike count, nothing known in
     advance.  (TF: lib_snn/proc.py, the reg_spike_loss_ratio block)

  3. the brake -- lambda growth is capped at growth_cap per epoch at ALL epochs, which
     bounds the R -> 0 => lambda -> inf positive feedback that kills a run outright; and
     inside an early window lambda decays instead of following the formula while the
     spike total sits below a floor.  The growth cap is the part with direct evidence
     (aggressive rho: collapse -> single-digit loss); the early-window floor has not
     been established causally, so it is off by default.  (TF: flags.py reg_spike_lr_brake)

Differences from the TF implementation, which are properties of SDT-V3, not choices:

  * the neuron (Multispike) emits floor(clamp(x,0,D)+0.5)/D -- a graded value, not a
    binary spike, and there is no loop over time steps (T=1).  The per-time-step
    accounting that TF needed disappears: R is complete after one forward pass.
  * sc_rate defaults to 1 (uniform).  In TF, 1-softmax measured 0.9999 with ~0 spread
    and reproduced the softmax results 4/4, i.e. the softmax was vestigial.  Setting it
    to 1 keeps the modified backward -- the part that does the work -- and avoids
    materializing a second tensor the size of every activation.  'wta_rev' restores the
    softmax weighting if it ever needs to be checked here.

Usage:

    import spike_reg
    names = spike_reg.convert_multispike(model, cfg)     # after loading a checkpoint
    ctl = spike_reg.LossRatioController(cfg)
    ...
    spike_reg.REG.lam = lam                              # set once per epoch
    spike_reg.REG.reset()                                # once per batch, before forward
    out = model(x); loss = criterion(...) + spike_reg.REG.lam * spike_reg.REG.total()
    ...
    lam, info = ctl.step(epoch, task_loss, R, spikes)    # once per epoch, after training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config

class SpikeRegConfig:
    """Everything the method needs, filled from argparse (see from_args)."""

    def __init__(self):
        self.enabled = False
        self.sc_rate = "one"        # 'one' | 'wta_rev'
        self.alpha = 7.0            # softmax temperature, 'wta_rev' only
        self.rho = 5.8e-4           # target reg/task loss ratio
        self.start_ep = 0           # lambda stays 0 before this epoch
        self.lam_max = 1e-4         # sanity bound only
        self.growth_cap = 1.5       # max lambda growth per epoch (0 disables)
        self.floor = 0.0            # early-window spike floor (0 disables)
        self.floor_ep = 30          # width of the early window, in epochs
        self.floor_decay = 0.5      # lambda multiplier while below the floor
        self.skip = ("lif",)        # module names NOT regularized

    @staticmethod
    def from_args(args):
        cfg = SpikeRegConfig()
        cfg.enabled = bool(getattr(args, "reg_spike", False))
        cfg.sc_rate = getattr(args, "reg_sc_rate", "one")
        cfg.alpha = getattr(args, "reg_alpha", 7.0)
        cfg.rho = getattr(args, "reg_rho", 5.8e-4)
        cfg.start_ep = getattr(args, "reg_start_ep", 0)
        cfg.lam_max = getattr(args, "reg_lam_max", 1e-4)
        cfg.growth_cap = getattr(args, "reg_growth_cap", 1.5)
        cfg.floor = getattr(args, "reg_brake_floor", 0.0)
        cfg.floor_ep = getattr(args, "reg_brake_ep", 30)
        cfg.floor_decay = getattr(args, "reg_brake_decay", 0.5)
        skip = getattr(args, "reg_skip", "lif")
        cfg.skip = tuple(s for s in skip.split(",") if s)
        return cfg

    def __repr__(self):
        return ("SpikeRegConfig(enabled={0.enabled}, sc_rate={0.sc_rate}, alpha={0.alpha}, "
                "rho={0.rho:.3g}, start_ep={0.start_ep}, growth_cap={0.growth_cap}, "
                "floor={0.floor}, floor_ep={0.floor_ep}, skip={0.skip})".format(self))


# ---------------------------------------------------------------- the reg term

class _L2NormWtaRev(torch.autograd.Function):
    """||x||_2 forward; d/dx = sc_rate/||x|| backward instead of x/||x||.

    sc_rate is a tensor broadcastable to x, or a 0-dim tensor for the uniform case.
    Runs in fp32 even under autocast -- a sum of squares over a whole activation
    tensor overflows fp16 quickly.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, sc_rate):
        nrm = torch.sqrt(torch.sum(x * x))
        ctx.save_for_backward(sc_rate, nrm)
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        ctx.x_device = x.device
        return nrm

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_out):
        sc_rate, nrm = ctx.saved_tensors
        # nrm == 0 means the whole tensor was zero -> no gradient, same as the TF
        # version.  Done with where() rather than an if, which would force a
        # device sync on every layer of every backward pass.
        safe = torch.where(nrm > 0, nrm, torch.ones_like(nrm))
        g = grad_out * sc_rate / safe * (nrm > 0).to(grad_out.dtype)
        if g.shape != ctx.x_shape:
            g = g.expand(ctx.x_shape).contiguous()
        return g, None


class SpikeRegState:
    """Per-batch collector.  One global instance, REG, shared by every wrapped neuron."""

    def __init__(self):
        self.enabled = False
        self.lam = 0.0            # python float, set once per epoch
        self.parts = []           # per-layer norms of the current batch
        self.spike_parts = []     # per-layer spike sums, detached, still on device
        self.n_layers = 0

    def reset(self):
        self.parts = []
        self.spike_parts = []
        self.n_layers = 0

    def add(self, norm_val, spike_sum):
        self.parts.append(norm_val)
        self.spike_parts.append(spike_sum)
        self.n_layers += 1

    def total(self):
        """R: the raw, pre-lambda regularization value of this batch."""
        if not self.parts:
            return 0.0
        return torch.stack(self.parts).sum()

    def spikes(self):
        """Graded spike total of this batch.  Kept on device -- .item() it once per
        batch, not once per layer, or every neuron costs a device sync."""
        if not self.spike_parts:
            return 0.0
        return torch.stack(self.spike_parts).sum()


REG = SpikeRegState()


class EIPMultispike(nn.Module):
    """Wraps a Multispike neuron, adds the regularization term and spike counting.

    Wrapping instead of replacing keeps the inner module untouched, so this works on
    the ms, s_direct and a2sg model files alike -- including a2sg, whose Quant backward
    is left exactly as it is.

    Note for a2sg: the inner Multispike owns persistent buffers, so wrapping renames its
    state_dict keys (x.spike_count_int -> x.inner.spike_count_int).  Wrap AFTER loading
    a checkpoint, or remap the keys.  The plain ms Multispike has no buffers at all, so
    nothing changes there.
    """

    def __init__(self, inner, cfg, state=None):
        super().__init__()
        self.inner = inner
        self.cfg = cfg
        self.state = state if state is not None else REG
        self.norm = float(getattr(inner, "norm", 1.0)) or 1.0
        # non-persistent: keeps state_dict identical to the unwrapped model
        self.register_buffer("spike_count_int", torch.tensor(0.0), persistent=False)
        self.register_buffer("total_count_int", torch.tensor(0.0), persistent=False)
        self.register_buffer("_one", torch.tensor(1.0), persistent=False)

    def _sc_rate(self, out):
        if self.cfg.sc_rate == "one":
            return self._one.to(out.dtype)
        # 'wta_rev': 1 - softmax(spike_count/alpha) over everything but the sample axis.
        # out is [T,B,C,H,W] or [T,B,C,N]; T and B together are the sample axis.
        sc = (out * self.norm).detach()
        flat = sc.flatten(0, 1).flatten(1)
        rate = 1.0 - F.softmax(flat / self.cfg.alpha, dim=1)
        return rate.view_as(out)

    def forward(self, x):
        out = self.inner(x)

        if self.training:
            if self.state.enabled:
                rate = self._sc_rate(out)
                x_reg = out if self.cfg.sc_rate == "one" else out * rate
                nrm = _L2NormWtaRev.apply(x_reg, rate)
                self.state.add(nrm, out.detach().sum())
        else:
            self.spike_count_int += out.detach().sum()
            self.total_count_int += out.numel()

        return out

    def extra_repr(self):
        return "sc_rate={}, norm={}".format(self.cfg.sc_rate, self.norm)


_NEURON_CLASS_NAMES = ("Multispike", "Multispike_first")


def convert_multispike(model, cfg, state=None, verbose=True):
    """Replace every Multispike in `model` with EIPMultispike.  Returns the names wrapped.

    Names in cfg.skip are left alone -- by default the head neuron ('lif'), which is the
    output layer and is not part of what the method regularizes (TF: loc == 'HID' only).
    The first downsampling block has no neuron at all (first_layer=True), so the input
    encoding is already outside the scope.
    """
    wrapped, skipped = [], []

    def walk(module, prefix):
        for name, child in list(module.named_children()):
            full = "{}.{}".format(prefix, name) if prefix else name
            if type(child).__name__ in _NEURON_CLASS_NAMES:
                # matched on the FULL dotted name: 'lif' is the head neuron, while
                # block3.*.lif are hidden neurons that must stay regularized
                if full in cfg.skip:
                    skipped.append(full)
                else:
                    setattr(module, name, EIPMultispike(child, cfg, state))
                    wrapped.append(full)
            else:
                walk(child, full)

    walk(model, "")
    if verbose:
        print("[spike_reg] wrapped {} neurons, skipped {}: {}".format(
            len(wrapped), len(skipped), skipped))
    return wrapped


# ---------------------------------------------------------------- lambda control

class LossRatioController:
    """lambda <- rho * L_task / R, once per epoch, with the brake on top.

    All inputs are expected to be already averaged across ranks (MetricLogger's
    global_avg does this), so every rank computes the same lambda and no broadcast is
    needed.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.lam = 0.0
        self.s_first = None      # spike total of the first regularized epoch

    def step(self, epoch, task_loss, R, spikes):
        cfg = self.cfg
        cur = self.lam
        info = {"lam": cur, "R": R, "spikes": spikes, "s_ratio": float("nan"), "brake_on": 0.0}

        if epoch < cfg.start_ep or R <= 0.0:
            self.lam = 0.0
            info["lam"] = 0.0
            return 0.0, info

        new = cfg.rho * max(task_loss, 1e-8) / R
        new = min(max(new, 0.0), cfg.lam_max)

        if spikes > 0.0:
            if self.s_first is None:
                self.s_first = spikes
            else:
                s_ratio = spikes / self.s_first
                info["s_ratio"] = s_ratio
                in_window = (cfg.floor > 0.0
                             and epoch < cfg.start_ep + cfg.floor_ep)
                if in_window and s_ratio < cfg.floor:
                    # below the floor: back off instead of following the formula
                    new = cur * cfg.floor_decay
                    info["brake_on"] = 1.0
                elif cfg.growth_cap > 0.0 and cur > 0.0:
                    # growth cap at all epochs: bounds R -> 0 => lambda -> inf
                    new = min(new, cur * cfg.growth_cap)

        self.lam = new
        info["lam"] = new
        return new, info


def add_args(parser):
    """Register the flags on main_finetune's argparse."""
    g = parser.add_argument_group("spike regularization")
    g.add_argument("--reg_spike", action="store_true", default=False,
                   help="enable spike regularization with loss-ratio lambda control")
    g.add_argument("--reg_sc_rate", default="one", choices=["one", "wta_rev"],
                   help="neuron weighting: uniform (default) or 1-softmax(spike/alpha)")
    g.add_argument("--reg_alpha", default=7.0, type=float,
                   help="softmax temperature, --reg_sc_rate wta_rev only")
    g.add_argument("--reg_rho", default=5.8e-4, type=float,
                   help="target reg/task loss ratio. NOTE the CIFAR value is not "
                        "transferable -- measure R and L_task here first")
    g.add_argument("--reg_start_ep", default=0, type=int,
                   help="lambda stays 0 before this epoch")
    g.add_argument("--reg_lam_max", default=1e-4, type=float, help="lambda sanity bound")
    g.add_argument("--reg_growth_cap", default=1.5, type=float,
                   help="max lambda growth factor per epoch at all epochs (0 disables)")
    g.add_argument("--reg_brake_floor", default=0.0, type=float,
                   help="early-window floor on S(e)/S(first); 0 disables. 0.22 was the "
                        "CIFAR setting")
    g.add_argument("--reg_brake_ep", default=30, type=int, help="width of that window")
    g.add_argument("--reg_brake_decay", default=0.5, type=float,
                   help="lambda multiplier while below the floor")
    g.add_argument("--reg_skip", default="lif", type=str,
                   help="comma-separated module names left unregularized")
    return parser
