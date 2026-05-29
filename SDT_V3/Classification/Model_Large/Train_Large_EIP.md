### Train Large Model with EIP

Environment: `lava` (`/home/kyccj/envs/lava/bin/`)
Data: `/srv2/kyccj/data` (ImageNet-1K, train/val)

cuDNN 충돌 방지를 위해 LD_LIBRARY_PATH 설정 필요:
```shell
export LD_LIBRARY_PATH=/home/kyccj/envs/lava/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/kyccj/envs/lava/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
```

---

### 1. Pretrain (MAE + EIP)

GPU 메모리 여유가 있을 때 (batch 256):
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/kyccj/envs/lava/bin/torchrun --standalone --nproc_per_node=8 \
  main_pretrain.py \
  --batch_size 256 \
  --blr 1.5e-4 \
  --warmup_epochs 20 \
  --epochs 200 \
  --model spikmae_12_512 \
  --mask_ratio 0.50 \
  --data_path /srv2/kyccj/data \
  --output_dir /srv2/kyccj/pretrain_eip \
  --log_dir /srv2/kyccj/pretrain_eip \
  --eip --eip_const 1e-8 --eip_alpha 3.0
```

GPU 메모리 부족할 때 (batch 64 + gradient accumulation, effective batch size 동일):
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/kyccj/envs/lava/bin/torchrun --standalone --nproc_per_node=8 \
  main_pretrain.py \
  --batch_size 64 \
  --accum_iter 4 \
  --blr 1.5e-4 \
  --warmup_epochs 20 \
  --epochs 200 \
  --model spikmae_12_512 \
  --mask_ratio 0.50 \
  --data_path /srv2/kyccj/data \
  --output_dir /srv2/kyccj/pretrain_eip \
  --log_dir /srv2/kyccj/pretrain_eip \
  --eip --eip_const 1e-8 --eip_alpha 3.0
```

---

### 2. Finetune (EIP only, FD/DFE off)

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/kyccj/envs/lava/bin/torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 100 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --layer_decay 0.75 \
  --finetune /srv2/kyccj/pretrain_eip/checkpoint-199.pth \
  --epochs 150 \
  --drop_path 0.1 \
  --model spikformer12_768 \
  --data_path /srv2/kyccj/data \
  --output_dir /srv2/kyccj/finetune_eip \
  --log_dir /srv2/kyccj/finetune_eip \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --dist_eval \
  --eip --eip_const 1e-8 --eip_alpha 3.0 \
  --no_fd_loss \
  --no_dfe_loss
```

---

### 3. Distillation (EIP only)

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/kyccj/envs/lava/bin/torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 196 \
  --blr 1e-3 \
  --warmup_epochs 5 \
  --epochs 100 \
  --drop_path 0.1 \
  --finetune /srv2/kyccj/finetune_eip/checkpoint-best.pth \
  --model spikformer12_512 \
  --data_path /srv2/kyccj/data \
  --output_dir /srv2/kyccj/distill_eip \
  --log_dir /srv2/kyccj/distill_eip \
  --dist_eval \
  --time_steps 1 \
  --kd \
  --input_size 224 \
  --teacher_model caformer_b36_in21ft1k \
  --reprob 0.25 \
  --mixup 0.5 \
  --cutmix 1.0 \
  --distillation_type hard \
  --eip --eip_const 1e-8 --eip_alpha 3.0 \
  --no_fd_loss \
  --no_dfe_loss
```

---

### EIP Flag 정리

| Flag | Description | Default |
|------|-------------|---------|
| `--eip` | EIP regularization 활성화 | False |
| `--eip_const` | EIP loss weight | 1e-8 |
| `--eip_alpha` | softmax temperature | 3.0 |
| `--no_fd_loss` | FD loss 비활성화 | False (켜짐) |
| `--no_dfe_loss` | DFE loss 비활성화 | False (켜짐) |

### Spike Count Logging

- Pretrain: epoch마다 콘솔에 `Total spikes per sample` 출력 + tensorboard `spike_count`
- Finetune: eval 시 `Total spikes`, `Encod spikes` 출력
