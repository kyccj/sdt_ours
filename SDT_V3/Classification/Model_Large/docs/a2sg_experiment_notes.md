# A2SG (Adaptive Approximation to Spike Gradient) 실험 정리

---

## 1. 모델 구현

### 1.1 A2SG 핵심 메서드 개요

**논문**: *Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training* (IEEE T-PAMI 2025)

A2SG는 Multi-spike Quantization의 backward pass에서 **Bayesian Optimization (BO)**을 활용하여 gradient를 조정하는 방법입니다.

#### 핵심 구성요소

| 구성요소 | 설명 |
|---------|------|
| `Quant` (autograd.Function) | Forward: `round(clamp(x, 0, T))`, Backward: STE + BO gradient 조정 |
| `Multispike` (nn.Module) | `Quant.apply(x) / norm`, spike counting 포함 |
| Gaussian Process (GP) | RBF 커널 + Cholesky 분해 기반 posterior 추정 |
| Expected Improvement (EI) | GP posterior로부터 최적 beta 탐색을 위한 acquisition function |

#### Backward Pass 동작 원리

1. **Threshold Region 분할**: 입력 `i`를 4개 구간으로 분할
   - `t1`: (0, 1], `t2`: (1, 2], `t3`: (2, 3], `t4`: (3, 4]

2. **Cascading Gradient 조정** (t4 → t3 → t2 → t1):
   - `t4`: `adjust_consistency_t4` — CV(Coefficient of Variation) 최대화
   - `t3`: `adjust_consistency` — t4의 결과를 reference로 cosine similarity 최대화
   - `t2`: `adjust_consistency` — t3의 결과를 reference로 cosine similarity 최대화
   - `t1`: `adjust_consistency` — t2의 결과를 reference로 cosine similarity 최대화

3. **BO 과정** (각 region):
   - 10개의 random beta 샘플링 → objective 평가
   - 최적 beta 주변에서 GP posterior 추정
   - EI로 15개 candidate에서 최적 beta 선택
   - 최종 gradient: `((beta/2) * i + 0.75) * grad_input`

4. **Warmup**: `train_counter` 기반으로 1 epoch warmup 후 BO 활성화

### 1.2 생성한 파일

#### `spikformer_a2sg.py` (Finetune 모델)

- **기반**: `spikformer.py` (기존 ms 모드 finetune 모델)
- **변경점**: `multispike` autograd.Function → `Quant` 클래스로 교체, `Multispike` 모듈에 BO gradient 로직 포함
- **아키텍처**: 4-stage hierarchical SNN Transformer (T=4)
  - Stage 1-2: ConvBlock (3x3 conv + BN)
  - Stage 3-4: Transformer Block (MS_Attention_Conv_qkv_id: K^T·V linear attention)
- **모델 팩토리**:
  - `spikformer12_512()`: dim=512, depth=[1,1,10], 55M params
  - `spikformer12_768()`: dim=768, depth=[1,1,10], 171M params
- `nn.Conv2d` / `nn.BatchNorm2d` 사용 (standard)

#### `MAE_SDT_a2sg.py` (Pretrain 모델)

- **기반**: `MAE_SDT.py` (기존 ms 모드 pretrain MAE)
- **변경점**: 동일한 `Quant`/`Multispike` 교체
- **아키텍처**: Spike-Masked Autoencoder
  - Encoder: SNN Transformer (`encoder.SparseConv2d`, `encoder.SparseBatchNorm2d`)
  - Decoder: ANN Transformer (standard `nn.Linear`)
  - Masked image modeling 방식 pretrain
- **모델 팩토리**:
  - `spikmae_12_512()`: embed_dim=512
  - `spikmae_12_768()`: embed_dim=768

### 1.3 수정한 파일

#### `main_pretrain.py`

```python
# 추가된 import
import MAE_SDT_a2sg

# 모델 생성 분기 추가
elif args.model_mode == "a2sg":
    model = MAE_SDT_a2sg.__dict__[args.model]()

# train_counter 업데이트 (training loop 내)
if args.model_mode == "a2sg":
    for m in model.modules():
        if isinstance(m, MAE_SDT_a2sg.Multispike):
            m.train_counter = epoch
```

#### `main_finetune.py`

```python
# 추가된 import
import spikformer_a2sg

# 모델 생성 분기 추가
elif args.model_mode == "a2sg":
    model = spikformer_a2sg.__dict__[args.model](kd=args.kd)

# train_counter 업데이트 (training loop 내)
if args.model_mode == "a2sg":
    for m in model.modules():
        if isinstance(m, spikformer_a2sg.Multispike):
            m.train_counter = epoch
```

#### `engine_finetune.py`

```python
# 추가된 import
import spikformer_a2sg

# evaluate 함수 내 spike counting 분기 추가
elif model_mode == "a2sg":
    for m in model.modules():
        if isinstance(m, spikformer_a2sg.Multispike):
            total_spike_count += m.spike_count_int.item()
            m.spike_count_int.zero_()
```

---

## 2. 실험 스크립트

### `run_a2sg_768.sh`

**위치**: `SDT_V3/Classification/Model_Large/run_a2sg_768.sh`

2-stage 파이프라인 (Pretrain → Finetune):

```bash
#!/bin/bash
set -e
source /home/kyccj/anaconda3/etc/profile.d/conda.sh
conda activate sdtv3_cls

GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=8
DATA_PATH="/media/hdd1/kyccj/data/ImageNet_down"  # ImageNet-1K
```

#### Stage 1: Pretrain (Spike-Masked Autoencoder)

| 파라미터 | 값 |
|---------|-----|
| 모델 | `spikmae_12_768` |
| 모드 | `a2sg` |
| batch_size (per GPU) | 128 |
| effective batch_size | 128 × 8 = 1024 |
| base lr | 1.5e-4 |
| warmup epochs | 20 |
| total epochs | 200 |
| mask ratio | 0.50 |
| weight decay | 0.05 |
| optimizer | AdamW (β=(0.9, 0.95), ε=1e-4) |

#### Stage 2: Finetune (Classification)

| 파라미터 | 값 |
|---------|-----|
| 모델 | `spikformer12_768` |
| 모드 | `a2sg` |
| batch_size (per GPU) | 100 |
| effective batch_size | 100 × 8 = 800 |
| base lr | 6e-4 |
| warmup epochs | 10 |
| total epochs | 150 |
| layer decay | 0.75 |
| drop path | 0.1 |
| reprob (random erasing) | 0.25 |
| mixup | 0.8 |
| cutmix | 1.0 |
| dist_eval | True |

---

## 3. 실험 환경 및 진행 상황

### 3.1 하드웨어

| 항목 | 사양 |
|------|------|
| GPU | 8× NVIDIA RTX A6000 (48GB each) |
| CUDA | 12.2 |
| Driver | 535.183.01 |

### 3.2 소프트웨어 (conda env: `sdtv3_cls`)

| 패키지 | 버전 |
|--------|------|
| Python | 3.9 |
| PyTorch | 2.0.0+cu117 |
| spikingjelly | 0.0.0.0.12 |
| timm | 0.6.12 |
| einops | latest |
| torchinfo | latest |
| numpy | 1.26.4 (< 2.0 호환성) |
| tensorboard | latest |

### 3.3 데이터셋

- **ImageNet-1K**: `/media/hdd1/kyccj/data/ImageNet_down`
- Train: 1000 classes, ~152GB
- Val: 50,000 images

### 3.4 NCCL 설정

8-GPU 학습 시 NCCL P2P 통신에서 hang이 발생하여 아래 환경변수 필수:

```bash
export NCCL_P2P_DISABLE=1
```

이 설정 없이는 `torch.distributed.init_process_group()` 이후 barrier에서 무한 대기 발생.
2-GPU에서는 문제 없으나 4-GPU 이상에서 재현됨.

### 3.5 실험 진행 상황

- **시작 시각**: 2026-02-27 20:02
- **현재 상태**: Stage 1 (Pretrain) 진행 중
- **Epoch 0 통계**:
  - 1 iteration ≈ 1.28초
  - 1 epoch ≈ 27분 (1251 iterations)
  - GPU 메모리: max 41,847 MB per GPU (48GB 중)
  - Loss: 1.099 → 안정적 감소 중
- **예상 소요시간**: Pretrain 200 epochs ≈ 90시간 (3.75일)
- **로그 위치**: `outputs/a2sg_768/train.log`
- **체크포인트 저장**: 매 20 epoch + 마지막 epoch

### 3.6 모니터링 명령어

```bash
# 로그 실시간 확인
tail -f outputs/a2sg_768/train.log

# GPU 사용량 확인
nvidia-smi

# 프로세스 확인
ps aux | grep main_pretrain

# debug log 확인
cat outputs/a2sg_768/pretrain/debug.log

# pretrain log.txt (에포크 단위 기록)
tail -f outputs/a2sg_768/pretrain/log.txt
```
