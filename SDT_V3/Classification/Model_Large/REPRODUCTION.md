# 171M (E-SpikeFormer / SDT-v3 large) 재현 기록 및 실행 절차

이 문서는 **작업을 이어받는 사람/에이전트를 위한 것**이다. 무엇이 돌았고, 어떤 수치가
나왔고, upstream 코드의 어떤 함정을 이미 고쳤는지를 담는다. 새 실험을 시작하기 전에
"이미 배제된 가설" 절을 먼저 읽을 것.

논문: Scaling Spike-driven Transformer with Efficient Spike Firing Approximation
Training (TPAMI 2025, arXiv 2411.16061)

---

## 0. 가장 먼저 알아야 할 것 — DFE / FD 는 이 저장소 소유자의 제안 방법이다

`spikformer_s_direct.py` 에는 `DFE_BatchNorm2d` 와 `fd_loss` 가 있다.
**이것은 SDT-v3 논문의 일부가 아니라 저장소 소유자가 추가한 자체 방법이다.**

그리고 현재 **동작하지 않는다(inert)**:
- 둘 다 `self.extra_losses` 에 append 만 하고, 이 리스트를 읽는 코드가 어디에도 없다
  (`grep -rn extra_losses` 로 확인 가능 — 전부 append/초기화, 소비처 0건)
- `Multispike_first.forward` 는 `Multispike.forward` 와 수치적으로 동일
- `DFE_BatchNorm2d.forward` 는 `super().forward(x)` 를 그대로 반환

즉 **코드를 그대로 돌리면 그게 논문 baseline이다.** 이 클래스들이 파일에 있다는 이유로
"논문 설정과 다르다" 고 판단하지 말 것.

실제로 켜려면 `engine_finetune.py` 에서 `model.modules()` 를 순회해 `extra_losses` 를
모아 학습 loss 에 더하는 코드가 필요하다. 그건 baseline 과 비교할 **별도 실험 arm** 이고
새 run 이 필요하다.

ASG / A2SG 는 이 코드에 없다.

---

## 1. 결과 요약 (4x RTX A6000 48GB)

| 단계 | 설정 | epochs | 소요 | val top-1 |
|---|---|---|---|---|
| MAE pretrain | D=4, eff batch 2048 | 200 | 8.0일 | — |
| finetune | D=8, eff 800 | 150 | 6.2일 | **74.59%** (ep149) |
| + distillation | D=8, eff 800 | 100 | 5.4일 | **82.78%** (ep94) |
| finetune | D=4, eff 800 | 150 | 6.2일 | 74.38% (ep149) |
| + distillation | D=4, eff 1568 | 100 | 5.4일 | 82.52% (ep98) |

**최고 결과 82.78%** = `171M_kd/checkpoint-best.pth`

논문 Table 1 (173M):

| Step (TxD) | 해상도 | Top-1 |
|---|---|---|
| 1x4 | 224 | 84.7 |
| **1x8** | **224** | **85.1**  <- 직접 비교 대상 |
| 1x8 | 384 | 86.2 |

**격차 2.32%p** (82.78 vs 85.1).

`86.2*` 의 `*` 는 표 캡션에 따르면 "input crops are enlarged to 384x384 in inference".
단, 이는 384로 별도 학습한 모델이다 (repo 도 `171M_1x8_384` 를 따로 배포). 우리 224 모델을
384로 평가하면 **80.90% 로 떨어진다** (검증함). 즉 86.2 는 직접 비교 대상이 아니다.

표기법: `T x D` 에서 T=timestep, D=학습 시 integer activation 상한 (논문 Table 1 캡션).
D 는 코드의 모듈 상수로, pretrain 은 `MAE_SDT.py` 의 `T=4`, finetune 은
`spikformer_s_direct.py` 의 `T`(= `SDT_D` env, 기본 8).

---

## 2. 이미 배제된 가설 — 다시 돌리지 말 것

격차 2.32%p 의 원인을 찾으려고 아래를 전부 실험했고, 모두 원인이 아니었다.

| 가설 | 결과 | 비용 |
|---|---|---|
| 유효배치를 문서값 1568 로 (lr 6.13e-3) | 82.52% (**-0.26%p**) | 5.4일 |
| EMA 가중치 사용 | 최고 82.40% (raw 82.52 보다 낮음) | 공짜 |
| 384 해상도 추론 | 80.90% (**-1.88%p**) | 수분 |
| finetune D=8 -> D=4 (pretrain D와 일치시킴) | 74.38% (**-0.21%p**) | 6.2일 |

참고로 D=4 KD 는 14 epoch 까지 D=8 KD 와 ±1%p 내로 겹쳐서 중단했다
(`171M_kd_d4/checkpoint-best.pth`, epoch 13 에서 resume 가능).

**남은 후보:**
1. **KD 단계의 `layer_decay` 가 1.0** — finetune 은 0.75 인데 KD 는 문서 Distillation
   블록에 해당 인자가 없어 기본값 1.0 이 들어갔다. 171M 규모에서 lr 3.13e-3 에
   layer-wise decay 없이 하위 레이어까지 전부 높은 lr 을 받는다. **가장 유력.**
   검증법: KD 를 `--layer_decay 0.75` 로 재실행 (5.4일)
2. **논문 Supplementary S5** — MIM mask ratio / decoder width·depth / distillation 세부
   설정이 전부 거기 있는데 arXiv PDF 에 미포함이다. 대조 불가.
3. 하드웨어 차이 (논문 8xA100-40G vs 4xA6000)

---

## 3. upstream 함정 4개 (이 브랜치에서 수정됨)

`main` (c479e6b) 으로 돌리면 아래에 걸린다. 반드시 이 수정이 들어간 상태로 쓸 것.

### 3.1 `spikformer_s_direct.py` — embed_dim 196 (치명적)
`MAE_SDT.spikmae_12_768` 은 `embed_dim=[192,384,768]` 인데
`spikformer12_768` 은 `[196,384,768]` 이었다. pretrain 체크포인트를 finetune 모델에
넣으면 **shape mismatch 37개로 즉시 죽는다.** `load_state_dict(strict=False)` 는
키 누락만 무시하고 shape 불일치는 예외를 던진다.

192 로 고친 후: shape 충돌 0개, 파라미터 99.5% 로드. 채워지지 않는 0.5% 는
classifier head / `block3.*.layer_scale*` / `spike_count_int` 버퍼로, 새로 학습되는 게 정상.

### 3.2 `main_finetune.py` — 체크포인트 폭증
`best_acc = 0` 이 초기화 후 한 번도 갱신되지 않아 `acc1 > best_acc` 가 매 epoch 참이 되고,
2GB 체크포인트를 **150개(287GB)** 저장했다. `best_acc` 를 갱신하고 `checkpoint-best.pth`
하나에 덮어쓰도록 수정. 수정 후 한 run 이 7.7GB.

같은 파일에서 EMA 도 처리했다. `MODEL_EMA=True` 로 매 step 173M EMA 를 갱신하면서
평가도 저장도 하지 않았다 (`misc.save_model` 은 `model` 인자를 무시하고
`model_without_ddp` 를 저장한다). 이제 EMA 도 평가/저장한다.

### 3.3 `metaformer.py` — teacher 가중치 경로
`/public/liguoqi/qxr/classification/caformer_b36_in21ft1k.pth` (저자 서버) 하드코딩이고
URL 로딩은 주석 처리돼 있어서 `--kd` 를 주면 즉시 FileNotFoundError.
로컬 경로 -> 공식 URL fallback 으로 수정.

teacher 정상 여부 검증법: val 1000장에서 top-1 이 86% 근처여야 한다 (실측 86.0%).

### 3.4 `engine_pretrain.py` — `functional` 미import
`functional.reset_net(model)` 을 import 없이 호출해서 **모든 rank 가 첫 step 에서
NameError 로 죽는다.** 모델의 `Multispike` 는 상태가 없는 autograd Function 이라
reset 이 불필요하다 (`engine_finetune.py` 는 이미 이 호출이 주석 처리돼 있다). 주석 처리.

---

## 4. 환경

```bash
conda create -n sdt_large python=3.9 -c conda-forge --override-channels -y
conda activate sdt_large
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1
pip install timm==0.5.4 spikingjelly==0.0.0.0.12 torchinfo==1.8.0 tensorboard==2.19.0 \
            einops==0.8.1 matplotlib==3.9.4 numpy==2.0.2 pillow==11.0.0
```

**timm 은 0.5.4 고정.** 상위 버전은 `timm.models.layers` / `timm.models.registry`
경로가 달라져 import 부터 깨진다.

---

## 5. 데이터셋 — val 폴더 함정

`util/datasets.py` 는 `ImageFolder` 를 그대로 쓴다. `val/` 에 클래스 폴더 1000개만
있어야 한다. `ILSVRC2012_img_val` 같은 폴더나 `.tar` 가 섞이면 **1001 클래스로 잡혀
train 과 라벨 인덱스가 어긋나고 평가 정확도가 통째로 무의미해진다.** (실제로 겪음:
val 이 1001 클래스 / 63,060장으로 잡혔다)

깨끗한 루트를 심볼릭 링크로 만든다 (원본 154GB 복사 불필요):

```bash
SRC=<원본 imagenet>; CLEAN=<새 루트>
mkdir -p $CLEAN/val && ln -sfn $SRC/train $CLEAN/train
for d in $(ls $SRC/val | grep -E '^n[0-9]{8}$'); do ln -sfn $SRC/val/$d $CLEAN/val/$d; done
```

반드시 검증:
```python
from torchvision import datasets
tr=datasets.ImageFolder(f"{CLEAN}/train"); va=datasets.ImageFolder(f"{CLEAN}/val")
assert len(tr.classes)==1000 and len(va.classes)==1000
assert len(tr)==1281167 and len(va)==50000
assert tr.class_to_idx == va.class_to_idx      # 이게 핵심
```

---

## 6. 실행 절차

`SDT_D` 로 D 를 고른다. **안 주면 조용히 8로 돌아간다.** (모듈 상수가 함수 기본 인자에
import 시점 바인딩되므로 런타임 변경 불가 — env 로만 선택)

### 6.1 MAE pretrain (200 epoch, 8.0일)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 main_pretrain.py \
  --model spikmae_12_768 \
  --batch_size 128 --accum_iter 4 \
  --blr 1.5e-4 --warmup_epochs 20 --epochs 200 --mask_ratio 0.50 \
  --num_workers 12 --pin_mem \
  --data_path $CLEAN --output_dir $OUT --log_dir $OUT
```
유효배치 2048, actual lr 1.20e-3. peak 34.6GB/GPU. 최종 `checkpoint-199.pth`.
pretrain 의 D 는 `MAE_SDT.py` 의 `T=4` 로 고정 (env 미적용).

### 6.2 finetune (150 epoch, 6.2일) -> 74.59%
```bash
export SDT_D=8
torchrun --standalone --nproc_per_node=4 main_finetune.py \
  --model spikformer12_768 --finetune $PRE/checkpoint-199.pth \
  --batch_size 100 --accum_iter 2 \
  --blr 6e-4 --warmup_epochs 10 --layer_decay 0.75 \
  --epochs 150 --drop_path 0.1 \
  --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --num_workers 12 --dist_eval \
  --data_path $CLEAN --output_dir $OUT --log_dir $OUT
```
유효배치 800, actual lr 1.87e-3. peak 34GB/GPU. 에폭당 59분.
정상 궤도 확인: ep10 에서 38% 내외, ep50 에서 67% 내외 (`ref_d8_finetune_log.txt` 대조).

### 6.3 distillation (100 epoch, 5.4일) -> 82.78%
```bash
export SDT_D=8
torchrun --standalone --nproc_per_node=4 main_finetune.py \
  --model spikformer12_768 --finetune $FT/checkpoint-149.pth \
  --kd --teacher_model caformer_b36_in21ft1k --distillation_type hard \
  --batch_size 100 --accum_iter 2 \
  --blr 1e-3 --warmup_epochs 5 --epochs 100 \
  --drop_path 0.1 --time_steps 1 --input_size 224 \
  --reprob 0.25 --mixup 0.5 --cutmix 1.0 \
  --num_workers 12 --dist_eval \
  --data_path $CLEAN --output_dir $OUT2 --log_dir $OUT2
```
**distillation 은 선택이 아니다.** 논문이 "we exploit the distillation technique during
fine-tuning" 이라고 명시하고, 실제 효과가 **+8.19%p (74.59 -> 82.78)** 로 가장 컸다.

초반에 정확도가 출발점(74.59%) 아래로 내려간다. ep33 에서 73% 수준이었다가 lr 이
떨어지며 역전한다. 이걸 실패로 보고 중단하지 말 것.

### 6.4 평가만
```bash
export SDT_D=8
torchrun --standalone --nproc_per_node=4 main_finetune.py \
  --model spikformer12_768 --kd --resume $CK --eval \
  --batch_size 50 --input_size 224 --time_steps 1 \
  --num_workers 8 --dist_eval --data_path $CLEAN --output_dir /tmp/ev --log_dir /tmp/ev
```

### 6.5 중단된 run 이어받기
```bash
--resume <dir>/checkpoint-best.pth     # model/optimizer/scaler/epoch/model_ema 포함
```
저장된 epoch+1 부터 lr 스케줄 위치까지 복원된다 (검증함).
주의 2가지:
- `--resume` 은 `model_ema` 를 복원하지 않는다 (upstream `load_model` 미지원). EMA 재시작.
- `checkpoint-best.pth` 는 정확도가 갱신될 때만 저장된다. 학습 후반에는 갱신이 뜸해져서
  resume 지점이 여러 epoch 뒤처질 수 있다. 주기 저장은 `epoch % 50 == 0` 뿐이다.

### GPU 수가 다를 때
`batch_size x accum_iter x GPU수` 로 유효배치를 맞춰라. lr 이 여기서 유도된다
(`actual lr = blr x eff_batch / 256`). 유효배치가 바뀌면 lr 이 같이 바뀐다.
bs=192/GPU 에서 46.3GB 였으니 48GB 카드는 128 이 상한선이다 (pretrain 기준).

---

## 7. 환경별 주의

- `NCCL_P2P_LEVEL=NVL` — 개발에 쓴 박스에서 **GPU0<->GPU1 P2P 경로가 깨져** NCCL
  all_reduce 가 행에 걸렸다. 증상: GPU util 100% 인데 메모리가 500MB 수준에서 진행 없음,
  DDP init 에서 멈춤. 4-GPU all_reduce 만 하는 최소 스크립트로 격리했고 (0,1) 페어만 실패,
  (2,3)/(0,2) 는 정상이었다. **다른 서버에서는 이 변수를 빼고 시작하고**, 같은 증상이
  나오면 그때 추가할 것.
- `spikformer_s_direct.py` 는 `import matplotlib.pyplot` 을 한다. headless 에서 문제되면
  `MPLBACKEND=Agg`.

---

## 8. 산출물 위치 (개발에 쓴 박스)

```
/data/kyccj/H-direct_new_base/ImageNet_SDTv3_large/ours/
  171M_pretrain/checkpoint-199.pth      pretrain 산출물 (finetune 입력)
  171M_finetune/checkpoint-149.pth      D=8 finetune 74.59%
  171M_kd/checkpoint-best.pth           최고 결과 82.78%
  171M_finetune_d4/checkpoint-149.pth   D=4 finetune 74.38%
  171M_kd_d4/checkpoint-best.pth        D=4 KD, ep13 에서 중단
/data/kyccj/pretrained/caformer_b36_in21ft1k.pth    teacher (377MB)
/data/kyccj/imagenet1k/                깨끗한 심볼릭 데이터 루트
/data/kyccj/migrate/                    이전용 번들 (git bundle + 안내)
```
