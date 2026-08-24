# Spike regularization (loss-ratio λ + 브레이크) 이식 정리

TF 쪽(`TensorFlow-SNNs/lib_snn`)에서 쓰던 방법을 SDT-V3 Large classification에 옮긴 것.
2026-08-24 작성.

---

## 1. 무엇을 옮겼나

| 부품 | TF 원본 | 여기 |
|---|---|---|
| 규제항 + 수정된 backward | `lib_snn/layers.py:2024 l2_norm_wta_rev` | `spike_reg._L2NormWtaRev` |
| 규제를 붙이는 뉴런 | `lib_snn/neurons.py` `reg_spike_out` 블록 | `spike_reg.EIPMultispike` |
| loss-ratio λ 제어 | `lib_snn/proc.py:1229` | `spike_reg.LossRatioController` |
| 브레이크 | `lib_snn/proc.py:1242`, `flags.py:865` | 같은 컨트롤러 안 |

**규제항**: 뉴런 층마다 `R_l = ‖spike_l × sc_rate_l‖₂`, 전체 `R = Σ_l R_l`.
손실에는 `loss = task_loss + λ·R`로 들어감.

**수정된 backward**: 보통 L2는 `d‖x‖/dx = x/‖x‖`라서 안 터진 뉴런(x=0)에는 gradient가 0.
대신 `sc_rate/‖x‖`를 쓰면 모든 뉴런에 gradient가 감. 이게 방법의 핵심.

**λ 제어**: epoch마다 `λ = ρ · L_task / R`. 한 스텝에 정확히 풀리므로 gain도 반복도 없음.
목표 스파이크 수나 baseline 같은 사전 지식이 필요 없다는 게 이 방식의 장점.

**브레이크**: ① λ 성장을 epoch당 `growth_cap`배로 제한 (전 구간) ② 초반 `floor_ep` epoch
동안 `S(e)/S(first) < floor`면 λ를 `floor_decay`배로 후퇴.
①은 근거가 확실함 (공격적 ρ에서 무브레이크는 붕괴, 브레이크는 생존).
②는 인과 근거가 아직 없어서 **기본값 0(꺼짐)**.

---

## 2. TF와 달라진 점 (SDT-V3 구조 때문)

1. **뉴런이 이진 스파이크가 아님.** `Multispike`는 `floor(clamp(x,0,4)+0.5)/4` — 0~4 정수
   레벨을 4로 나눈 값. 게다가 T=1이라 타임스텝 루프가 없음.
   → TF에서 R을 T번 더해야 했던 정산 문제가 여기선 없음. forward 한 번이면 R이 완성됨.
2. **`sc_rate` 기본값이 1 (균일).** TF에서 `1-softmax`가 0.9999에 분산 ~0으로 측정됐고
   `sc_rate=1` 대조가 4/4 일치 → softmax는 흔적 기관으로 확정. 여기선 처음부터 1로 두고,
   `--reg_sc_rate wta_rev`로 softmax 가중을 켤 수 있게만 해둠 (활성화 텐서 하나를 더
   만들기 때문에 메모리를 씀).
3. **λ는 python float.** 텐서가 아니므로 DDP/AMP와 얽히지 않음. epoch 끝에 모든 rank가
   MetricLogger로 이미 reduce된 같은 값으로 계산 → broadcast 불필요.

---

## 3. 파일별 변경

- **`spike_reg.py` (신규)** — 위 4개 부품 전부. 다른 파일에서 가져다 쓰기만 함.
- **`engine_finetune.py`**
  - `train_one_epoch`: forward 전 `REG.reset()`, criterion 뒤 `loss += REG.lam * REG.total()`,
    `task_loss / reg_R / spikes / lam`을 metric_logger에 기록
  - `evaluate`: model_mode별 분기 3개를 하나로 통합. **기존 ms 모드는 여기서 AttributeError로
    죽었음** — plain `Multispike`에 `spike_count_int`가 없는데 접근했기 때문. hasattr로 막고,
    래퍼가 있으면 래퍼가 세도록 함
  - `evaluate`의 스파이크 정규화가 `/50000` 하드코딩이었음 → 실제로 본 이미지 수로 변경
- **`main_finetune.py`** — 인자 추가, 체크포인트 로드 뒤 `convert_multispike`, epoch 루프에서
  컨트롤러 갱신, `log.txt`에 `reg_*` 필드 추가
- **`util/datasets.py`** — `--subset_classes` / `--subset_frac` (ρ 스캔용, 결정론적)
- **`spikformer.py`** — `MS_Block.forward`의 `T, B, C, N = x.shape`가 주석 처리돼 있어서
  `choice="base"` 경로(= `spikformer12_512`)가 NameError로 못 돌았음. 주석 해제.
  **이건 이식과 무관한 원래 버그.** 171M(`spikformer12_768`)은 `choice="large"`라 그 분기를
  안 타서 영향 없었음
- **`smoke_spike_reg.py` (신규)** — 데이터셋 없이 도는 M0 점검

---

## 4. 어느 뉴런에 걸리나

`--reg_skip`(기본 `lif`)에 든 **전체 경로 이름**만 제외. 즉 head 앞 출력 뉴런 하나만 빠지고
나머지 107개 전부 규제 대상 (TF의 `loc == 'HID'`에 대응).

- 입력 인코딩(`downsample1_1`)은 `first_layer=True`라 뉴런 자체가 없음 → 자동으로 제외됨
- `spikformer12_768`(large)은 `MS_Block.lif` 12개를 만들어 놓고 forward에서 안 씀 →
  래핑은 되지만 호출이 안 되므로 실제 기여 층은 95개

## 5. 플래그

| 플래그 | 기본값 | 뜻 |
|---|---|---|
| `--reg_spike` | off | 규제 켜기 |
| `--reg_rho` | 5.8e-4 | 목표 reg/task 손실비. **CIFAR 값이라 그대로 쓰면 안 됨 (§6)** |
| `--reg_sc_rate` | `one` | `one`(균일) 또는 `wta_rev`(1-softmax) |
| `--reg_alpha` | 7.0 | softmax 온도, `wta_rev`에서만 |
| `--reg_start_ep` | 0 | 이 epoch 전까지 λ=0 |
| `--reg_growth_cap` | 1.5 | epoch당 λ 성장 상한 (0이면 해제) |
| `--reg_brake_floor` | 0.0 | 초반 창 스파이크 바닥 (CIFAR에선 0.22였음, 기본 꺼짐) |
| `--reg_brake_ep` | 30 | 그 창의 길이 |
| `--reg_brake_decay` | 0.5 | 바닥 아래일 때 λ 배율 |
| `--reg_skip` | `lif` | 규제 제외할 모듈 전체 경로, 쉼표 구분 |
| `--reg_count` | off | 규제 없이 eval 스파이크 카운트만 (baseline arm용) |
| `--subset_classes` | 0 | 앞의 N개 클래스만 사용 (0=전체) |
| `--subset_frac` | 1.0 | 클래스당 이미지 비율 |

## 6. ρ는 아직 정해지지 않았다

`5.8e-4`는 CIFAR 네 설정(V16/R19 × C10/C100)에서 "스파이크 잔량 70%"에 필요했던 ρ의
기하평균이다. 그 넷에 맞춘 값(in-sample)이고, 여기는 뉴런이 0~4 정수 레벨이라 R의 스케일
자체가 다르다. **측정 없이 쓰면 안 된다.**

절차:
1. `--reg_spike --reg_rho <아무 값>`으로 1 epoch 돌려 `log.txt`의 `reg_R`과 `train_task_loss`를 본다
   (λ는 첫 epoch에 0이라 학습에 영향 없음)
2. 서브셋으로 ρ 3점 스캔 → ρ→잔량 곡선
3. 원하는 잔량에 해당하는 ρ로 full 학습

`smoke_spike_reg.py` 기준 초기값 (224px, 무작위 가중치):

| 모델 | R | task_loss | ρ=5.8e-4일 때 λ | 스파이크/이미지 |
|---|---|---|---|---|
| spikformer12_512 | 3.19e4 | 6.85 | 1.25e-7 | 3.0e6 |
| spikformer12_768 | 3.06e4 | 6.90 | 1.31e-7 | 3.96e6 |

## 7. 비용 (A6000 1장, batch 8, 224px)

| 모델 | 규제 off | 규제 on | 증가 |
|---|---|---|---|
| 512 | 0.102 s/iter, 2454 MiB | 0.144 s/iter, 2454 MiB | 시간 +41%, 메모리 +0 |
| 768 | 0.099 s/iter, 3890 MiB | 0.126 s/iter, 3890 MiB | 시간 +27%, 메모리 +0 |

메모리 증가가 없는 건 `sc_rate=1`이라 추가 텐서를 안 만들기 때문. `wta_rev`로 바꾸면
활성화 하나 크기의 텐서가 층마다 더 생긴다. (TF에서는 오버헤드가 93%였음 — 타임스텝 루프가
없어진 만큼 싸졌다.)

## 8. 검증 상태 (2026-08-24)

- M0 스모크: 512/768 모두 통과 — 규제항 유한, backward 도달(547~619 텐서), 안 터진 뉴런에
  gradient 들어감(수정 backward 1.0 vs 일반 L2 0.0), 브레이크 발동/성장 상한 동작
- 단일 GPU e2e: ImageNet 10클래스 × 2 epoch 완주. λ = 5.2e-8 → 3.4e-8로 갱신,
  `ρ = λR/L_task`가 5.8e-4로 정확히 유지됨
- 2 GPU DDP e2e: 완주, rank 간 값 일치
- **아직 안 한 것**: 실제 학습 규모에서의 효과. baseline 가중치가 이 서버에 없어서
  (`*.pth` 0건) 정확도 비교는 다른 서버에서 해야 함

## 9. 실행 예시

baseline (규제 없이, 스파이크만 기록):

```bash
torchrun --standalone --nproc_per_node=8 main_finetune.py \
  --model spikformer12_768 --model_mode ms --batch_size 100 --epochs 150 \
  --blr 6e-4 --layer_decay 0.75 --warmup_epochs 10 --drop_path 0.1 \
  --data_path <IMAGENET> --output_dir <OUT> --log_dir <OUT> \
  --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval \
  --reg_count
```

규제 arm (브레이크는 성장 상한만):

```bash
torchrun --standalone --nproc_per_node=8 main_finetune.py \
  ... 위와 동일 ... \
  --reg_spike --reg_rho <M1에서 정한 값> --reg_growth_cap 1.5
```

ρ 스캔 (서브셋):

```bash
python main_finetune.py --model spikformer12_512 --batch_size 64 --epochs 30 \
  --data_path <IMAGENET> --subset_classes 100 \
  --reg_spike --reg_rho <후보> --output_dir <OUT> --log_dir <OUT>
```
