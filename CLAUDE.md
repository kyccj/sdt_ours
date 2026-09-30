# 이 저장소 작업 시 먼저 읽을 것

SDT-v3 / E-SpikeFormer 포크. 주 작업은 **171M large 모델 ImageNet-1k 재현**이며,
진행 상황·검증된 수치·upstream 함정은 전부 아래 문서에 있다.

**-> `SDT_V3/Classification/Model_Large/REPRODUCTION.md`**

새 실험을 제안하거나 시작하기 전에 그 문서의 "이미 배제된 가설" 절을 읽어라.
격차 원인 후보 4개를 이미 실험해서 배제했다 (각각 5~6일 소요).

## 특히 오해하기 쉬운 두 가지

1. **`spikformer_s_direct.py` 의 `DFE_BatchNorm2d` / `fd_loss` 는 논문 구성요소가 아니다.**
   저장소 소유자의 자체 제안 방법이고, 현재 **동작하지 않는다** — `extra_losses` 에
   append 만 하고 읽는 코드가 없으며, forward 는 vanilla 와 수치적으로 동일하다.
   즉 코드를 그대로 돌리면 그게 논문 baseline이다. 이 클래스가 있다는 이유로
   "논문과 다른 설정" 이라고 판단하지 말 것.

2. **`main` 브랜치에는 pretrain->finetune 을 막는 버그가 있다.**
   `embed_dim` 이 196 이어서 pretrain 체크포인트 로드가 shape mismatch 37개로 죽는다.
   수정은 `fix/large-model-repro-patches` 브랜치에 있다 (총 4개 함정, 문서 3절 참고).

## 현재 최고 결과

pretrain 200ep -> finetune 150ep (74.59%) -> distillation 100ep (**82.78%**)
논문 대비: 85.1% (173M, 1x8, 224) 기준 **-2.32%p**

## 환경

`timm==0.5.4` 고정, torch 2.5.1+cu124, python 3.9. 상세는 문서 4절.
데이터셋은 val 폴더에 클래스 1000개만 있어야 한다 (문서 5절 — 실제로 겪은 함정).
