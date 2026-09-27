# LIBERO SmolVLA — Trace-Augmentation 실험 (this PC / rtx4070ti-super)

다른 PC에서 진행하던 libero-vla 비교 작업의 **동일 벤치(LIBERO) 재현 + trace-augmentation(WAM식) 검증** 파트를 이 PC에서 돌린 결과.

## 가설
VLA(SmolVLA)의 약점(오차 누적 / 물리 일반화 부족)을 **미래 궤적(2D point-trace) 예측을 보조 목적으로** 학습시켜 개선할 수 있는가 (거대 world-model 없이 dynamics 감각 주입 = ATM/MolmoAct trace 아이디어의 경량판).

## 셋업
- 정책: **SmolVLA** (`lerobot/smolvla_base`, SmolVLM2-500M + flow-matching action expert)
- 벤치: **LIBERO** (`hf-libero`, lerobot 0.4.4 통합), suite=`libero_spatial`, 데이터 `lerobot/libero`(432 spatial ep)
- 격리 venv (lerobot_env 상속), headless EGL 렌더
- 학습: expert-only, 12000 step, batch 8, from smolvla_base

## 방법 (Track B)
1. **trace 라벨 생성** (`gen_traces.py`): CoTracker3(online)로 agentview 영상에서 10×10=100점 격자의 미래 궤적 추출 → `(T,100,2)`
2. **trace-aug 학습** (`train_trace.py`): SmolVLA action expert의 특징(action_out_proj 입력)에서 미래 16프레임 trace를 예측하는 보조 헤드 추가, loss = action(flow-matching) + λ·trace(MSE)
   - `trace_weight=1.0` = treatment, `0.0` = control (동일 설정, 매칭 비교)
3. **평가**: libero_spatial 100ep, init_states=true(in-dist)/false(OOD 분포이동)

## 결과 (libero_spatial, n=100)

| 모델 | in-dist | OOD (init_states=false) |
|---|---|---|
| smolvla_libero (공식 참조) | 66% | — |
| baseline-repro (lerobot표준, 20k) | 58% (@5k=14%) | — |
| **trace_aug** (expert-only+trace, 12k) | **49%** | **46%** |
| **control** (expert-only, no-trace, 12k) | **50%** | **49%** |
| **Δ (trace − control)** | **−1%p** | **−3%p** |

(n=300 tight re-eval 진행 중 — 오차범위 ±2.9%p로 축소해 재확인)

## 결론 (정직하게)
- **이 셋업에선 trace 보조효과 미검출** — in-dist(−1%p)·OOD(−3%p) 모두 trace_aug가 control보다 높지 않음. n=100 노이즈(SE±5%p) 내 = 통계적으로 0과 구분 불가.
- "WAM/trace 아이디어 반증"이 아니라 **좁은 조건(오프라인·12k·spatial·약한 shift·단일 시드)의 null**. 오프라인 trace는 covariate shift(오차누적)를 근본적으로 못 잡음(예정된 결과).
- 다음: 강한 OOD(시각교란/LIBERO-Plus/교차suite), 다중 시드, **on-policy(DAgger+trace)**, trace 설계 튜닝, MolmoAct 실제 계획 trace(Track C).

## 재현 시 주의 — 실제로 겪은 버그 5개
1. `itertools.cycle(dataloader)` → 배치(이미지) 캐싱으로 스텝당 ~15MB RAM 누수 → OOM. 캐싱 없는 cycle 사용.
2. 고아 프로세스(PPID=1)가 shmem 점유 → RAM cascade OOM. watchdog으로 자동 정리.
3. `train_expert_only=false` → 사전학습 VLM 파괴 → eval 0%. expert-only 유지.
4. trace 헤드는 **학습되는 특징**(expert)에 연결해야 함(얼린 VLM에 붙이면 무효 probe).
5. **`postprocessor_overrides`(action 역정규화 stats) 누락** → base(SO-100) stats로 역정규화 → eval 전부 0%. `dataset.meta.stats`로 override 필수. ← 모든 0%의 진범.

## 파일
- `scripts/` — gen_traces.py, train_trace.py, run_experiment.sh, tight_eval.sh, ood_eval.sh, watchdog.sh, env.sh
- `results/` — 각 eval의 pc_success 요약 json
