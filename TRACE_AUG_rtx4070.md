# LIBERO SmolVLA trace-augmentation 실험 (rtx4070ti-super)

다른 PC 에서 진행하던 libero-vla 비교 작업 중, 동일 벤치(LIBERO) 재현과
trace-augmentation(WAM 식) 검증 파트를 이 PC 에서 돌린 결과임.

## 가설

VLA(SmolVLA) 의 약점인 오차 누적과 물리 일반화 부족을, 미래 궤적(2D point-trace) 예측을
보조 목적으로 학습시켜 개선할 수 있는가. 거대 world model 없이 dynamics 감각만 주입하는
ATM / MolmoAct trace 아이디어의 경량판임.

## 셋업

- 정책: SmolVLA (`lerobot/smolvla_base`, SmolVLM2-500M + flow-matching action expert)
- 벤치: LIBERO (`hf-libero`, lerobot 0.4.4 통합), suite `libero_spatial`, 데이터 `lerobot/libero` (spatial 432 episode)
- 격리 venv (lerobot_env 상속), headless EGL 렌더
- 학습: expert-only, 12,000 step, batch 8, `smolvla_base` 에서 시작

## 방법 (Track B)

1. **trace 라벨 생성** (`gen_traces.py`). CoTracker3(online) 로 agentview 영상에서 10x10=100 점 격자의 미래 궤적을 뽑아 `(T,100,2)` 로 저장
2. **trace-aug 학습** (`train_trace.py`). SmolVLA action expert 의 특징(`action_out_proj` 입력) 에서 미래 16 frame trace 를 예측하는 보조 헤드를 붙이고, loss = action(flow matching) + λ x trace(MSE)
   - `trace_weight=1.0` 이 treatment, `0.0` 이 control. 나머지 설정은 완전히 동일한 matched comparison 임
3. **평가**. `libero_spatial`, `init_states=true` 가 in-distribution, `false` 가 OOD(초기 상태 분포이동)

## 결과

최종 수치는 조건당 300 episode (task 10 개 x 30) 기준, 표준오차 ±2.9 %p.

| 조건 | trace-aug | control | Δ (trace − control) |
|---|---:|---:|---:|
| in-distribution | 47.0 % | 48.3 % | **−1.3 %p** |
| OOD (`init_states=false`) | 49.0 % | 51.7 % | **−2.7 %p** |

조건당 100 episode 예비 측정도 방향이 같았음.

| 조건 | trace-aug | control | Δ |
|---|---:|---:|---:|
| in-distribution | 49.0 % | 50.0 % | −1.0 %p |
| OOD | 46.0 % | 49.0 % | −3.0 %p |

참고선.

| 모델 | 성공률 |
|---|---:|
| `smolvla_libero` (공식) | 66 % |
| baseline 재현 (lerobot 표준, 20k step) | 58 % (5k step 에서는 14 %) |

## 결론

- **이 셋업에서 trace 보조효과는 검출되지 않음.** 4 개 측정(2 조건 x 2 표본크기) 전부 trace-aug 가 control 을 넘지 못함
- 차이는 전부 표본 노이즈 안이라 통계적으로 0 과 구분되지 않음
- WAM / trace 아이디어의 반증이 아니라, 좁은 조건(오프라인, 12k step, spatial, 약한 shift, 단일 시드) 에서의 null 임
- 오프라인 trace 는 covariate shift(오차 누적) 를 근본적으로 잡지 못함. 예정된 결과에 가까움

다음으로 해볼 것.

- 강한 OOD (시각 교란, LIBERO-Plus, 교차 suite)
- 다중 시드
- on-policy (DAgger + trace)
- trace 설계 튜닝
- MolmoAct 의 실제 계획 trace 사용 (Track C)

## 재현 시 주의, 실제로 겪은 버그 5 개

| 증상 | 원인 | 해결 |
|---|---|---|
| 학습 중 RAM OOM | `itertools.cycle(dataloader)` 가 배치(이미지 포함) 를 전부 캐싱해 step 당 약 15MB 누수 | 캐싱 없는 자체 `cycle` |
| RAM cascade OOM | 고아 프로세스(PPID=1) 가 shmem 점유 | `watchdog.sh` 로 자동 정리 |
| eval 0 % | `train_expert_only=false` 로 사전학습 VLM 파괴 | expert-only 유지 |
| trace 무효 | trace 헤드를 얼린 VLM 에 부착해 gradient 가 안 흐름 | 학습되는 expert 특징에 부착 |
| **eval 전면 0 %** | **`postprocessor_overrides` 누락으로 action 을 base(SO-100) stats 로 역정규화** | `dataset.meta.stats` 로 override |

마지막 것이 모든 0 % 의 진범이었음. 학습은 멀쩡했고 eval 만 깨져 있었음.

## 파일

`scripts/trace_aug/`

| 파일 | 역할 |
|---|---|
| `env.sh` | 공통 경로와 환경변수. `VLA_WAM` 으로 override |
| `eval_lib.sh` | `lerobot-eval` 공통 러너 (`trace_eval`) |
| `gen_traces.py` | CoTracker3 로 trace 라벨 생성 |
| `train_trace.py` | trace 보조 손실을 붙인 학습 루프. 인자로 treatment / control 전환 |
| `run_experiment.sh` | 두 arm 학습과 짧은 eval |
| `tight_eval.sh` | 최종 측정, 조건당 300 episode |
| `ood_eval.sh` | OOD 예비 측정, 조건당 100 episode |
| `watchdog.sh` | 고아 프로세스 정리와 상태 이벤트 출력 |

`results/trace_aug_results.txt` 에 각 eval 의 `pc_success` 요약이 있음.

`train_trace.py` 가 읽는 `spatial_episodes.json` (libero_spatial 에 해당하는 episode index 432 개)
은 이 저장소에 포함돼 있지 않음. `EPISODES_JSON` 으로 경로를 지정할 수 있음.

## 다른 머신에서 돌릴 때

```bash
VLA_WAM=/my/path bash scripts/trace_aug/run_experiment.sh
```

경로는 전부 `env.sh` 한 곳에서 나옴. 기본값은 이 실험을 실제로 돌린 머신 경로임.
