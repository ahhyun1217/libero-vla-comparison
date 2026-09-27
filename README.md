# 16GB GPU 한 대에서 VLA 모델 비교와 증류 실험

RTX 5070 Ti(VRAM 16GB, 시스템 RAM 15GB) 한 대에서 LeRobot 의 vision-language-action
policy 들을 LIBERO 로 측정하고, **작은 모델을 후처리로 키울 수 있는지** 실험한 기록.

- 모든 수치는 이 머신에서 직접 측정함
- 실패한 시도와 오진 과정도 그대로 남김

## 한 줄 결론

**후처리 기법보다 모델 선택이 훨씬 중요했음.**

- SmolVLA 를 증류·DAgger·전체학습으로 7시간 넘게 끌어올려 48% → 58% 를 얻음
- 기성품 X-VLA 를 다운로드하니 **92%**

## 전체 결과

조건: `libero_10` task 6 ("흰 머그컵을 접시에, 초콜릿 푸딩을 접시 오른쪽에"),
seed 1000, 50 episode

| 모델 | 크기 | 성공률 | VRAM | 우리가 한 일 |
|---|---:|---:|---:|---|
| MolmoAct2 | 5B | **100 %** | 14.2 GB | 기성품 평가 |
| **X-VLA** | **0.9B** | **92 %** | **5.0 GB** | 기성품 평가 |
| VLA-JEPA | 2.3B | 86 % | 7.4 GB | 기성품 평가 |
| SmolVLA + DAgger | 0.45B | 58 % | 2.3 GB | 직접 증류 |
| SmolVLA + VLM 전체학습 | 0.45B | 58 % | 2.3 GB | 직접 학습 |
| SmolVLA + 증류 | 0.45B | 56 % | 2.3 GB | 직접 증류 |
| SmolVLA 원본 | 0.45B | 48 % | 2.3 GB | 기준선 |
| X-VLA 12층 + 증류 | 0.5B | **0 %** | 2.9 GB | 직접 압축 (실패) |

**크기 순서와 성능 순서가 일치하지 않음.** 0.9B 인 X-VLA 가 2.3B 인 VLA-JEPA 를 이김.

## 1. 기성품 모델 비교

### Suite 전체 (SmolVLA vs MolmoAct2)

suite 별 10 task × 2 episode

| 정책 | goal | spatial | object | libero_10 | 평균 |
|---|---:|---:|---:|---:|---:|
| SmolVLA (0.5B) | 100 % | 80 % | 95 % | 70 % | 86.3 % |
| MolmoAct2 (5B) | 90 % | 100 % | 100 % | 95 % | **96.3 %** |

- task 가 어려워질수록 격차가 벌어짐
- 가장 단순한 goal 에서는 SmolVLA 가 앞섬
- 장기 과제 libero_10 에서 25%p 뒤집힘

### ACT: 학습량이 전부였음

같은 데이터(단일 task 43 episode), 같은 하이퍼파라미터, `--steps` 만 변경

| Step | 성공률 | Episode 당 | 학습 시간 |
|---:|---:|---:|---:|
| 20,000 | 60 % | 3.6 초 | 12 분 |
| 100,000 | **100 %** | **2.7 초** | 57 분 |

- 길게 학습한 쪽이 **episode 당 시간도 짧음**
- 더 좋은 policy 가 더 적은 simulator step 으로 목표에 도달하기 때문
- ACT 는 language conditioning 이 없어 학습한 task 만 수행함. 옆 task 에 넣으면 **0 %**

## 2. 작은 모델을 키우려는 시도 (전부 실패에 가까움)

- 대상: SmolVLA (0.45B)
- teacher: MolmoAct2 (5B)
- task: `libero_10` task 6

| 시도 | 바꾼 것 | 데이터 | 성공률 |
|---|---|---:|---:|
| 기준선 | 없음 | 없음 | 48 % |
| **증류** | 데이터 출처를 teacher rollout 으로 | 36 ep / 8,409 frame | 56 % |
| **DAgger** | 학생이 실패한 상태를 teacher 가 교정 | 66 ep / 19,691 frame | 58 % |
| **VLM 전체 학습** | 학습 파라미터 306M (68 %) 로 확대 | 위와 동일 | 58 % |
| **EMA** | 가중치 지수이동평균 | 위와 동일 | **0 %** |
| **temporal ensemble** | ACT 에 chunk 평균 적용 (`coeff=0.01`) | 없음 | 100 % → **82 %** |

- 데이터를 2.3배 늘려도 **+2%p**
- 학습 파라미터를 수십 배 늘려도 **+0%p**
- 세 축을 모두 건드려도 58 % 에서 멈춤

> EMA 0 % 와 temporal ensemble 의 성능 하락은 기법 자체의 실패가 아니라
> **구현이 실제로 적용되지 않았을 가능성**이 있음. lerobot 에서 EMA 는 TDMPC/VQBeT 에만
> 있고 SmolVLA 에는 없음. `--ema.enable=true` 가 조용히 무시되었는지 확인이 필요함.

### 모델 압축(Shallow-pi 방식)은 더 크게 실패함

X-VLA 24층 teacher 에서 앞 12층만 물려받아 student 를 만들고 재학습함.

| | 성공률 |
|---|---:|
| X-VLA 24층 (원본) | 92 % |
| **대조군: 24층을 우리 스크립트로 재저장** | **90 %** |
| X-VLA 12층 + 20k step 재학습 | **0 %** |

**대조군이 핵심임.**

- 90 % 가 나왔으므로 잘라내기·저장 스크립트는 정상
- 따라서 **층을 절반으로 줄인 것 자체**가 원인임이 확정됨

논문(Shallow-pi, 18층→6층, 성능 하락 1%p 미만)과 우리 조건의 차이

| | 논문 | 우리 |
|---|---|---|
| 재학습 데이터 | 대규모 | **36 episode** |
| 학습 범위 | 전체 | **encoder 동결** (OOM 회피) |
| 증류 신호 | 정교한 설계 | 단순 action 모방 |

층을 자르는 것은 쉽지만 **잃어버린 표현력을 되찾는 데는 큰 자원이 듦.**

## 3. 왜 58 % 에서 막혔나

남은 설명은 **아키텍처 자체**임. SmolVLA 는 세 겹으로 얇게 설계돼 있음.

| 설계 | 값 |
|---|---|
| VLM 층 사용 | 전체의 **절반(16층)** 만 |
| visual token | frame 당 **64개** 로 제한 |
| action expert 폭 | VLM 의 **0.75배** |

- 이 셋은 학습으로 바꿀 수 없음
- 특히 visual token 64개 제한은 "머그컵과 푸딩의 상대 위치" 같은 공간 정보를 뭉갤 수 있음

반대로 성능이 좋았던 모델들은 서로 다른 축에서 강했음.

| 모델 | 데이터 | 적응 전략 | 파라미터 배치 |
|---|---|---|---|
| MolmoAct2 | 최대 규모 + 백본 재학습(Molmo2-ER) | 전체 학습 | 백본 4B, state 256 token |
| **X-VLA** | **290K episode / 7 플랫폼** | **soft prompt 만 1%(9M) 학습** | **액션 모듈 24층** |
| VLA-JEPA | 상대적으로 적음 | world model 보조 손실 | 백본 2B, 액션 헤드 12층 |
| SmolVLA | 22.9K episode | expert 만 학습 | **양쪽 다 얇음** |

- X-VLA 는 백본이 작은데 **액션 모듈이 VLA-JEPA 의 두 배 깊음**
- 조작 과제에서는 "장면 이해"보다 "어떻게 움직일지 계산"에 용량을 쓰는 편이 유리했던 것으로 보임

## 4. 16GB 에 무엇이 들어가는가

측정한 VRAM peak

| Policy | Parameter | 정밀도 | VRAM peak | 여유 |
|---|---:|---|---:|---:|
| ACT | 0.05 B | fp32 | 1,283 MiB | 15.0 GB |
| SmolVLA | 0.45 B | fp32 | 2,280 MiB | 13.7 GB |
| X-VLA | 0.9 B | fp32 | 5,002 MiB | 11.1 GB |
| GR00T | 3 B | bf16 | 6,575 MiB | 9.5 GB |
| VLA-JEPA | 2.3 B | fp32 | 7,382 MiB | 8.7 GB |
| Pi0.5 | 3.5 B | bf16 | 9,417 MiB | 6.7 GB |
| MolmoAct2 | 5 B | bf16 + AMP | 14,202 MiB | 1.9 GB |

학습은 별개임.

- SmolVLA: expert 만 3.4GB, VLM 까지 4.8GB 로 여유 있음
- X-VLA: batch 8 에서 **15.8GB 로 OOM**. batch 2 + encoder 동결로 5.7GB

### 진짜 병목은 VRAM 이 아니라 시스템 RAM 이었음

이 머신은 **RAM 15GB 가 VRAM 16GB 보다 작음.**

- fp32 checkpoint 는 GPU 로 가기 전에 RAM 에 먼저 올라감
- 14GB 짜리 `pi05_libero` 는 **GPU 를 건드려보지도 못하고** 로딩 중에 종료됨
- traceback 도 남지 않아 원인 파악에 시간이 걸림
- `groot_lr_*`(12GB)도 같은 이유로 실패

**bf16 배포본을 찾으면 대부분 해결됨.** 같은 모델의 bf16 버전은 7.5GB 였음.
디스크상 checkpoint 크기를 RAM 여유와 먼저 비교할 것.

## 5. 환경 설정에서 막히는 지점

### 설치

- **eval 이 도는 중에 같은 venv 에서 `uv sync` 를 실행하지 말 것.**
  실행 중 `bddl` 이 제거되어 평가 4건이 죽었음. 설치와 실행은 순차 분리
- **`num2words`, `bddl`** 은 lockfile 에 없어 `uv sync` 가 매번 삭제함. 동기화 후 재설치
- **`egl-probe`** (LIBERO 의존성)는 CMake 4 이상에서 빌드 실패.
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` 를 줄 것. 자체 glad 포함이라 시스템 GL 헤더는 불필요

### 실행

- **LIBERO 는 첫 import 때 대화형으로 질문함.** 백그라운드 실행이 EOFError 로 죽으니
  `~/.libero/config.yaml` 을 미리 생성
- **FFmpeg 라이브러리가 없으면** torchcodec 이 비디오를 못 읽음.
  `--dataset.video_backend=pyav` 를 명시 (이 머신은 libavutil 이 0개였음)
- **Headless 렌더링**에는 `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` 필요.
  `lerobot-eval` 에는 실시간 표시 기능이 없고 mp4 로만 저장하므로 물리 모니터는 무의미함
- **`--eval.recording` 은 LIBERO 에서 동작하지 않음.** feature 이름에 `/` 가 들어가
  dataset 생성이 거부되고, `_build_raw_frame` 은 `observation.images.*` 를 기대함

### 비교를 망칠 수 있는 설정

- **`n_action_steps` 기본값이 policy 마다 다름** (ACT 100, SmolVLA/MolmoAct2 10,
  VLA-JEPA 7, X-VLA 32). 고정하지 않으면 policy 가 아니라 action horizon 을 측정하게 됨
- **`lerobot/smolvla_libero` 는 기본값 50** 인데 LIBERO 에서 약 20%p 를 깎음 (lerobot#4614)
- **X-VLA 는 `--env.control_mode=absolute` 가 필요함.** 다른 정책과 조건이 달라지는 지점
- **camera key 이름이 checkpoint 마다 다름.** rename_map 필요
  - SmolVLA: `camera1` / `camera2`
  - MolmoAct2-LIBERO: `image` / `wrist_image`
  - MolmoAct2-SO100_101: `cam0` / `cam1`
  - VLA-JEPA: `image` / `image2`
- **checkpoint 의 `config.json` 을 그대로 믿지 말 것.** `lerobot/smolvla_libero` 의
  `input_features` 는 camera 3개 / state 6차원으로 적혀 있으나 실제 학습 데이터는
  camera 2개 / state 8차원임. preprocessor 에 저장된 통계가 실제 값
- 일부 배포 checkpoint 에는 **컨테이너 절대경로**(`/models-nvme/groot_base`)나 이 LeRobot
  버전에 등록되지 않은 **policy type**(`pi05_openpi`)이 박혀 있음
- **Pi0.5 는 gated base model**(`google/paligemma-3b-pt-224`) 접근 권한이 필요함

### 커스텀 수집기를 짤 때 (LeRobot 내부 API)

`scripts/vla_collect.py` 에 아래 내용이 전부 반영돼 있음.

- import 위치
  - `make_env`, `make_env_pre_post_processors`: `lerobot.envs.factory`
  - `preprocess_observation`: `lerobot.envs.utils`
- `make_env` 반환은 `{suite: {task_id: VectorEnv}}` **2단 중첩 dict**
- `observation.state` 는 raw 관측에 없고 **env preprocessor 통과 후** 생성됨.
  policy 에 따라 차원이 다름 (MolmoAct2 8, **X-VLA 20**). 패딩이므로 앞 8개만 저장
- dataset writer 는 이미지를 **HWC uint8** 로 받음. dataset 이 주는 건 CHW float[0,1]
- `LeRobotDataset.create(fps=...)` 에 float 를 넣으면 비디오 인코딩이
  `'float' object has no attribute 'numerator'` 로 실패. `int()` 로 변환
- 모든 episode 를 쓴 뒤 **`dst.finalize()`** 를 호출해야 `meta/episodes` 가 flush 됨.
  빠뜨리면 이후 로드가 HF Hub 조회로 넘어가 404
- MolmoAct2 preprocessor 에 action 을 같이 넘기면 gripper 범위 검증에서 실패함.
  teacher 는 예측에 관측만 필요하므로 `observation.*` 만 전달할 것
- **저장할 action 은 `env_post` 를 통과한 값임.** X-VLA 기준 `env_post` 는 내부 20차원
  패딩을 환경용 7차원으로 푸는 단계
- **bash 스크립트에 `pgrep -f` 대기 루프를 넣지 말 것.** 호출자의 명령줄까지 잡아
  자기 자신을 감지해 영원히 멈춤. 이 실험에서 두 번 겪었음

## 6. 실험 위생에서 배운 것

- **대조군 없이는 원인을 못 찾음.**
  X-VLA 12층이 0 % 였을 때 "증류 실패"인지 "스크립트 버그"인지 알 수 없었음.
  층을 자르지 않고 재저장만 한 대조군을 세워 90 % 를 확인하고서야 원인이 확정됨
- **표본이 작으면 아무 말도 못 함.**
  2 episode 에서 SmolVLA 가 이 task 에 0 % 로 나와 "개선 여지가 최대"라고 판단했는데,
  50 episode 로 재면 **48 %** 였음. 실험 설계의 전제 자체가 틀렸던 것
- **비교 대상이 같은 것인지 먼저 확인할 것.**
  teacher 출력(20차원 내부 표현)과 저장된 데이터(7차원 환경 행동)의 스케일이 다른 것을
  보고 "저장 버그"라 오진했음. 애초에 서로 다른 공간의 값이었음
- **조건을 고정하지 않으면 정책이 아니라 설정을 측정함.**
  `n_action_steps` 를 통일하지 않았을 때 ACT 성적이 50 % 와 60 % 로 갈렸음

## 저장소 구성

```
scripts/
  vla_collect.py       rollout 수집 공통 모듈 (환경·정책·저장 로직)
  collect_*.py         teacher / student / X-VLA 수집기 (얇은 설정 래퍼)
  relabel_student.py   학생 상태에 teacher action 라벨링 (DAgger 2단계)
  merge_datasets.py    Dataset Aggregation (D <- D u D_i)
  carve_student.py     teacher 에서 얕은 student 잘라내기 (Shallow-pi)
  run_*.sh             파이프라인 실행 스크립트
  watch_live.py        headless VNC 롤아웃 뷰어
results/               측정 원본 출력 (실행 묶음별 1개 파일)
videos/                선별한 rollout (성공, 실패)
papers/                각 policy 논문 PDF (git 제외, 인용은 ARCHITECTURE.md)
```

구조 비교는 [ARCHITECTURE.md](ARCHITECTURE.md) 에 논문과 구현체를 근거로 정리함.

## 재현 방법

```bash
uv sync --locked --extra molmoact2 --extra libero --extra vla_jepa
uv pip install num2words bddl                       # sync 할 때마다 지워짐
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
bash scripts/compare3.sh
```

## 학습한 모델

- [`AmberHyunKIM/act_libero_goal_task0`](https://huggingface.co/AmberHyunKIM/act_libero_goal_task0): 100k step, 100 %
- [`AmberHyunKIM/act_libero_goal_task0_20k`](https://huggingface.co/AmberHyunKIM/act_libero_goal_task0_20k): 20k step, 60 %

## 한계와 주의사항

- **task 1개, 50 episode 기준임.** 48/56/58 % 사이 차이는 성공 몇 회 차이라
  통계적으로 구분되지 않음. 방향성만 참고할 것
- MolmoAct2 는 bf16 + AMP, 나머지는 fp32. 16GB 에서는 다른 선택지가 없었으므로
  통제된 변수가 아니라 하드웨어 제약임
- X-VLA 만 `control_mode=absolute` 를 씀. 정책 특성이라 맞출 수 없는 차이
- EMA 와 temporal ensemble 결과는 구현이 실제로 적용되었는지 검증하지 않았음
- GR00T 와 Pi0.5 는 표에 없음. 둘 다 시도했고 막힌 지점은 위에 적어두었음

---

## 부록: trace-augmentation (별도 머신, RTX 4070 Ti Super)

본편(RTX 5070 Ti)의 "후처리로 작은 모델을 키울 수 있는가" 질문을 두 번째 머신에서 **한 축 더** 검증함. 여기서 시도한 후처리는 **trace-augmentation** — SmolVLA 에 "미래 2D 궤적(point-trace) 예측"을 보조 손실로 추가해 dynamics 감각을 주입하는 방식 (거대 world-model 없이 ATM/MolmoAct 의 trace 아이디어만 경량화).

- 라벨: **CoTracker3** 로 agentview 영상에서 100점 격자의 미래 궤적 추출
- 부착: SmolVLA action expert 특징(`action_out_proj` 입력) → 미래 16 frame trace 예측 헤드
- 학습: expert-only, 12k step, `libero_spatial` 432 episode. `trace_weight=1` 이 treatment, `0` 이 control (동일 설정 매칭 비교)

### 결과 (`libero_spatial`, 300 episode, SE ±2.9 %p)

| 조건 | trace-aug | control | Δ |
|---|---:|---:|---:|
| in-distribution | 47.0 % | 48.3 % | **−1.3 %p** |
| OOD (init state 이동) | 49.0 % | 51.7 % | **−2.7 %p** |

예비 100 episode 측정도 −1 %p / −3 %p 로 방향 동일함.

- **trace 후처리 효과 미검출.** 4개 측정(2조건 × 2 표본크기) 모두 trace-aug 가 control 을 못 넘음
- 차이는 전부 표본 노이즈(n=300 기준 SE ±2.9 %p) 안 → 통계적으로 0 과 구분되지 않음
- 참고선: `smolvla_libero`(공식) 66 %, lerobot 표준 재현(20k) 58 %

→ 본편 결론 **"후처리 기법보다 모델 선택이 중요"** 를 한 번 더 지지함.

### 왜 안 나왔나 (추정)

- 이 trace 는 **오프라인**이라 covariate shift(오차 누적)를 근본적으로 못 잡음 — 오프라인 증류가 큰 이득을 못 준 본편 결과와 같은 맥락
- CoTracker 격자점은 "의미 있는 계획"이 아니라 기하학적 점이라 신호가 약했을 수 있음
- 진짜 검증에는 **on-policy(DAgger+trace) · 강한 OOD(LIBERO-Plus / 교차 suite) · 다중 시드**가 필요

### 재현 시 겪은 버그 5개

| 증상 | 원인 | 해결 |
|---|---|---|
| 학습 중 RAM OOM | `itertools.cycle(dataloader)` 가 배치를 전부 캐싱 | 캐싱 없는 cycle |
| RAM cascade OOM | 고아 프로세스(PPID=1)가 shmem 점유 | watchdog 자동 kill |
| eval 0 % | `train_expert_only=false` 로 사전학습 VLM 파괴 | expert-only 유지 |
| trace 무효 | trace 헤드를 얼린 VLM 에 부착 | 학습되는 expert 특징에 부착 |
| **eval 전면 0 %** | **`postprocessor_overrides` 누락 → action 을 base(SO-100) stats 로 역정규화** | `dataset.meta.stats` 로 override |

상세·재현·스크립트: [`TRACE_AUG_rtx4070.md`](TRACE_AUG_rtx4070.md), `scripts/trace_aug/`, `results/trace_aug_results.txt`
