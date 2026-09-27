# 아키텍처 비교: ACT, SmolVLA, X-VLA, MolmoAct2

각 policy의 논문과 LeRobot 구현체(`src/lerobot/policies/`)를 함께 읽고 정리함.
수치는 이 실험에서 실제로 쓴 config 값.

## 한눈에 보기

| | ACT | SmolVLA | X-VLA | MolmoAct2 |
|---|---|---|---|---|
| Parameter | 51.6M | 0.45B | 0.9B | 5B |
| 측정 성공률 | (단일 task 100%) | 48% | **92%** | **100%** |
| 시각 encoder | ResNet18 (ImageNet) | SigLIP (SmolVLM-2) | Florence-2 | Molmo2-ER |
| 언어 입력 | 없음 | SmolLM2 decoder | Florence-2 | Molmo2-ER |
| Action 생성 | CVAE + transformer decoder | Flow matching | Flow matching | Flow matching (DiT) + discrete token |
| 액션 모듈 깊이 | enc 4 / dec 1 | expert 폭 0.75배 | **24층** | DiT expert |
| VLM 연결 | (없음) | 층별 KV (cross-attn) | encoder 출력 + domain 임베딩 | **층별 KV 대응** |
| 적응 전략 | 전체 학습 | expert 만 학습 | **soft prompt 1%(9M)** | 전체 학습 |
| 논문 | `papers/ACT.pdf` | `papers/SmolVLA.pdf` | (X-VLA) | `papers/MolmoAct2.pdf` |

## ACT (Action Chunking with Transformers)

`Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware`
(Zhao, Kumar, Levine, Finn. Stanford / UC Berkeley / Meta, 2023)

```
이미지 4대 -> ResNet18 -> transformer encoder(4층) -> decoder(1층) -> action chunk k개
                                    ^
                          z (style variable, 32차원)
```

**구성**

| 항목 | 값 |
|---|---|
| `dim_model` | 512 |
| `n_heads` | 8 |
| `dim_feedforward` | 3200 |
| Encoder / Decoder | 4층 / 1층 |
| CVAE latent | 32차원, encoder 4층 |
| `chunk_size` | 100 |
| Loss | L1 + KLD |

**설계 의도**

- 논문이 풀려는 문제는 **compounding error**임
  - 한 step씩 예측하면 오차가 누적되어 학습 분포를 벗어남
  - k step을 한 묶음(chunk)으로 예측하면 유효 horizon이 k분의 1로 줄어듦
- 사람 시연의 변동성을 흡수하기 위해 conditional VAE로 학습함
  - style variable `z`가 그 역할을 담당
  - **추론 시 `z`는 prior의 평균(즉 0)으로 고정**되므로 결정론적으로 동작함

**Temporal ensembling**

- 논문은 chunk를 다 쓰고 다음 chunk를 뽑는 방식 대신, 매 step마다 policy를 다시 질의하고
  겹치는 chunk들을 가중 평균해서 궤적을 매끄럽게 만드는 방식을 제안함
- 이 실험에서 `n_action_steps`를 100에서 10으로 줄였을 때 성공률이 50%에서 60%로 오른 것이
  같은 방향의 결과임. 관측을 더 자주 반영하면 좋아짐

**한계**

- **언어 입력이 구조에 없음.** task를 지정할 통로가 없으므로 학습한 task 하나만 수행 가능함

## SmolVLA

`SmolVLA: A vision-language-action model for affordable and efficient robotics`
(Hugging Face, 2025)

```
이미지 + 언어 + state -> SmolVLM-2 (앞 L/2 층만)
                              | keys, values
                              v cross-attention
                    Action expert (폭 0.75d) -> Euler 적분 10 step -> action chunk 50개
```

**구성**

| 항목 | 값 |
|---|---|
| VLM | SmolVLM2-500M-Video-Instruct |
| 사용 층 | 16층 (`num_vlm_layers=16`) |
| Visual token | frame당 64개 |
| Action expert 폭 | VLM hidden의 0.75배 |
| Attention | cross-attn 기본, 2층마다 self-attn |
| Flow step | 10 |
| Vision encoder | frozen |

**경량화 수단 3가지** (모두 논문에 명시됨)

1. **Layer skipping**
   - VLM 전체 층의 절반(`N = L/2`)만 사용함
   - 근거: VLM 마지막 층 feature가 downstream에 최선이 아니라는 선행 연구
   - LLM과 action expert의 연산량이 같이 절반으로 줄어듦
2. **Visual token 축소**
   - image tiling을 쓰지 않고 global image에 pixel shuffle을 적용함
   - frame당 **64 token**으로 제한
3. **Action expert 축소**
   - hidden size를 VLM의 `0.75 x d`로 둠

**Flow matching action expert**

- conditional Flow Matching Transformer 구조임
- `A_t^τ = τA_t + (1-τ)ε` 경로에서 vector field `u = ε - A_t`를 예측하도록 학습함
- 추론 시 Euler 적분으로 noise에서 action으로 이동함
- **cross-attention과 causal self-attention을 교차 배치**한 것이 특징
  - CA 층은 VLM의 key/value를 참조함
  - SA 층은 action token들이 서로를 보게 함
  - 논문은 이 교차 구조가 성공률과 추론 속도를 모두 개선했다고 보고함
  - SA가 action chunk를 매끄럽게 만드는 역할이라고 설명함

**Pretraining 데이터**

- 커뮤니티 데이터 **481개 dataset, 22.9K episode, 10.6M frame**

## X-VLA

`X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA`

```
이미지 + 언어 -> Florence-2 encoder
                      |
domain_id ---> soft prompt 32개 (로봇/환경별)
                      v
         soft transformer (hidden 1024 x depth 24)
                      | flow matching 10 step (역방향 t: 1->0)
                 action chunk 32개
```

**구성**

| 항목 | 값 |
|---|---|
| VLM | Florence-2 |
| policy transformer | hidden 1024, **depth 24**, heads 16 |
| 도메인 수 | **30** |
| 도메인당 soft prompt | **32개** |
| Flow step | 10 |
| chunk | 32 |
| action 차원 | 20 으로 padding |

**핵심: 도메인별 soft prompt**

```python
self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
                                    #    30           32          x 1024
```

- 로봇/환경(domain)마다 **학습된 벡터 32개**를 사전처럼 들고 있다
- 추론 시 `domain_id` 로 해당 프롬프트를 꺼내 쓴다
- 새 임베디먼트 적응 시 백본을 건드리지 않고 **전체의 1%(9M)만** 학습한다
- 논문: 290K episode / 7개 로봇 플랫폼으로 Phase I 사전학습 후,
  Phase II 에서 도메인별 프롬프트만 학습

추론 루프는 다른 모델과 달리 **역방향 적분**이다.

```python
for i in range(steps, 0, -1):
    t = i / steps
    x_t = x1 * t + action * (1 - t)
    action = self.transformer(domain_id=domain_id, action_with_noise=x_t, proprio=proprio, t=t, **enc)
```

**왜 0.9B 로 2.3B 를 이겼나 (측정: 92% vs 86%)**

| | X-VLA | VLA-JEPA |
|---|---|---|
| 백본 | 작음 (Florence-2) | 큼 (Qwen3-VL 2B) |
| **액션 모듈** | **24층** | 12층 |
| 사전학습 | **290K episode** | 상대적으로 적음 |

백본이 작은 대신 **액션 모듈이 두 배 깊다.** 조작 과제에서는 장면 이해보다
"어떻게 움직일지 계산"에 용량을 쓰는 편이 유리했던 것으로 보인다.

## MolmoAct2

`MolmoAct2: Action Reasoning Models for Real-world Deployment` (Allen AI)

```
이미지 + 언어 -> Molmo2-ER (embodied reasoning VLM)
                      | 각 층의 keys, values
                      v 층별로 대응
         DiT-style action expert -> flow matching 8 step -> continuous action
                      또는 FAST tokenizer -> discrete action token
```

**구성**

| 항목 | 값 |
|---|---|
| VLM | Molmo2-ER (Molmo2-4B 기반) |
| State token | 256개 |
| Flow step | 8 |
| Action mode | `both` (continuous + discrete) |
| Action 차원 | 32로 padding |
| 기본 dtype | bfloat16 |

**구조적 차별점: 층별 KV conditioning**

논문이 기존 VLA의 action expert와 다르다고 명시한 두 가지.

1. **DiT-style transformer에 flow matching 목표**를 씀
   - diffusion과 flow 기반 생성에서 검증된 denoising transformer 구조를 그대로 가져옴
2. **expert를 VLM hidden state에 조건화하지 않고, 각 expert 층을 대응하는 VLM 층의
   key/value에 조건화함**
   - 논문 표현으로는 연산 profile을 비슷하게 유지하면서 VLM 자신이 쓰는 attention state를
     노출하는 방식임
   - ablation에서 더 효과적이었다고 보고함

> SmolVLA도 VLM의 key/value를 cross-attention으로 참조함. 다만 **층별 대응(layer-wise
> correspondence)까지 맞추는 것이 MolmoAct2 쪽 설계**임.

**Molmo2-ER: 백본부터 embodied reasoning 전용**

- 출발점: 일반 VLM은 semantic 이해에 최적화되어 있어서 로봇 policy가 필요한 능력을 못 다룸
  - metric 거리, free space, cross-view object tracking, scene geometry
- Molmo2를 **3.3M 규모 spatial-embodied corpus로 specialize-then-rehearse 방식으로
  finetune**해 Molmo2-ER을 만듦
- 13개 embodied reasoning benchmark 중 9개에서 GPT-5와 Gemini Robot-ER 1.5 Thinking을
  앞섰다고 보고함

**두 가지 action 표현**

- **Continuous**: flow matching으로 부드러운 궤적 생성
- **Discrete**: MolmoAct2-FAST Tokenizer가 **1초 분량의 32차원 continuous action을
  압축된 discrete sequence로** 변환
- `action_mode=both`로 둘을 함께 학습하고, 추론 시 `inference_action_mode`로 선택함
- 이 실험에서는 continuous를 사용함

**MolmoAct2-Think** (이 LeRobot 포트에는 아직 없음)

- 문제 제기: 추론 기반 VLA가 매 step마다 거의 동일한 연산을 반복함
- 해결: **timestep 사이에 변한 scene 영역의 token만 autoregressive하게 예측**하는
  adaptive depth reasoning
- 정적인 scene 비율만큼 latency가 줄어듦
- LeRobot 포트에는 아직 포함되지 않음

## 측정 결과와 구조의 연결

| 관찰 | 구조적 근거 |
|---|---|
| ACT가 학습한 task 100%, 다른 task 0% | 언어 입력 통로가 구조에 없음 |
| ACT가 가장 빠름 (2.7초/episode) | ResNet18 + 5층 transformer, 51.6M |
| SmolVLA가 goal에서 이기고 spatial, libero_10에서 짐 | VLM 앞 16층만 사용하고 visual token을 64개로 제한함. 단순 지시에는 충분하지만 공간 관계와 다단계 추론에는 표현력 부족 |
| MolmoAct2가 어려운 suite에서 앞섬 | 백본이 embodied reasoning 전용(Molmo2-ER)이고, 층별 KV conditioning으로 VLM attention state를 직접 활용함 |
| MolmoAct2가 14GB 사용 | 5B 가중치에 state 256 token이 만드는 긴 context |

**Flow step 수는 방향이 반대임**

- 더 큰 MolmoAct2가 8 step, 작은 SmolVLA가 10 step
- MolmoAct2가 episode당 6.5초로 SmolVLA(4.1초)와 2배 이내 차이를 유지한 데에 이 설정도
  작용함

## 인용

```bibtex
@article{zhao2023learning,
  title   = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author  = {Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal = {arXiv preprint arXiv:2304.13705},
  year    = {2023}
}
```

- SmolVLA: Hugging Face, <https://huggingface.co/blog/smolvla>
- MolmoAct2: Allen AI, <https://allenai.org/blog/molmoact2>

PDF 원문은 `papers/` 에 두었고 저작권을 고려해 git에는 포함하지 않음 (`.gitignore`).
