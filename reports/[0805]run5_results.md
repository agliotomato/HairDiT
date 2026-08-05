run5(densification) 결과

## 요약

- run4 대비 run5는 **밀도 혼합 증강(DensifyAug) + LPIPS 활성 기준(step-warmup → 샘플별 noise-gate)** 두 가지를 동시에 바꾼 실험.
- `[0803]seed_test.md` §5와 동일한 방향오차·시드불일치 지표(structure tensor)를 8장 × seed{1,2,3,42}에 적용한 결과, **run4 epoch15가 6개 조건(run4/run5 × epoch5/10/15) 중 GT 오차·seed 불일치 모두 최선**이고 run5는 epoch15(계획된 최종 지점)에서도 못 따라옴 — seed 불일치는 8/8 이미지 전부 악화(평균 −16.8%p).
- 코드 변경 자체는 설계·로그 실측 모두 정합(§코드 상에서 바꾼 거) — 즉 버그가 아니라 **밀도 증강이 방향 일관성을 실제로 해치는 것으로 보임**. 원인이 밀도 증강 단독인지 LPIPS 변경과의 결합 효과인지는 이 분석만으로 분리 안 됨(run5는 2변수 변경, §결과 해석 시 주의).

## 1. 결과 이미지 및 run4와 비교

run4 vs run5, 둘 다 epoch15(진짜 동일epoch, §2-2 지표와 짝 맞춤). sketch는 conditioning에 실제로
쓰인 `data/test/recolor_sketch`.

### CM_1067
seed42 하단 노이즈 여전히 존재  
seed1, 2, 3모두 머릿결이 더 노이지해짐(헤어 상단부분 머릿결이 run4대비 블러리하고 노이지하게 나타남)  
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1067.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1067.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1067.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1067.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1067.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1067.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1067.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1067.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1067.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1067.png" width="130"> |

### CM_1082
seed42 우측 상단, seed3 좌측 상단 머릿결 노이즈해짐
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1082.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1082.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1082.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1082.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1082.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1082.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1082.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1082.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1082.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1082.png" width="130"> |

### CM_1027
seed42, seed1 상단에서 위 두 케이스에서 지적한 노이즈해짐, 블러리해짐이 가장 잘보임. 머릿결이 엉키는 것과 같이 보이는 현상이 일부 케이스에서 반복

| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1027.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1027.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1027.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1027.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1027.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1027.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1027.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1027.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1027.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1027.png" width="130"> |

### CM_1007
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1007.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1007.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1007.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1007.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1007.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1007.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1007.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1007.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1007.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1007.png" width="130"> |

### CM_1033
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1033.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1033.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1033.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1033.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1033.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1033.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1033.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1033.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1033.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1033.png" width="130"> |

### CM_1068
seed1 좌측하단, seed3 가르마 노이즈
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1068.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1068.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1068.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1068.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1068.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1068.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1068.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1068.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1068.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1068.png" width="130"> |

### CM_1084
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1084.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1084.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1084.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1084.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1084.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1084.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1084.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1084.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1084.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1084.png" width="130"> |

### CM_1172
| run | sketch | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| run4 epoch15 | <img src="../data/test/recolor_sketch/CM_1172.png" width="130"> | <img src="../outputs/0806/run4/42/epoch15/CM_1172.png" width="130"> | <img src="../outputs/0806/run4/1/epoch15/CM_1172.png" width="130"> | <img src="../outputs/0806/run4/2/epoch15/CM_1172.png" width="130"> | <img src="../outputs/0806/run4/3/epoch15/CM_1172.png" width="130"> |
| run5 epoch15 | <img src="../data/test/recolor_sketch/CM_1172.png" width="130"> | <img src="../outputs/0806/run5/42/epoch15/CM_1172.png" width="130"> | <img src="../outputs/0806/run5/1/epoch15/CM_1172.png" width="130"> | <img src="../outputs/0806/run5/2/epoch15/CM_1172.png" width="130"> | <img src="../outputs/0806/run5/3/epoch15/CM_1172.png" width="130"> |

## 2. 방향 오차 / 시드 불일치 평가 ([0803]seed_test.md §5 방법론)

방향오차 / 시드 불일치 지표 data 8장 × seed{1,2,3,42} 6개 조건(run4/run5 × epoch5/10/15) 전체에 적용

### 2-1. macro 평균 (8장)

| run | GT 오차 평균 [deg] | coherence | seed 불일치 [deg] |
|---|---|---|---|
| run4 epoch5  | 20.61 | 0.769 | 18.53±3.70 |
| run4 epoch10 | 19.59 | 0.762 | 16.50±3.00 |
| **run4 epoch15** | **19.29** | **0.761** | **15.80±3.07** |
| run5 epoch5  | 21.99 | 0.731 | 20.33±5.00 |
| run5 epoch10 | 20.09 | 0.745 | 17.75±2.97 |
| run5 epoch15 | 20.83 | 0.708 | 18.43±3.59 |

### 2-2. per-image: run4 epoch15 → run5 epoch15, seed 불일치 변화 (진짜 동일epoch 비교)

| img | run4 ep15 | run5 ep15 | 변화 |
|---|---|---|---|
| CM_1007 | 18.70 | 19.69 | −5.3% |
| CM_1027 | 16.48 | 20.53 | −24.6% |
| CM_1033 | 14.85 | 16.69 | −12.4% |
| CM_1067 | 14.48 | 17.30 | −19.5% |
| CM_1068 | 16.82 | 19.46 | −15.7% |
| CM_1082 | 16.02 | 18.78 | −17.2% |
| CM_1084 | 19.55 | 23.69 | −21.1% |
| CM_1172 | 9.54 | 11.28 | −18.3% |

**8장 전부 예외 없이 악화, 평균 −16.8%p (std 5.9%p, n=8).** 

## 해석
- 정성적으로도 정량적으로도 run5가 run4보다 열세
- 가설(미검증): DensifyAug가 원본 stroke 사이 빈 공간에 보간된 stroke로 색만 채우다 보니, 모델이 그 영역의 방향을 원본 stroke 방향보다 더 평균화/불확실하게 학습했을 가능성. 동일 epoch 대응비교(5·10·15 전부)에서 run5의 coherence가 run4보다 항상 낮음(0.731<0.769, 0.745<0.762, 0.708<0.761)이 방향은 일치하나 인과 검증은 안 됨.

## 코드 상에서 바꾼 거

run4(0730, `configs/lpips_low_phase1.yaml`) 대비 학습에 영향을 주는 변경은 **두 가지**.
나머지는 전부 동일 — 32ch 아키텍처, lr 1e-4, batch 16, warmup 500 step, `w_lpips` 0.002,
scale_sync, flow matte 선형 가중, `epochs: 40`(cosine LR T_max가 묶여 있어 축소하지 않음).

### 1. LPIPS 활성 기준 — 학습 진행도에서 노이즈 레벨로

**변경 내용.** LPIPS를 켜는 기준을 "학습이 얼마나 진행됐는가"에서 "이 샘플이 얼마나
노이즈가 심한가"로 전환.

| | run4 | run5 |
|---|---|---|
| 기준 | 전체 step의 30% 경과 | 샘플의 노이즈 계수 σ ≤ 0.7 |
| 판정 단위 | 배치 전체 on/off | 샘플별 |
| 활성 시점 | step 2244(= epoch 13 시작)부터 | step 1부터 |
| LPIPS가 걸린 epoch | 15 epoch 중 3 | 15 epoch 중 15 |

**근거.** flow matching에서 LPIPS는 `x0 = x_t − σ·v_pred` 로 복원한 이미지에 걸림. σ가 클수록
이 복원이 부정확해, 고노이즈 구간의 perceptual gradient는 신호라기보다 잡음에 근접.
epoch warmup은 이 문제를 "초반에 아예 끄는" 방식으로 우회하지만, 일단 켜진 뒤에는 고노이즈
샘플도 그대로 유입. PixelGen(Eq. 9)은 시간축이 아니라 노이즈 축에서 잘라내며, 그 논문의
τ=0.3이 우리 규약에서 σ ≤ 0.7에 해당. 해당 논문의 ablation 실험에서도 τ=0.1은 효과가
거의 없고 τ=0.6은 성능을 크게 해치는 것으로 보고되어, τ=0.3이 적정 작동 구간임을 뒷받침.

σ ≤ 0.7은 σ의 정의역 [0,1] 중 하위 70% 구간을 남기는 컷오프이지만, 이것이 곧 "샘플의 70%가
활성"을 뜻하지는 않음. 학습에서 σ는 [0,1] 균등분포가 아니라 logit-normal(shift=3.0)
분포로 샘플링되며, 이 분포는 고노이즈 쪽으로 치우쳐 있어 중앙값이 이미 0.751로 컷오프(0.7)
보다 높음. 즉 σ를 하나 뽑으면 절반 이상의 확률로 이미 컷오프를 넘는다는 뜻이므로, 실제
활성 샘플 비율은 CDF로 계산하면 약 **40%**로 예측됨.

**실측**:

| 지표 | 사전 예측 | 실측 | 판정 |
|---|---|---|---|
| 활성 샘플 비율 | 0.401 | **0.4026** (표본 6,008) | 일치 |
| 실효 세기 `R_lpips` | 0.015~0.030 유지 | **중앙 0.0246 / 평균 0.0265** (표본 60, 0.0159~0.0425) | 대역 유지, run4보다 소폭 높음 |

활성 샘플이 40%로 줄고 정규화가 활성 부분집합 평균으로 바뀌었는데도 `w_lpips=0.002`
재조정 없이 목표 대역에 진입. 다만 run4 실측(0.0178~0.0280) 대비 중앙값이 약 0.005 높은 쪽 => 같은 자릿수는 유지

### 2. 학습 데이터 — 밀도 혼합 증강

**변경 내용.** 원본 sketch 하나로만 학습하던 것을, 밀도가 다른 4단계를 epoch마다 번갈아
쓰도록 전환. 순서는 ∞(원본) → T21 → T15 → T12 → 반복이며, **한 epoch 안의 3,000장은
모두 같은 단계**를 사용(샘플별 무작위 아님).

**threshold 세트 선정 근거.**

| 선택 | 이유 |
|---|---|
| ∞(원본) 반드시 포함 | 누락 시 "densified로 학습했으니 densified 입력에서 잘하는 건 당연"이라는 반론이 성립하고, 원본 입력 성능 붕괴 위험 |
| T15 포함 | 8/4~5 추론 검증에서 채택한 작동점 |
| T12까지만 | dose-response가 밀도 0.14 부근부터 포화(그 아래는 개선폭이 잡음 수준) |
| T9 이하 배제 | 포화 구간인 데다 자동 stroke 비중이 커져, "수동 annotation이 hair wisp junction 형성에 필수"라는 SHS §6.4 논지와 충돌할 소지 |

추가 stroke는 **덧붙이기만** 하고 원본 stroke를 대체하지 않음. 색은 가장 가까운 원본
stroke에서 전파하되 **stroke 재착색 이후**에 수행 — 순서가 뒤집히면 추가 stroke만 옛
색을 갖는 불일치 발생. 기하 마스크는 SHS 공개 구현(`getSketchCompletion`)을 그대로 사용하고
`threshold` 외에는 무수정.

**실측.** `densify_t` 기록이 설계표와 그대로 일치.

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 밀도 | ∞ | T21 | T15 | T12 | ∞ | T21 | T15 | T12 | ∞ | T21 | T15 | T12 | ∞ | T21 | **T15** | T12 |

### 결과 해석 시 주의 — run5는 2변수 변경

run5는 run4 대비 **학습 데이터와 LPIPS 활성 규칙 두 가지**가 상이. 따라서 run5의 개선분을
밀도 증강 단독 효과로 귀속 불가. 밀도 단독 효과의 근거는 **재학습 없이 입력만 교체한
추론 검증(8/4~5)** 이 제공하며, run5는 "학습 분포로 흡수한 뒤에도 효과가 유지되는가"를 보는
실험으로 위치시킴.

### 3. GT color recolor
이미 한 stroke 당 한 color로 되어있었음. 이전의 픽셀단위로 recolor되어있게 보였던 이미지(reports/[0803]seed_test.md의 ### 3.1 입력 이미지, CM_1082)는 예전에 픽셀 단위로 recolor하게 만들어놨던 sketch를 그대로 써서 그런 것이고, 학습할 때는 한 stroke 당 한 color를 쓰도록 되어있음
이번 실험부터 추론시에도 해당 방식으로 recolor된 sketch를 쓰도록 변경. 



