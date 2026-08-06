

| Run | Densify | LPIPS 활성 방식 | run4 대비 변경점 | 실험 목적 |
|---|---|---|---|---|
| **run4** | OFF | Step-warmup (기존 방식, epoch 기준 30%) | 기준 조건 | Baseline |
| **run5** | ON | Noise-gate | Densify + LPIPS 활성 기준 동시 변경 | 두 변수의 복합 효과 확인 |
| **run5_1** | OFF | Noise-gate | LPIPS 활성 기준만 변경 | Noise-gate 효과 분리 |
| **run5_2** | ON | Step-warmup (기존 방식, epoch 기준 30%) | Densify만 변경 | Densify 효과 분리 |

※ 네 run 모두 모델 아키텍처와 그 외 학습 조건은 동일하게 유지했으며, run5_1/run5_2는 run4와 동일한 조건에서 한 변수씩 분리해 비교하기 위한 실험이다.

## 요약

## 1. 결과 이미지 비교

### 1.1 run4 vs run5 vs run5_1 vs run5_2

epoch 15 · seed 42 고정, `data/test` 5장(CM_1027 / CM_1067 / CM_1082 / CM_1033 / CM_1068)에 대한 비교다.
좌측부터 입력 스케치, run4(baseline), run5(densify+noise-gate), run5_1(noise-gate만), run5_2(densify만) 순이다.

| image | 스케치 | run4 | run5 | run5_1 | run5_2 |
|---|---|---|---|---|---|
| CM_1027 | ![sketch CM_1027](../data/test/recolor_sketch/CM_1027.png) | ![run4 CM_1027](../outputs/0806/run4/42/epoch15/CM_1027.png) | ![run5 CM_1027](../outputs/0806/run5/42/epoch15/CM_1027.png) | ![run5_1 CM_1027](../outputs/0806/run5_1/42/epoch15/CM_1027.png) | ![run5_2 CM_1027](../outputs/0806/run5_2/42/epoch15/CM_1027.png) |
| CM_1067 | ![sketch CM_1067](../data/test/recolor_sketch/CM_1067.png) | ![run4 CM_1067](../outputs/0806/run4/42/epoch15/CM_1067.png) | ![run5 CM_1067](../outputs/0806/run5/42/epoch15/CM_1067.png) | ![run5_1 CM_1067](../outputs/0806/run5_1/42/epoch15/CM_1067.png) | ![run5_2 CM_1067](../outputs/0806/run5_2/42/epoch15/CM_1067.png) |
| CM_1082 | ![sketch CM_1082](../data/test/recolor_sketch/CM_1082.png) | ![run4 CM_1082](../outputs/0806/run4/42/epoch15/CM_1082.png) | ![run5 CM_1082](../outputs/0806/run5/42/epoch15/CM_1082.png) | ![run5_1 CM_1082](../outputs/0806/run5_1/42/epoch15/CM_1082.png) | ![run5_2 CM_1082](../outputs/0806/run5_2/42/epoch15/CM_1082.png) |
| CM_1033 | ![sketch CM_1033](../data/test/recolor_sketch/CM_1033.png) | ![run4 CM_1033](../outputs/0806/run4/42/epoch15/CM_1033.png) | ![run5 CM_1033](../outputs/0806/run5/42/epoch15/CM_1033.png) | ![run5_1 CM_1033](../outputs/0806/run5_1/42/epoch15/CM_1033.png) | ![run5_2 CM_1033](../outputs/0806/run5_2/42/epoch15/CM_1033.png) |
| CM_1068 | ![sketch CM_1068](../data/test/recolor_sketch/CM_1068.png) | ![run4 CM_1068](../outputs/0806/run4/42/epoch15/CM_1068.png) | ![run5 CM_1068](../outputs/0806/run5/42/epoch15/CM_1068.png) | ![run5_1 CM_1068](../outputs/0806/run5_1/42/epoch15/CM_1068.png) | ![run5_2 CM_1068](../outputs/0806/run5_2/42/epoch15/CM_1068.png) |

※ 입력 스케치는 `data/test/recolor_sketch` — 네 run의 추론에 실제로 conditioning으로 들어간 그 이미지다
(`[0805]results.md` §2의 `infer_custom.py --sketch data/test/recolor_sketch`). densify 미적용 원본이며,
densify는 학습 시 입력 증강으로만 적용됐다.
※ run5_1은 `epoch15`와 `epoch15_infer` 산출물이 동일해(해시 일치) `epoch15` 디렉터리를 사용했다.

### 1.2 seed별 비교

epoch 15 고정, 같은 이미지를 seed `{1, 2, 3, 42}`로 생성한 결과다. 한 행이 하나의 run이므로
**행 내부에서 seed끼리 얼마나 흔들리는지**가 §2의 seed 불일치에 대응한다. 행 라벨의 값은
§2-2의 해당 이미지 seed 불일치[deg]다.

#### CM_1082

| run | seed 1 | seed 2 | seed 3 | seed 42 |
|---|---|---|---|---|
| run4 (16.02°) | ![run4 s1](../outputs/0806/run4/1/epoch15/CM_1082.png) | ![run4 s2](../outputs/0806/run4/2/epoch15/CM_1082.png) | ![run4 s3](../outputs/0806/run4/3/epoch15/CM_1082.png) | ![run4 s42](../outputs/0806/run4/42/epoch15/CM_1082.png) |
| run5 (18.78°) | ![run5 s1](../outputs/0806/run5/1/epoch15/CM_1082.png) | ![run5 s2](../outputs/0806/run5/2/epoch15/CM_1082.png) | ![run5 s3](../outputs/0806/run5/3/epoch15/CM_1082.png) | ![run5 s42](../outputs/0806/run5/42/epoch15/CM_1082.png) |
| run5_1 (14.37°) | ![run5_1 s1](../outputs/0806/run5_1/1/epoch15/CM_1082.png) | ![run5_1 s2](../outputs/0806/run5_1/2/epoch15/CM_1082.png) | ![run5_1 s3](../outputs/0806/run5_1/3/epoch15/CM_1082.png) | ![run5_1 s42](../outputs/0806/run5_1/42/epoch15/CM_1082.png) |
| run5_2 (18.91°) | ![run5_2 s1](../outputs/0806/run5_2/1/epoch15/CM_1082.png) | ![run5_2 s2](../outputs/0806/run5_2/2/epoch15/CM_1082.png) | ![run5_2 s3](../outputs/0806/run5_2/3/epoch15/CM_1082.png) | ![run5_2 s42](../outputs/0806/run5_2/42/epoch15/CM_1082.png) |

#### CM_1068

| run | seed 1 | seed 2 | seed 3 | seed 42 |
|---|---|---|---|---|
| run4 (16.82°) | ![run4 s1](../outputs/0806/run4/1/epoch15/CM_1068.png) | ![run4 s2](../outputs/0806/run4/2/epoch15/CM_1068.png) | ![run4 s3](../outputs/0806/run4/3/epoch15/CM_1068.png) | ![run4 s42](../outputs/0806/run4/42/epoch15/CM_1068.png) |
| run5 (19.46°) | ![run5 s1](../outputs/0806/run5/1/epoch15/CM_1068.png) | ![run5 s2](../outputs/0806/run5/2/epoch15/CM_1068.png) | ![run5 s3](../outputs/0806/run5/3/epoch15/CM_1068.png) | ![run5 s42](../outputs/0806/run5/42/epoch15/CM_1068.png) |
| run5_1 (14.89°) | ![run5_1 s1](../outputs/0806/run5_1/1/epoch15/CM_1068.png) | ![run5_1 s2](../outputs/0806/run5_1/2/epoch15/CM_1068.png) | ![run5_1 s3](../outputs/0806/run5_1/3/epoch15/CM_1068.png) | ![run5_1 s42](../outputs/0806/run5_1/42/epoch15/CM_1068.png) |
| run5_2 (19.96°) | ![run5_2 s1](../outputs/0806/run5_2/1/epoch15/CM_1068.png) | ![run5_2 s2](../outputs/0806/run5_2/2/epoch15/CM_1068.png) | ![run5_2 s3](../outputs/0806/run5_2/3/epoch15/CM_1068.png) | ![run5_2 s42](../outputs/0806/run5_2/42/epoch15/CM_1068.png) |

두 이미지는 8장 중 `run4 → run5_1` 개선폭과 `run4 → run5_2` 악화폭이 함께 큰 사례로 골랐다
(CM_1082: −1.65 / +2.89, CM_1068: −1.93 / +3.14). 정성 비교는 선택된 사례에 대한 예시이며,
판정 근거는 §2-1의 8장 macro 평균이다.

※ seed 간 차이 중 육안으로 가장 먼저 보이는 것은 모발 **색상**이지만, §2의 seed 불일치는 hair
matte 내부의 structure tensor **방향**만 coherence 가중으로 측정한 값이므로 색 변화는 지표에
반영되지 않는다. 이 표는 방향 안정성의 참고 자료로만 본다.



## 2. 방향 오차 / 시드 불일치 평가 ([0803]seed_test.md §5 방법론)

`[0803]seed_test.md` §5의 structure tensor 지표를 그대로 적용했다. 평가 대상은
`data/test` 8장 × seed `{1, 2, 3, 42}`이며, `sigma_i=3`, `erode_px=6`을 사용했다.

- **GT 오차**: 생성 이미지와 GT 원본의 평균 방향 오차(도)
- **coherence**: 생성 이미지의 방향 선명도
- **seed 불일치**: 같은 run·epoch에서 4개 seed의 6개 쌍을 직접 비교한 평균 방향 차이(도)

### 2-1. Macro 평균 (8장)

| run | epoch | GT 오차 평균 [deg] | coherence | seed 불일치 [deg] |
|---|---:|---:|---:|---:|
| run4 | 5 | 20.61 | 0.769 | 18.53±3.70 |
| run4 | 10 | 19.59 | 0.762 | 16.50±3.00 |
| **run4** | **15** | **19.29** | **0.761** | **15.80±3.07** |
| run5 | 5 | 21.99 | 0.731 | 20.33±5.00 |
| run5 | 10 | 20.09 | 0.745 | 17.75±2.97 |
| run5 | 15 | 20.83 | 0.708 | 18.43±3.59 |
| run5_1 | 5 | 19.47 | 0.789 | 15.73±3.12 |
| run5_1 | 10 | 19.37 | 0.758 | 15.78±2.95 |
| **run5_1** | **15** | **18.10** | **0.770** | **14.09±2.52** |
| run5_2 | 5 | 21.54 | 0.720 | 18.39±3.67 |
| run5_2 | 10 | 20.49 | 0.739 | 16.99±3.19 |
| run5_2 | 15 | 20.57 | 0.724 | 17.98±3.19 |

### 2-2. Epoch 15 per-image 비교

| image | run4 GT/seed | run5 GT/seed | run5_1 GT/seed | run5_2 GT/seed |
|---|---:|---:|---:|---:|
| CM_1007 | 18.92 / 18.70 | 20.08 / 19.69 | **16.87 / 15.72** | 19.66 / 20.07 |
| CM_1027 | 19.04 / 16.48 | 21.22 / 20.53 | **17.19 / 14.14** | 20.69 / 18.83 |
| CM_1033 | 16.35 / 14.85 | 17.74 / 16.69 | **15.98 / 14.17** | 16.70 / 15.91 |
| CM_1067 | 17.51 / 14.48 | 18.98 / 17.30 | **16.93 / 13.38** | 19.20 / 17.54 |
| CM_1068 | 18.32 / 16.82 | 19.94 / 19.46 | **16.98 / 14.89** | 20.11 / 19.96 |
| CM_1082 | 17.66 / 16.02 | 19.25 / 18.78 | **16.69 / 14.37** | 18.90 / 18.91 |
| CM_1084 | 19.92 / 19.55 | 22.75 / 23.69 | **17.59 / 17.43** | 22.41 / 21.38 |
| CM_1172 | 26.61 / 9.54 | 26.73 / 11.28 | **26.59 / 8.66** | 26.93 / 11.23 |


### 2-3. 해석

- epoch15 기준 `run5_1`은 run4보다 GT 오차가 **19.29 → 18.10°**, seed 불일치가
  **15.80 → 14.09°**로 모두 낮고, coherence도 **0.761 → 0.770**으로 높다. 따라서
  densify를 끈 상태에서 noise-gate를 적용한 run5_1은 방향 안정성 측면에서 오히려 안정되었다
- `run5_2`는 run4 대비 epoch15에서 GT 오차가 **19.29 → 20.57°**, coherence가
  **0.761 → 0.724**, seed 불일치가 **15.80 → 17.98°**로 악화됐다.
- 따라서 이번 8장·4 seed 평가에서는 **run5_2의 열화가 densify 축에서 재현**됐다. 반면
  run5_1은 run4보다 오히려 개선되어, run5에서 관찰된 방향 불안정성을 noise-gate 단독
  효과로 설명하기 어렵다.
- 단, 표본은 8장이고 seed 불일치는 상대 지표이므로, 이를 모든 입력에 대한 절대적 인과
  증명으로 해석하지 않고 이번 평가 범위의 ablation 결과로 해석한다.


## 3. 분석

### 3.1 DensifyAug의 방향성 영향

`run4`↔`run5_2`는 densify만, `run4`↔`run5_1`은 LPIPS 활성 규칙만 다르므로 epoch15 비교로
두 변수를 분리할 수 있다.

```text
run5_2 − run4:   GT 방향 오차  19.29 → 20.57°  (+1.28)
                 coherence      0.761 → 0.724   (−0.037)
                 seed 불일치   15.80 → 17.98°  (+2.18)
```

8개 이미지 모두 같은 방향이고, densify를 끈 `run5_1`은 반대로 개선됐다. 따라서 run5의 방향
불안정성과 coherence 저하를 noise-gate 단독으로 설명하기 어렵고, **DensifyAug가 주된 열화
요인**으로 판단한다.

가능한 메커니즘은 densification이 원본 stroke 사이 빈 공간에 보간 stroke를 넣으면서
annotation에 없던 영역의 방향까지 조건으로 학습시킨다는 것이다. 서로 다른 방향의 stroke
사이가 평균화되거나 연결된 구조로 보이면, 생성 시 방향 prior와 seed noise가 개입할 여지가
오히려 커진다. 입력 진단에서도 T21→T15→T12로 갈수록 stroke density와 connected component
변화가 커져, densify가 선 두께가 아니라 입력 기하 자체를 바꾼다는 점을 확인했다.

### 3.2 Noise-gate의 영향 — 개선은 사실이나 "게이트 위치" 효과가 아니다

epoch15에서 `run5_1`은 GT 오차 19.29→18.10°, seed 불일치 15.80→14.09°, coherence
0.761→0.770으로 세 지표 모두 개선됐다. **noise-gate가 방향성 열화를 유발하지 않았다**는 것이
§2가 직접 지지하는 진술이다.

다만 게이트를 바꾸면 적용 위치와 **누적 투여량이 함께** 바뀐다 — `run5_1`은 `run4`의 5배
LPIPS 노출을 받았다(§3.3-a). `w_lpips` 값이 같다고 "위치만 바뀌었다"고 할 수 없다.

2×2를 교호작용까지 분해하면:

```text
게이트 효과:   densify OFF   run5_1 − run4   = −1.19°   (개선)
               densify ON    run5   − run5_2 = +0.26°   (소멸)
densify 효과:  warmup        run5_2 − run4   = +1.28°
               noise-gate    run5   − run5_1 = +2.73°   (2배)
```

두 변수는 가법적이지 않고, LPIPS 노출 증가가 densified sketch의 악영향까지 증폭한다. 정확한
기술은 **"densify를 끈 조건에서 LPIPS 노출을 늘렸더니 방향 지표 세 개가 모두 개선됐다"** 이고,
남는 조합은 **densify OFF + LPIPS 증량**이다.

### 3.3 LPIPS 영향에 대한 해석상의 주의

LPIPS는 방향성을 직접 최적화하는 손실이 아니라 복원 이미지의 perceptual feature를 제한할
뿐이므로, 방향 지표의 변화는 간접적 결과다. 또한 `run5`는 2변수 동시 변경이라 그 열화만으로
게이트를 판단할 수 없다 — 이번 2×2에서 게이트 축은 `run4→run5_1`, densify 축은
`run4→run5_2`다.

가장 중요한 주의는 **`w_lpips=0.002`가 step-warmup 기준으로 보정된 값**이라는 점이다.
아래 (a)~(e)는 그 값이 게이트 교체 이후 근거를 잃었다는 분석이다.

#### (a) 게이트 교체는 세기를 유지한 변경이 아니었다 — 누적 5배

`w_lpips`는 손대지 않았고 config에도 "재보정하지 않는다"고 적혀 있지만
(`configs/run5_2_densify_phase1.yaml:61-62`), `losses.py:304-323`의 두 분기는 LPIPS가 걸리는
총량이 다르다.

| | 활성 시점 | 배치 내 적용 | epoch15까지 누적 |
|---|---|---:|---:|
| warmup (run4 / run5_2) | `int(0.3×40×187)=step 2244` = **epoch 13** | 전체 배치 | **3 epoch** |
| noise-gate (run5 / run5_1) | step 0부터 상시 | σ≤0.7 부분집합 | **15 epoch** |

`logs/run5_2_densify_phase1.log:58`에서 `lpips_active_fraction`이 `Epoch 13/40` 첫 스텝에
`1.0000`으로 켜지는 것으로 확인된다(noise-gate run은 전 구간 0.40 상시).
**epoch15 시점에서 `run5_1`은 `run4`의 5.0배 노출을 받았다.**

여기에 `losses.py:343-347`이 활성 **부분집합의 mean**을 취해(warmup은 전체 배치 mean) 활성
샘플당 가중치가 `w/(0.40·B)`로 **2.5배**가 된다. 코드가 `[0801]loss.md` §2-T2의
T2-b(`w ← w/게이트율`)를 이미 암묵 수행 중이라, config 주석과 실제 거동이 어긋난다.

#### (b) 그런데 `R_lpips`는 이 차이를 못 잡는다

| run | 게이트 | `R_lpips` 평균 | 중앙값 | 범위 | n |
|---|---|---:|---:|---|---:|
| run4 | warmup | 0.0225 | 0.0222 | 0.0178~0.0287 | 52 |
| run5 | noise-gate | 0.0265 | 0.0249 | 0.0159~0.0425 | 29 |
| run5_1 | noise-gate | 0.0268 | 0.0262 | 0.0181~0.0483 | 28 |
| run5_2 | warmup | 0.0233 | 0.0238 | 0.0217~0.0244 | 7 |

(네 run 모두 `w_lpips=0.002`. run5_2의 n=7은 LPIPS가 epoch 13부터만 켜져 그 전에는 `R`이
계산되지 않기 때문이다 — `losses.py:360`.)

`x0_pred = x_t − σ·v_pred`(`losses.py:333`)라 LPIPS 그래디언트에 σ가 곱해지는데, 게이트는
σ가 작은 샘플만 남기므로 부분집합 집중(2.5배)과 σ 감소가 상쇄한다. 실측 σ 분포로 계산하면
`√(1/0.40) × √(0.303/0.542) ≈ 1.18`로 표의 변화(0.0225→0.0268, 1.19배)와 일치한다
(`P(σ≤0.7)=0.4002`, 로그 실측 0.4026과 일치).

**즉 `R_lpips`는 순간 그래디언트 비라서 활성 구간이 3 epoch인지 15 epoch인지를 반영하지
못한다.** "R이 밴드 안이니 세기는 동일하다"는 착수 체크리스트의 전제는 누적 투여량에 대해
성립하지 않는다.

#### (c) 탐색된 적 없는 구간 — `R ∈ (0.02, 1.0)`

| run | `R` | `w_lpips` | 관측 |
|---|---:|---:|---|
| run3 | 1.016 | 0.1 | frizz. LPIPS 활성 epoch에서 flow loss 반등(`[0730]results.md` §3-2) |
| run4 / 0730 | ~0.022 | 0.002 | frizz 소멸, 대신 가닥 뭉개짐·방향 흐트러짐 |

`[0729]retrain_plan_v2.md` §2-2가 중간 구간(`R=0.2~0.5`)을 선택지 ②로 적어두고 "재현 성공 후
이분탐색"으로 미뤘으나 **실행되지 않았고, 50배 간격이 그대로 비어 있다.**

`[0801]loss.md` §0·§5는 이 축을 "frizz↔blur 양 끝"으로 보고 R 미세조정 런을 기각했다.
**그러나 그 진단은 게이트 도입 이전 데이터다.** `R=1.0`의 frizz는 σ≥0.75에서 품질 낮은
`x0_pred`에 디테일을 요구한 결과로 해석됐는데(`[0801]loss.md` §1-3), noise-gate가 제거한 것이
정확히 그 구간이다 — 기각 근거가 그대로 적용되지 않는다.

#### (d) 외부 논문의 수치는 이식되지 않는다

**PixelGen은 배관이 다르다.** Algorithm 1(p.15)의 `xθ = netθ(xt,t,c)`에서 보듯 **pixel
diffusion + x-prediction**이라 LPIPS 그래디언트가 출력에 직행한다. 우리는 latent +
v-prediction이라 `(−σ) × VAE decoder Jacobian`을 경유하고 matte 마스킹(`losses.py:107-108`,
머리를 검은 배경에 오려붙여 VGG에 투입)까지 붙는다. 분모도 pixel L2 vs
latent L2/`s≈37`(`losses.py:272-274`)로 다르다. 실제로 `λ1=0.1`을 그대로 넣었을 때 우리 실측
`R`은 **1.016**이었다(`[0729]retrain_plan_v2.md` §0) — 저쪽 보조항이 우리에겐 flow와 동급이
된다. 게다가 그 값은 P-DINO·REPA가 함께 있는 3항 목적함수에서 튜닝됐다(Table 5b~5d).
문턱도 `timeshift=1.0` 전제라, `shift=3.0`인 우리에게 `σ≤0.7`은 스케줄 기준 마지막
**43.75%**(샘플 40%)이지 PixelGen의 70%가 아니다.

**LPL은 배관이 맞는 유일한 선례다** — [arXiv:2411.04873](https://arxiv.org/abs/2411.04873),
latent diffusion + VAE decoder 경유 + ε/v/flow 전부에서 검증, `[0801]loss.md` §2-T1의 1순위
후보. `w_LPL≈3.0`으로 저자 표현상 **"전체 loss의 약 1/5 기여"**, 게이트는 SNR 문턱
`τ_σ∈[3,6]`. 그러나 우리 σ 분포에 대입하면 `τ_σ=3 → σ≤0.366`(통과율 4.95%),
`τ_σ=6 → 2.29%`로 사실상 LPIPS가 꺼진다.

**이식되는 것은 숫자가 아니라 목표 수준이다** — LPL은 perceptual을 전체 loss의 1/5로 놓고
우리는 그래디언트 비 2%다. 측정량이 달라(loss 비 vs grad-norm 비) 직접 비교는 불가하지만 두
논문 모두 우리보다 한 자릿수 이상 강하다. 따라서 **"방향이 100% 일치하는 논문을 찾아 그대로
채택"하는 경로는 성립하지 않는다.** 실효 세기는 예측 타깃·공간·`scale_sync` 제수·마스킹·
σ 분포·게이트의 mean 규약 전부에 의존하는 파이프라인 고유량이고, 이식 가능한 것은 ① 설계의
형태(perceptual을 저노이즈에 한정)와 ② 측정 프로토콜(`R`을 재서 환산)뿐이며 둘 다 이미 있다.

#### (e) 제안 — 게이트 고정, 세기만 스윕

`R`은 `w`에 선형이므로 run5_1 실측(`w=0.002` → `R=0.0268`)에서 환산한다.

| 조건 | 목표 `R` | `w_lpips` | 성격 |
|---|---:|---:|---|
| run5_1 (기준) | 0.027 | 0.002 | 현재 최선 조건 |
| **run6_1** | ~0.10 | **0.0075** | 현행 3.7배, 안전한 첫 걸음 |
| **run6_2** | ~0.30 | **0.022** | `[0729]` ②구간, LPL "1/5 기여"에 근접 |

- 두 run 모두 densify OFF + noise-gate 유지(run5_1에서 `lpips` 한 줄만 변경). 정규화 규약은
  같이 바꾸지 않는다 — 실효 세기를 바꾸지 않는 재매개화라 변수만 늘어난다.
- 판정은 §2의 8장×4 seed 방향 지표 + `[0805]results.md` §3의 색·구조 지표. `w`가 커지면
  `gradient_clip=1.0` 발동 빈도가 늘 수 있어 함께 기록한다.
- 로깅 보강: 지금 남는 건 순간 비뿐이라 (a)의 5배 차이가 드러나지 않는다. **누적 LPIPS/flow
  gradient 비**를 추가해야 run 간 비교가 가능하다.

**한계** — ① 2×2는 셀당 n=1이고 학습 시드가 미고정이며(`[0805]results.md` §6-4) `run5_1`은
투여량과 위치가 동시에 바뀐 조건이라, 위 논거는 부호를 설명하는 메커니즘이지 인과 증명이
아니다. ② (b)의 1.18 추정은 decoder Jacobian이 σ에 무관하다고 본 값으로, PixelGen
Appendix A Fig.6(b)에 따르면 noise-gate 쪽에 유리하게 편향돼 있다. ③ 평가셋 8장이고, 방향
지표만으로는 frizz와 sharpness를 구별하지 못하므로 `R`을 올릴 때 정성 관찰이 필요하다.

### 3.4 결론

1. `run5_2`의 densify 조건은 방향 오차·coherence·seed 불일치 모두에서 열화됐다.
2. `run5_1` 조건은 방향 지표상 run4보다 개선됐다. 다만 게이트 위치와 누적 LPIPS 노출량(5배)이
   함께 바뀐 것이므로, 개선을 게이트 위치 단독 효과로 귀속할 수 없다.
3. 따라서 run5의 노이즈·머릿결 엉킴은 noise-gate보다 densification과 더 강하게 연관된다.
   두 변수는 가법적이지 않고, densify의 손상은 LPIPS 노출이 늘어난 조건에서 약 2배로 커진다.
4. DensifyAug 내부의 구체적 원인은 추가 통제 실험 없이는 확정할 수 없다.
5. LPIPS의 최적 가중치도 미확정이며, 지금까지 밟아본 실효 세기는 `R≈0.02`와 `R≈1.0` 두 점뿐이다.
   그 사이 구간은 게이트 도입 이후 한 번도 평가되지 않았다(§3.3-c).
