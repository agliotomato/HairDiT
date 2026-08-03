# Seed에 따른 Hair Directionality 비교

## 0. 요약
1. run2의 phase1에서 방향 노이즈 안나오는 건 seed고정이 안된 것이었음 seed42로 고정하니 방향 노이즈 나옴
2. mcs2의 결과 다른 seed(1, 2, 3)에 대해 실험해봤을 때에도 이미지가 잘 나옴
3. run2p1, run4에서 seed에 따라 방향성 불일치 정도가 달라짐(현재로서는 seed 42에서만 방향 불일치 확인)

## 1. 목적

Sparse한 hair stroke 조건에서 stroke 사이의 머리카락 방향은 모델의 prior에 의해 결정될 수 있으며, 이 과정이 random seed의 영향을 받는지 확인한다.

특히 다음 가설을 검증한다.

1. hair directionality 문제는 특정 random seed에서만 두드러질 수 있다.
2. mcs2의 기존 결과가 상대적으로 좋은 seed에서 생성된 결과일 가능성이 있다.
3. run2p1, run4에서도 seed에 따라 방향성 불일치 정도가 달라질 수 있다.

---

## 2. 비교 대상

다음 세 체크포인트를 비교한다.

| | mcs2 (run1) | run2 (0720) | run4 (0730, 현재실험) |
|---|---|---|---|
| 아키텍처 | 17ch, matte_feat 비-zero-init | 32ch, B_matte zero-init | 32ch, B_matte zero-init (run2와 동일) |
| timestep → DiT | raw σ (0~1, prior 무력화) | σ×1000 (prior 정상) | σ×1000 |
| phase1 데이터 | unbraid 3000, 187 step/ep | unbraid+braid 6000장, 375 step/ep | unbraid 3000, 187 step/ep |
| phase2 데이터 | braid 1000 | phase1과 동일(both) | **없음 — phase1만 학습** |
| LR (phase1) | 1e-4 | 1e-4 | 1e-4 |
| flow 항 | `Σ(m·d²)/N` | `Σ(m²·d²)/Σm` (scale-sync 없음) | `Σ(m·d²)/Σm ÷ s` (scale-sync + matte 선형 가중 `m²→m` 복원) |
| **lpips 실효 세기 `R`** | **≈0.9** | **≈0.018** (flow가 55× 압도) | **≈0.022** (실측 0.018~0.028) |
| 본 실험에 사용한 체크포인트 | phase2 최종 (로컬 없음) | `checkpoints/run2_phase1/epoch_30_infer.pth` | `checkpoints/run4_phase1/epoch_30_infer.pth` |

---

## 3. 실험 조건

### 3.1 입력 이미지

방향성 문제가 명확하게 관찰되는 `data/paper/`의 이미지 2장을 사용한다.

| ID | img (face) | matte | sketch (gt) |
|---|---|---|---|---|
| CM_1067 | <img src="../data/paper/img/CM_1067.png" width="110"> | <img src="../data/paper/matt/CM_1067.png" width="110"> | <img src="../data/paper/sketch_gt/CM_1067.png" width="110"> |
| CM_1082 | <img src="../data/paper/img/CM_1082.png" width="110"> | <img src="../data/paper/matt/CM_1082.png" width="110"> | <img src="../data/paper/sketch_gt/CM_1082.png" width="110"> |


### 3.2 Random seed

각 체크포인트 및 이미지에 대해 다음 4개 seed로 inference를 수행한다.

```text
seed = 42, 1, 2, 3
```

---

## 4. 정성 비교

> run2p1·run4는 2026-08-03 재생성 — `epoch_30_infer.pth`, `num_steps 28 / bld_mode full / bld_soft_steps 24 / pixel_blend α=0.75`, config `joint_phase1.yaml`. seed 이외 조건은 전부 동일.
> mcs2 행은 기존 렌더를 그대로 둔 것이다(로컬에 체크포인트 없음).

## 4.1 Image 1 (CM_1067)

| 모델 | Seed 42 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| **mcs2** | <img src="../outputs/0803/seed_mcs2/42/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_mcs2/1/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_mcs2/2/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_mcs2/3/CM_1067.png" width="180"> |
| **run2p1** | <img src="../outputs/0803/seed_run2/42/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run2/1/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run2/2/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run2/3/CM_1067.png" width="180"> |
| **run4** | <img src="../outputs/0803/seed_run4/42/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run4/1/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run4/2/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run4/3/CM_1067.png" width="180"> |

---

## 4.2 Image 2 (CM_1082)

| 모델 | Seed 42 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| **mcs2** | <img src="../outputs/0803/seed_mcs2/42/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_mcs2/1/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_mcs2/2/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_mcs2/3/CM_1082.png" width="180"> |
| **run2p1** | <img src="../outputs/0803/seed_run2/42/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run2/1/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run2/2/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run2/3/CM_1082.png" width="180"> |
| **run4** | <img src="../outputs/0803/seed_run4/42/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run4/1/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run4/2/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run4/3/CM_1082.png" width="180"> |

> `data/paper/img/CM_1082.png`(face 입력)가 교체된 뒤 세 모델 모두 새 입력으로 재렌더됐다 — 세 행 비교 유효.


---

## 4.3 참고 — run4 colorful sketch 입력

같은 run4 epoch30 체크포인트에 colorful sketch(`data/paper/sketch/`)를 넣은 seed 스윕. 방향성 판정은 §4.1~4.2(gt sketch)로 하고, 여기서는 **색 반영이 seed에 따라 흔들리는지**만 참고로 본다.

| 이미지 | Seed 42 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| CM_1067 | <img src="../outputs/0803/seed_run4_color/42/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run4_color/1/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run4_color/2/CM_1067.png" width="180"> | <img src="../outputs/0803/seed_run4_color/3/CM_1067.png" width="180"> |
| CM_1082 | <img src="../outputs/0803/seed_run4_color/42/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run4_color/1/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run4_color/2/CM_1082.png" width="180"> | <img src="../outputs/0803/seed_run4_color/3/CM_1082.png" width="180"> |


---

## 5. 분석

### 5.1 Seed 영향

* 동일한 입력과 체크포인트에서도 seed에 따라 stroke 사이에 생성되는 머리카락 방향이 달라졌다.
* 이는 sparse한 stroke만으로는 중간 영역의 방향성이 충분히 제한되지 않으며, 해당 영역이 초기 noise와 모델 prior의 영향을 받기 때문으로 해석할 수 있다.
* 다만 seed 영향의 크기는 체크포인트별로 차이가 있었다.

### 5.2 mcs2의 lucky seed 가능성

* mcs2가 모든 seed에서 일관되게 좋은 결과를 보이는지 확인한다.
* seed 42에서만 상대적으로 좋은 결과를 보이고 다른 seed에서 방향성 문제가 증가한다면, 기존 mcs2 결과가 lucky seed였을 가능성이 있다.
* 반대로 여러 seed에서 안정적인 결과를 보인다면, mcs2의 성능 차이는 seed 이외의 학습 방식이나 데이터 처리 차이에서 발생했을 가능성이 높다.

### 5.3 run2p1과 run4 비교

* run2p1에서 과거 방향성 문제가 보고되지 않았던 것이 모델 자체의 안정성 때문인지, 당시 사용한 이미지 및 seed 조합 때문인지 구분한다.
* run4의 LPIPS 조정은 푸석거림 개선에는 영향을 주었으나, 방향성 문제는 별개의 요인일 가능성이 있다.
* run4에서도 특정 seed에서만 방향성 문제가 발생한다면, LPIPS보다는 sparse stroke 조건과 prior 의존성이 주요 원인으로 판단된다.

---

## 6. 결론

현재 결과를 통해 다음을 확인하였다.

1. 푸석거림 및 고주파 뻣뻣함 문제는 LPIPS 가중치 조정 후 개선되었다.
2. Hair directionality는 동일 체크포인트에서도 random seed에 따라 달라질 수 있다.
3. 특히 sparse stroke 사이 영역은 입력 조건보다 모델 prior 및 초기 noise의 영향을 상대적으로 크게 받는 것으로 보인다.
4. mcs2가 seed 전반에서 안정적인지, 특정 seed에서만 좋은 결과를 생성하는지는 본 비교 결과를 통해 판단해야 한다.
5. 이후 hair directionality 불일치를 수치화할 수 있는 정량 평가를 추가할 예정이다.

---

## 8. 추가 확인 사항

* [ ] SHS 논문에서 hair stroke를 dense하게 구성하는 전처리 또는 augmentation이 사용되었는지 확인
* [ ] mcs2 초기 학습 코드 및 데이터 전처리 과정에서 stroke densification이 적용되었는지 확인
* [ ] Hair directionality 정량 평가 방법 적용
* [ ] 필요 시 seed 수와 평가 이미지 수 확대
