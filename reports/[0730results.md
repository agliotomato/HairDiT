# [2026-07-26] 학습 결과 및 분석

> reports/[0729]retrain_plan_v2.md에 따라 학습 진행한 결과 리포트

## 요약
- **고주파 뻣뻣함(frizz) 해결** — LPIPS 실효 세기 R을 run2 수준(R≈0.02)으로 낮춘 효과로 추정
- **헤어 방향 노이즈는 잔존** — run3보다는 약하지만 완전히 사라지지 않음, run2의 phase2 양상과 유사

## 결과 사진

> seed42기준. phase1 epoch5/10/15/20/25/30/35/40 비교.

### gt sketch

| 파일명 | img | sketch | epoch5 | epoch10 | epoch15 | epoch20 | epoch25 | epoch30 | epoch35 | epoch40 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM_1067 | <img src="../data/paper/img/CM_1067.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1067.png" width="70"> |
| CM_1082 | <img src="../data/paper/img/CM_1082.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1082.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1082.png" width="70"> |
| CM_1068 | <img src="../data/paper/img/CM_1068.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1068.png" width="70"> |
| CM_1172 | <img src="../data/paper/img/CM_1172.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch5/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch10/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch15/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch20/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch25/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch30/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch35/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch40/CM_1172.png" width="70"> |


### Colorful sketch

| 파일명 | img | sketch | epoch5 | epoch10 | epoch15 | epoch20 | epoch25 | epoch30 | epoch35 | epoch40 |
|---|---|---|---|---|---|---|---|---|---|---|
| CM_1067 | <img src="../data/paper/img/CM_1067.png" width="70"> | <img src="../data/paper/sketch_gt/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1067.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1067.png" width="70"> |
| CM_1068 | <img src="../data/paper/img/CM_1068.png" width="70"> | <img src="../data/paper/sketch/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1068.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1068.png" width="70"> |
| CM_1172 | <img src="../data/paper/img/CM_1172.png" width="70"> | <img src="../data/paper/sketch/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1172.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1172.png" width="70"> |
| braid_2625 | <img src="../data/paper/img/braid_2625.png" width="70"> | <img src="../data/paper/sketch/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch5_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch10_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch15_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch20_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch25_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch30_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch35_color/braid_2625.png" width="70"> | <img src="../outputs/0730/epoch40_color/braid_2625.png" width="70"> |
| CM_1084_1 | <img src="../data/unbraid_new/img/CM_1084_1.png" width="70"> | <img src="../data/unbraid_new/sketch/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch5_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch10_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch15_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch20_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch25_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch30_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch35_color/CM_1084_1.png" width="70"> | <img src="../outputs/0730/epoch40_color/CM_1084_1.png" width="70"> |


## 1. 고주파 뻣뻣함(frizz) 해결

run3에서 헤어가 뻣뻣하게 나오는 문제 해결 - lpips 영향 낮춘 것의 결과

| | epoch10 | epoch20 | epoch30 | epoch40 |
|---|---|---|---|---|
| **run3** | <img src="../outputs/0725_phase1/epoch10/seed42/paper/sketch_gt/CM_1067.png" width="140"> | <img src="../outputs/0725_phase1/epoch20/seed42/paper/sketch_gt/CM_1067.png" width="140"> | <img src="../outputs/0725_phase1/epoch30/seed42/paper/sketch_gt/CM_1067.png" width="140"> | <img src="../outputs/0725_phase1/epoch40/seed42/paper/sketch_gt/CM_1067.png" width="140"> |
| **이번 실험** | <img src="../outputs/0730/epoch10/CM_1067.png" width="140"> | <img src="../outputs/0730/epoch20/CM_1067.png" width="140"> | <img src="../outputs/0730/epoch30/CM_1067.png" width="140"> | <img src="../outputs/0730/epoch40/CM_1067.png" width="140"> |



## 2. 헤어 방향 노이즈는 잔존

그럼에도 헤어 영역의 방향 노이즈 잔존, run2와 같은 양상. 낮은 lpips + flow loss만으로 해결되지 않음

| | 초반 | 중반 | 후반 |
|---|---|---|---|
| **run2 phase2** (ep10/15/20) | <img src="../outputs/results/joint_phase2_epoch10/sketch_gt/CM_1067.png" width="140"> | <img src="../outputs/results/joint_phase2_epoch15/sketch_gt/CM_1067.png" width="140"> | <img src="../outputs/results/joint_phase2_epoch20/sketch_gt/CM_1067.png" width="140"> |
| **이번 실험 phase1** (ep20/25/30) | <img src="../outputs/0730/epoch20/CM_1067.png" width="140"> | <img src="../outputs/0730/epoch25/CM_1067.png" width="140"> | <img src="../outputs/0730/epoch30/CM_1067.png" width="140"> |
| **mcs2** (기준, 정렬+선명) | <img src="../outputs/figure/hair-dit_mcs2/gt/CM_1067.png" width="140"> | <img src="../outputs/figure/hair-dit_mcs2/gt/CM_1067.png" width="140"> | <img src="../outputs/figure/hair-dit_mcs2/gt/CM_1067.png" width="140"> |



## 3. loss추이

> `logs/lpips_low.log`·`logs/perceptual_lpips_low.log` 기준.
> **로컬 로그가 ep24까지만 동기화돼 있어 ep25~40 구간은 미확인** (렌더는 ep40까지 있음).

### 3-1. 의도한 LPIPS 세기가 실제로 걸렸는지

| 항목 | 목표 | 실측 |
|---|---|---|
| `R_lpips` (22회 로깅) | 0.015~0.025 | min **0.0178** / mean **0.0221** / max 0.0280 |
| `s_raw` | 28~51 (run3 밴드) | 37.0 (전 epoch 안정) |
| `clamp_hi` / `clamp_lo` | 0% | 0.000 / 0.000 |
| LPIPS 활성 | step 2244 = ep13 | ep13부터 `loss_lpips` 등장 |

→ run2 실측 `R≈0.018`과 같은 급. **조건은 설계대로 걸렸으므로 결과는 신뢰 가능.**

### 3-2. flow loss — LPIPS 켜진 시점에서 run3와 갈림

`loss_flow_eq12`(= `Σ(m·d²)/Σm`, scale-sync 전 값) epoch 평균.

| ep | 이번 실험 | Δ(ep1 대비) | run3 | Δ(ep2 대비) |
|---|---|---|---|---|
| 2 | 4.124 | −16.3% | 4.053 | 0.0% |
| 5 | 3.850 | −21.9% | 3.734 | −7.9% |
| 10 | 3.784 | −23.2% | 3.685 | −9.1% |
| **12** (LPIPS off 마지막) | 3.734 | −24.2% | **3.610** | **−10.9%** |
| **13** (LPIPS on) | **3.729** | −24.3% | **3.689** | **−9.0%** ↑ |
| 15 | 3.728 | −24.4% | 3.713 | −8.4% ↑ |
| 20 | 3.637 | −26.2% | 3.561 | −12.1% |
| 24 | 3.547 | −28.0% | 3.563 | −12.1% |

- **run3는 LPIPS가 켜지는 ep13에서 flow loss가 반등**(3.610 → 3.689 → 3.713, ep15까지 +2.9%). `R≈1.0`의 LPIPS가 flow 항을 밀어낸 것.
- **이번 실험은 반등 없이 계속 하강**(3.734 → 3.729 → 3.728). `R≈0.02`라 LPIPS가 flow를 못 밀어냄.
- `loss_lpips` 자체는 0.2525(ep13) → 0.2464(ep24)로 거의 안 움직임. 기여분이 `0.002 × 0.246 ≈ 0.0005`라 `loss_total`(0.1012+0.0005)에 사실상 영향 없음.(run2와 동일)

### 3-3. step 매칭 비교 (run2 phase1)

run2는 375 step/epoch라 epoch 라벨로는 직접 비교 불가 — 글로벌 스텝을 맞추면:

| 글로벌 step | run2 phase1 | 이번 실험 |
|---|---|---|
| ~1,870 | 3.860 (ep5) | **3.784** (ep10) |
| ~2,244 | 3.804 (ep6) | **3.734** (ep12) |
| ~2,992 | 3.717 (ep8) | **3.681** (ep16) |
| ~3,740 | 3.714 (ep10) | **3.637** (ep20) |
| ~4,488 | 3.680 (ep12) | **3.547** (ep24) |

세 런 모두 **첫 500스텝에서 −16~20%를 쏟고 1,500~5,000스텝 구간에서 평평**해지는 같은 모양.
flow loss가 평평한 것은 flow matching 특성(매 스텝 σ·노이즈를 새로 뽑아 샘플링 분산이 지배)이지
학습이 멈춘 것이 아님 — **train loss는 진행 신호로 쓸 수 없음.**