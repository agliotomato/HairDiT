# Cross-Identity Color-Sketch — 3-seed 집계 보고
---

## 1. 실험 개요

| 항목 | 내용 |
|------|------|
| 목적 | 외형 누설 진단 — 배경(face/img)을 **다른 인물**로 바꿔도 출력 헤어가 sketch에 충실한가 |
| 설계 | **Cross-TYPE**: braid sketch(A) → unbraid 이미지(B), unbraid sketch(A) → braid 이미지(B) |
| 입력 고정 | sketch = matte = A (해당 유형), face/img = B (반대 유형) |
| sketch 버전 | **original_sketch만** (원본 알록달록 스케치 그대로, 재착색 X — sketch에 A 정답색 없음) |
| 노이즈 시드 | 42, 1, 2 (3개) — 매핑은 고정, 노이즈만 변동 → mean±std |
| 대상 | braid 107 + unbraid 466 = 573장 |
| 모델 | mcs1~6 (각 epoch_40), 6구성 |
| 평가 | 헤어 영역(GT matte)만, 5지표 |

**보고 단위**:
- per-image 지표(Sketch LPIPS·Sketch ΔE px·Edge IoU·PSNR)는 **braid / unbraid 분리** + **macro-avg**.
- **macro-avg = (braid 평균 + unbraid 평균) / 2**.
- Hair FID는 2048-dim 공분산이라 소표본 rank-deficient → **통합(573)만** 보고.

**마스킹**:
- 전 지표 GT matte(`dataset/{split}/matte/test`, A 기준)로 헤어 영역 격리.
- ΔE·PSNR·FID = soft-alpha, Edge IoU = binary. 출력 stem=A_id 라 sketch/matte 전부 A 기준.

**색 지표(Sketch ΔE px)**
- CIEDE2000 + **L=50 음영정규화**(밝기 무시, 색조/채도만)
- 위치별 추종을 반영하는 px만 보고** (정합은 엄격).

---

## 2. 결과 (mean ± std, 3-seed)

**집계 정의 (2중 평균)**: 각 시드마다 split 이미지(braid 107 / unbraid 466)에 대해 지표 평균 → 시드 3개(42·1·2)를 다시 평균. **± = 3시드 간 표준편차**(노이즈 재현성 분산). **굵게 = split 내 지표별 best.**

### 2-1. Braid (n=107)

| 모델 | Sketch LPIPS↓ | Sketch ΔE px↓ | Edge IoU↑ | PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| **Ours** (mcs1) | 0.501±0.002 | **17.82±0.18** | 0.092±0.000 | **11.47±0.08** |
| **Ours+Gate** (mcs2) | 0.499±0.003 | 19.07±0.15 | 0.092±0.000 | 11.02±0.11 |
| **Sketch-only** (mcs3) | 0.478±0.002 | 17.89±0.20 | 0.089±0.000 | 11.01±0.09 |
| **Sketch-only+Gate** (mcs4) | **0.463±0.003** | 19.38±0.14 | 0.088±0.000 | 10.77±0.13 |
| **Raw-only** (mcs5) | 0.503±0.003 | 18.79±0.16 | 0.092±0.000 | 11.17±0.11 |
| **Matte-CNN-only** (mcs6) | 0.496±0.003 | 18.56±0.19 | **0.093±0.000** | 11.38±0.09 |

### 2-2. Unbraid (n=466)

| 모델 | Sketch LPIPS↓ | Sketch ΔE px↓ | Edge IoU↑ | PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| **Ours** (mcs1) | 0.701±0.022 | 16.63±0.07 | **0.052±0.001** | 10.96±0.11 |
| **Ours+Gate** (mcs2) | 0.725±0.023 | 17.07±0.03 | **0.052±0.001** | 10.73±0.13 |
| **Sketch-only** (mcs3) | 0.691±0.022 | **15.42±0.16** | 0.051±0.001 | 10.55±0.15 |
| **Sketch-only+Gate** (mcs4) | **0.672±0.029** | 17.19±0.13 | 0.049±0.002 | 10.58±0.15 |
| **Raw-only** (mcs5) | 0.712±0.019 | 17.14±0.03 | 0.051±0.001 | 10.68±0.14 |
| **Matte-CNN-only** (mcs6) | 0.694±0.025 | 16.88±0.05 | **0.052±0.001** | **10.81±0.12** |

### 2-3. Macro-avg = (braid + unbraid) / 2

| 모델 | Sketch LPIPS↓ | Sketch ΔE px↓ | Edge IoU↑ | PSNR↑ |
|------|:---:|:---:|:---:|:---:|
| **Ours** (mcs1) | 0.601±0.012 | 17.23±0.12 | **0.072±0.000** | **11.22±0.10** |
| **Ours+Gate** (mcs2) | 0.612±0.011 | 18.07±0.08 | **0.072±0.000** | 10.87±0.12 |
| **Sketch-only** (mcs3) | 0.584±0.011 | **16.66±0.18** | 0.070±0.000 | 10.78±0.12 |
| **Sketch-only+Gate** (mcs4) | **0.567±0.014** | 18.28±0.14 | 0.069±0.001 | 10.67±0.14 |
| **Raw-only** (mcs5) | 0.607±0.011 | 17.96±0.09 | 0.071±0.000 | 10.93±0.12 |
| **Matte-CNN-only** (mcs6) | 0.595±0.013 | 17.72±0.12 | **0.072±0.000** | 11.10±0.10 |

### 2-4. Hair FID (통합 573, unpaired)

| 모델 | Hair FID↓ |
|------|:---:|
| **Ours** (mcs1) | **140.11±3.10** |
| Ours+Gate (mcs2) | 152.28±1.52 |
| Sketch-only (mcs3) | 159.95±3.13 |
| Sketch-only+Gate (mcs4) | 161.76±1.61 |
| Raw-only (mcs5) | 148.73±2.49 |
| Matte-CNN-only (mcs6) | 148.10±1.85 |

- ± (3시드 std)가 전 지표·전 모델에서 작음 → 노이즈만 바꿔도 결과 일정 = 노이즈 통제 양호.
- braid가 unbraid보다 Sketch LPIPS·Edge IoU 우위 — braid 헤어가 구조 재현이 쉬운 편 ([0618] gt_sketch와 동일 경향).

---

## 3. 해석

### 3-1. HairFID, PSNR(화질)은 Ours가 우위
- **Hair FID: Ours 140.1로 최저**
    - 2등 Matte-CNN-only 148.1
    - Sketch-only는 **160.0으로 최악**.
- PSNR

    - macro·braid Ours 최고(11.22 / 11.47)  
    - unbraid는 Matte-CNN-only(10.81)가 우의

### 3-2. 색 추종(Sketch ΔE px)은 split별로 갈림
- **unbraid·macro는 Sketch-only(mcs3) 최저**(15.42 / 16.66) 
    - 알록달록 sketch 색을 가장 충실히 재현
    - sketch 정보만 받으니 색을 가장 직접적으로 표현(GT sketch와 같은 결과)

- 그러나 **braid는 Ours(mcs1)가 최저**(17.82). 
    - stroke-출력 픽셀 정합이 어려움 → Sketch-only가 색은 베껴도 위치가 어긋나 px ΔE가 나빠짐
    - matte-conditiong의 효과로 해석 가능


### 3-3. Gate 효과
- Ours+Gate·Sketch-only+Gate 모두 Gate 없는 버전 대비 **Hair FID 악화**(152↔140, 162↔160)·ΔE px 악화 → Gate가 색 추종/리얼리즘에 불리.



## 부록 — 산출물
- seed별: `eval_results/crossid/mcs{N}_original_sketch_seed{S}_summary.csv` (braid/unbraid/macro/combined573 컬럼; Sketch ΔE mean 값도 포함하나 표에선 px만 보고), `..._per_image.csv`
- 3seed 집계: `eval_results/crossid/summary_3seed.csv`
- 추론 출력: `outputs/crossid/{braid,unbraid}/mcs{N}_original_sketch/seed{42,1,2}/` (총 10,314장)
