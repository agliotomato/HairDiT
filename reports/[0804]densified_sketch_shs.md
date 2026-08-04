# [2026-08-04] Stroke Densification 추론 검증 결과 — SHS 공식 코드 반영판


## 0. 결론 먼저

> **"stroke 밀도를 올리면 방향의 seed 종속성이 줄어드는가?" → 예, 그리고 밀도에 거의 선형적으로 비례해서.**

run4(0730 체크포인트, phase1 epoch30)에 densified sketch만 입력해 재학습 없이 추론한 결과,
**seed 간 방향 불일치가 baseline 대비 최대 22~27% 감소**했고 GT 방향 오차도 함께 감소. 처음
자체 구현(K=25/15/11, `[0804]densified_sketch.md`) 3점만으로는 "약한 densification에서 이미
개선분 대부분을 얻고 포화"되는 것처럼 보였으나, PI 지적(§1)으로 SHS가 공개한 공식
auto-completion 코드로 교체하고 threshold를 6~27까지 8단계로 촘촘히 sweep한 결과
**baseline부터 밀도 0.14 부근까지 거의 매끈한 단조 감소 곡선**이 확인됨 — 이전의 "포화"
판정은 표본이 3점뿐이었던 데서 온 착시였음. 자체 구현(L1/L2/L3)과 SHS 공식 코드(threshold
sweep) 점들이 밀도축에서 서로 끼어들며 같은 곡선 위에 놓여, **구현 방식 차이가 결과에
미치는 영향은 무해함**도 함께 확인됨. 다만 mcs2 수준(seed 완전 강건)에는 아직 미달.

---

## 1. PI 피드백 반영 — SHS 공식 코드로 교체

**배경**: 자체 구현(`[0804]densified_sketch.md` §1, K=25/15/11)은 SHS 논문 서술만 보고
재구현한 것이라 SHS 공개 코드 `autocompletion/unbraid_completion.py`와 파라미터 의미가 다름 —
우리 `K`(dilation, L∞ 반경)와 SHS `threshold`(EDT, L2 거리)는 같은 숫자라도 강도가 다르고,
최초 예측은 "SHS 기본값(threshold=15)이 L1(K=25)보다 약할 것".

**교체 원칙(PI 지시)**: `getSketchCompletion()`을 그대로 쓰고 하드코딩된 `threshold=15` 한
줄만 함수 인자로 분리, 그 외 수정 금지. SHS 코드는 이진 마스크만 반환하므로 색 전파
(`_propagate_color`) + blend(원본 우선)만 추가.

**첫 sweep(15/12/9/6)**: SHS 기본값(threshold=15) 밀도(0.1224/0.1289)가 L1(0.1194/0.1270)과
거의 같음 — 예측과 반대. threshold=15 고정, `small_cc`만 바꿔 원인 진단(CM_1067):

| small_cc | 의미 | new_density |
|---|---|---|
| 240 | SHS 코드 그대로 | 0.1224 |
| 675 | L2(K=15)의 `3K²` 상당 | 0.1173 |
| 1875 | L1(K=25)의 `3K²` 상당 | 0.1067 |

→ 버그 아님. SHS `small_cc=240`이 우리 기준(K=25에서 1875)보다 7.8배 관대해 "거리 임계가
약하다"는 효과를 상쇄 — PI가 §2-2에서 예견한 상쇄 효과가 실측 확인됨. `small_cc`는 SHS
값(240) 유지.

**sweep 확장**: L1보다 약한 지점 확보를 위해 threshold를 18/21/24/27까지 확장(threshold↑ =
밀도↓). `small_cc`·`matte>230`·`skeletonize(method='lee')` 등은 그대로.

| threshold | CM_1067 밀도 | CM_1082 밀도 |
|---|---|---|
| 27 | 0.0804 | 0.0834 |
| 24 | 0.0888 | 0.0924 |
| 21 | 0.0976 | 0.1023 |
| 18 | 0.1099 | 0.1181 |
| **15(기본)** | 0.1224 | 0.1289 |
| 12 | 0.1400 | 0.1425 |
| 9 | 0.1547 | 0.1528 |
| 6 | 0.1621 | 0.1625 |

(baseline은 matte>230 기준 0.0679/0.0728 — 자체 구현의 matte>127 기준(0.0689/0.0737)과 약
1.5% 차이. §4 병합 표는 각 구현 실측값을 그대로 사용.)

시각화 검증(`[0804]densified_sketch.md` §1.2 체크리스트)도 전 threshold 지점에서 통과.

---

## 2. 입력 sketch — SHS 공식 코드 threshold sweep (밀도순)

기존 seed 실험과 동일 조건(체크포인트 run4 phase1 epoch30, 20-step)에서 sketch 입력만 교체.

| | T27(.080/.083) | T21(.098/.102) | T15기본(.122/.129) | T9(.155/.153) |
|---|---|---|---|---|
| CM_1067 | <img src="../data/densified_shs/T27/CM_1067.png" width="130"> | <img src="../data/densified_shs/T21/CM_1067.png" width="130"> | <img src="../data/densified_shs/T15_shs_default/CM_1067.png" width="130"> | <img src="../data/densified_shs/T9/CM_1067.png" width="130"> |
| CM_1082 | <img src="../data/densified_shs/T27/CM_1082.png" width="130"> | <img src="../data/densified_shs/T21/CM_1082.png" width="130"> | <img src="../data/densified_shs/T15_shs_default/CM_1082.png" width="130"> | <img src="../data/densified_shs/T9/CM_1082.png" width="130"> |

---

## 3. 생성 결과 — SHS 공식 코드 조건 × seed

baseline·mcs2 참조는 기존 렌더 재사용, SHS 8개 threshold(T27~T6)는 신규 추론. 밀도 오름차순.

### 3.1 CM_1067

seed42에서 여전히 좌측 하단 노이즈 발생, 우측 하단 노이즈는 완화

| run (밀도) | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| baseline (.068) | <img src="../outputs/0803/seed_run4/42/CM_1067.png" width="115"> | <img src="../outputs/0803/seed_run4/1/CM_1067.png" width="115"> | <img src="../outputs/0803/seed_run4/2/CM_1067.png" width="115"> | <img src="../outputs/0803/seed_run4/3/CM_1067.png" width="115"> |
| SHS_T27 (.080) | <img src="../outputs/0804/densified_shs/T27/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T27/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T27/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T27/3/CM_1067.png" width="115"> |
| SHS_T24 (.089) | <img src="../outputs/0804/densified_shs/T24/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T24/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T24/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T24/3/CM_1067.png" width="115"> |
| SHS_T21 (.098) | <img src="../outputs/0804/densified_shs/T21/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T21/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T21/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T21/3/CM_1067.png" width="115"> |
| SHS_T18 (.110) | <img src="../outputs/0804/densified_shs/T18/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T18/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T18/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T18/3/CM_1067.png" width="115"> |
| SHS_T15기본 (.122) | <img src="../outputs/0804/densified_shs/T15_shs_default/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T15_shs_default/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T15_shs_default/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T15_shs_default/3/CM_1067.png" width="115"> |
| SHS_T12 (.140) | <img src="../outputs/0804/densified_shs/T12/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T12/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T12/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T12/3/CM_1067.png" width="115"> |
| SHS_T9 (.155) | <img src="../outputs/0804/densified_shs/T9/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T9/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T9/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T9/3/CM_1067.png" width="115"> |
| SHS_T6 (.162) | <img src="../outputs/0804/densified_shs/T6/42/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T6/1/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T6/2/CM_1067.png" width="115"> | <img src="../outputs/0804/densified_shs/T6/3/CM_1067.png" width="115"> |
| (참조) mcs2 | <img src="../outputs/0803/seed_mcs2/42/CM_1067.png" width="115"> | <img src="../outputs/0803/seed_mcs2/1/CM_1067.png" width="115"> | <img src="../outputs/0803/seed_mcs2/2/CM_1067.png" width="115"> | <img src="../outputs/0803/seed_mcs2/3/CM_1067.png" width="115"> |

### 3.2 CM_1082

seed42, seed1 상단 머릿결 노이즈 완화

| run (밀도) | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| baseline (.073) | <img src="../outputs/0803/seed_run4/42/CM_1082.png" width="115"> | <img src="../outputs/0803/seed_run4/1/CM_1082.png" width="115"> | <img src="../outputs/0803/seed_run4/2/CM_1082.png" width="115"> | <img src="../outputs/0803/seed_run4/3/CM_1082.png" width="115"> |
| SHS_T27 (.083) | <img src="../outputs/0804/densified_shs/T27/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T27/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T27/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T27/3/CM_1082.png" width="115"> |
| SHS_T24 (.092) | <img src="../outputs/0804/densified_shs/T24/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T24/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T24/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T24/3/CM_1082.png" width="115"> |
| SHS_T21 (.102) | <img src="../outputs/0804/densified_shs/T21/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T21/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T21/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T21/3/CM_1082.png" width="115"> |
| SHS_T18 (.118) | <img src="../outputs/0804/densified_shs/T18/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T18/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T18/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T18/3/CM_1082.png" width="115"> |
| SHS_T15기본 (.129) | <img src="../outputs/0804/densified_shs/T15_shs_default/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T15_shs_default/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T15_shs_default/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T15_shs_default/3/CM_1082.png" width="115"> |
| SHS_T12 (.142) | <img src="../outputs/0804/densified_shs/T12/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T12/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T12/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T12/3/CM_1082.png" width="115"> |
| SHS_T9 (.153) | <img src="../outputs/0804/densified_shs/T9/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T9/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T9/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T9/3/CM_1082.png" width="115"> |
| SHS_T6 (.163) | <img src="../outputs/0804/densified_shs/T6/42/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T6/1/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T6/2/CM_1082.png" width="115"> | <img src="../outputs/0804/densified_shs/T6/3/CM_1082.png" width="115"> |
| (참조) mcs2 | <img src="../outputs/0803/seed_mcs2/42/CM_1082.png" width="115"> | <img src="../outputs/0803/seed_mcs2/1/CM_1082.png" width="115"> | <img src="../outputs/0803/seed_mcs2/2/CM_1082.png" width="115"> | <img src="../outputs/0803/seed_mcs2/3/CM_1082.png" width="115"> |

---

## 4. 판정 — 방향 지표 (자체 구현 + SHS 공식 코드 병합, 밀도순)

방향 지표는 기존에 캘리브레이션된 파라미터(`sigma_i=3`, `erode_px=6`, GT=`data/test/ori_image`)를 그대로 재사용.
아래 표는 자체 구현(K 기반)과 SHS 공식 코드(threshold 기반) 조건을 밀도 오름차순으로 병합한 것.

### 4.1 CM_1067

| run (밀도) | seed42 | seed1 | seed2 | seed3 | GT 오차 mean±std | coherence | **seed 불일치** |
|---|---|---|---|---|---|---|---|
| baseline (.068) | 16.56 | 16.51 | 17.17 | 16.16 | 16.60±0.42 | 0.751 | **14.41±0.41** |
| SHS_T27 (.080) | 15.68 | 16.04 | 16.87 | 15.93 | 16.13±0.51 | 0.761 | **13.27±0.46** |
| SHS_T24 (.089) | 15.49 | 15.80 | 16.79 | 15.73 | 15.95±0.57 | 0.771 | **12.85±0.61** |
| SHS_T21 (.098) | 15.08 | 15.82 | 16.50 | 15.52 | 15.73±0.60 | 0.775 | **12.51±0.55** |
| SHS_T18 (.110) | 14.72 | 15.67 | 16.31 | 15.30 | 15.50±0.67 | 0.784 | **11.75±0.50** |
| L1_mild (.119) | 15.04 | 15.53 | 16.15 | 15.77 | 15.62±0.46 | 0.787 | **11.36±0.53** |
| SHS_T15기본 (.122) | 14.60 | 15.68 | 15.97 | 15.40 | 15.41±0.59 | 0.795 | **11.17±0.49** |
| SHS_T12 (.140) | 14.75 | 15.69 | 15.89 | 15.22 | 15.39±0.51 | 0.804 | **10.55±0.40** |
| SHS_T9 (.155) | 14.94 | 15.66 | 16.09 | 15.03 | 15.43±0.54 | 0.805 | **10.55±0.37** |
| SHS_T6 (.162) | 15.32 | 15.47 | 16.07 | 15.14 | 15.50±0.40 | 0.801 | **11.01±0.29** |
| L2_mid (.164) | 15.27 | 15.39 | 16.13 | 15.05 | 15.46±0.47 | 0.808 | **10.62±0.34** |
| L3_strong (.172) | 15.30 | 15.27 | 16.19 | 14.92 | 15.42±0.54 | 0.809 | **10.75±0.37** |
| (참조) mcs2 | 15.93 | 15.64 | 16.13 | 15.24 | 15.73±0.38 | 0.748 | **10.12±0.17** |

### 4.2 CM_1082

| run (밀도) | seed42 | seed1 | seed2 | seed3 | GT 오차 mean±std | coherence | **seed 불일치** |
|---|---|---|---|---|---|---|---|
| baseline (.073) | 16.12 | 16.64 | 17.70 | 16.59 | 16.76±0.67 | 0.758 | **14.34±0.58** |
| SHS_T27 (.083) | 15.60 | 16.20 | 17.14 | 16.26 | 16.30±0.63 | 0.770 | **13.61±0.49** |
| SHS_T24 (.092) | 15.34 | 16.02 | 16.96 | 16.11 | 16.11±0.66 | 0.772 | **13.35±0.39** |
| SHS_T21 (.102) | 15.07 | 15.74 | 16.72 | 15.79 | 15.83±0.68 | 0.776 | **12.94±0.34** |
| SHS_T18 (.118) | 14.92 | 15.39 | 16.32 | 15.78 | 15.60±0.59 | 0.786 | **12.49±0.47** |
| L1_mild (.127) | 14.86 | 15.72 | 16.36 | 15.84 | 15.69±0.62 | 0.790 | **12.60±0.43** |
| SHS_T15기본 (.129) | 14.61 | 15.33 | 16.20 | 15.65 | 15.44±0.66 | 0.797 | **11.98±0.48** |
| SHS_T12 (.142) | 14.48 | 15.01 | 15.98 | 15.37 | 15.21±0.63 | 0.806 | **11.47±0.49** |
| SHS_T9 (.153) | 14.46 | 14.98 | 16.04 | 15.40 | 15.22±0.66 | 0.810 | **11.16±0.45** |
| L2_mid (.161) | 14.78 | 14.89 | 15.94 | 15.60 | 15.30±0.56 | 0.807 | **11.39±0.39** |
| SHS_T6 (.163) | 14.53 | 15.00 | 16.21 | 15.44 | 15.30±0.72 | 0.805 | **11.17±0.47** |
| L3_strong (.172) | 14.83 | 14.94 | 16.23 | 15.83 | 15.46±0.68 | 0.806 | **11.31±0.43** |
| (참조) mcs2 | 14.51 | 14.88 | 15.20 | 15.21 | 14.95±0.33 | 0.796 | **9.72±0.43** |

### 4.3 독립 분석

가이드 §5 판정표: "std 감소 + mean 감소 → 진단 확증", "K(밀도)에 따른 단조 추세 → 가장 강한 증거".

1. **GT 오차 mean**: 두 이미지 모두 densification 조건 전부가 baseline보다 낮음. **mean 증가
   (OOD 신호) 없음.**
2. **GT 오차 std**(표의 "±" 값): 조건 간 큰 추세 없이 0.4~0.72 사이에서 흔들림 — 이전에 이미
   확인한 대로 이 std는 감도가 낮은 지표라 아래 3번을 판정에 사용.
3. **seed 불일치**: 위 §4.1·4.2 표에서 baseline → SHS_T27 → T24 → T21 → T18 → L1 → SHS_T15
   → SHS_T12 구간이 **두 이미지 모두 거의 완벽하게 단조 감소**(잡음 ±0.1-0.3 수준). 밀도
   0.14 부근(SHS_T12) 이후부터 SHS_T9 → SHS_T6/L2_mid → L3_strong 구간은 10.5~11.3 사이에서
   완만하게 등락(두 이미지에서 T6·L2의 상대 순서가 뒤바뀔 만큼 밀도차가 거의 없음) — 진짜
   포화 지점은 여기부터.

**종합**: 처음엔 L1/L2/L3 3점만 봐서 "약한 densification만으로 개선분 대부분을 얻고 포화"되는
것처럼 보였으나, §1에서 SHS 공식 코드로 8단계 sweep을 추가해 병합한 결과 baseline~밀도 0.14
구간에서 거의 완벽한 단조 dose-response가 나타남 — **이전의 "포화" 판정은 표본 부족 때문이었고,
밀도가 늘수록 seed 불일치가 계속 줄어드는 것이 맞음.** 또한 자체 구현(K계열)과 SHS 공식 코드
(threshold계열) 점들이 밀도축에서 서로 끼어들며 같은 곡선 위에 놓여, **구현 방식 차이가
결과를 오염시키지 않았음**도 함께 확인됨.

---

## 5. 이미지별 온도차

전체 sweep(자체 구현 3점 + SHS 공식 8점) 기준 최고 개선치: CM_1067은 baseline 14.41 → 최저
10.55(SHS_T12/T9 동률), **26.8% 감소**. CM_1082는 baseline 14.34 → 최저 11.16(SHS_T9),
**22.2% 감소**. CM_1067이 여전히 더 크게 개선되지만, 표본을 3점에서 11점(자체+SHS)으로
촘촘히 하자 CM_1082도 이전 최선(L3, 11.31)보다 더 낮은 지점(11.16)을 찾아내 격차가 다소
좁혀짐. mcs2 참조와의 잔여 격차는 CM_1067 0.43(10.55→10.12)로 거의 붙은 반면, CM_1082는
1.44(11.16→9.72)로 여전히 남아 있음 — 표본이 이미지 2장뿐이라 이 잔여 격차가 이미지 고유
특성(머리 길이·웨이브 정도) 때문인지 우연인지는 지금 데이터로 단정 불가.

---
