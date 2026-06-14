# Sanity-Test 실험 요청 — Leakage(A vs B) + Harmony Tracking(리라이팅) 설계 (WACV 2027)

**논문**: SketchHair-DiT (WACV 2027, Round 1 enroll 6/19 · submission 6/26 AoE)
**성격**: 추론-only (재학습 불필요), controlled sanity-test

---

## 약어·개념 정리

| 약어 | 의미 |
|------|------|
| SoT | Single Source of Truth (위 권위 문서) |
| Ours / Ours+Gate / Sketch-only / Raw-only / Matte-CNN-only | 공식 명명 = 내부 mcs1 / mcs2 / mcs3 / mcs5 / mcs6 (논문 본문엔 mcs 코드 금지) |
| HairMapper | 헤어 제거(bald화) 도구 |
| GT 스케치 | 원본 헤어에서 추출한 스케치 |
| M / matte | hair matte (타겟 영역) |
| BLD | per-step 배경 latent 주입 — **타겟 이미지의 유일한 역할 = z_bg(BLD 소스)** |
| region IoU | 출력 헤어 segmentation ∩ 타겟 matte (영역 채움 완성도) |
| chroma | 채도 (낮으면 muddy) |
| WB / LUT | White Balance / 색 변환 테이블 |
| T (톤 변환) | 리라이팅에 적용한 전역 변환 — **우리가 정확히 아는 값** |
| tracking ratio | 출력 헤어의 hue·luminance 이동량 ÷ 적용한 T (0=장면 무시, 1=완전 추종) |
| OOD | Out-Of-Distribution (학습 분포 밖) |

---

## 공통 셋업

- **이미지 선정**: 1차는 **unbraid 1장 + braid 1장**으로 전 시나리오 파이프라인 가동(T3·T1 우선) → 방향 확인 + figure 후보 확보. 2차로 **논문에 수치로 올릴 시나리오만 각 2~3장으로 보강**(특히 T3 drop, T6 tracking — 1장으론 케이스 특수성에 좌우되어 일관성 주장 불가, 평균±범위로 보고).
  - 선정 기준: ① 거부감 없는 portrait + **깨끗한 matte/HairMapper 결과**(B 조건 품질이 관건) ② unbraid/braid는 배경·조명이 서로 다른 이미지 ③ T6용 스케치 stroke 색은 무채색이 아닌 **뚜렷한 색**(hue tracking 측정용) ④ 기존 cross-identity·money figure 이미지와 재사용 가능하면 우선.
- seed / num_steps / per-step BLD **전부 고정** — 시나리오당 변수 1개만.
- 표본이 적으므로 **FID 사용 금지**(소표본 왜곡) → region IoU·Edge IoU·chroma·정성 위주. 필요 시 KID(소표본 불편 추정).
- 모든 비교에서 스케치·matte·인물 동일, **구성(config)만 변수**.
- 보고 시 각 시나리오의 입력(스케치/matte/타겟 이미지)도 함께 첨부 (figure 재활용 목적).

---

## 타겟 이미지 정의 (🔴 최종 — 두 축 분리)

### Leakage 축: A vs B (단일 변수)

| 조건 | 내용 | 만드는 법 |
|------|------|----------|
| **A** | GT 헤어 + 원본 장면 (원본 그대로) | — |
| **B** | **bald** + 원본 장면 (장면·인물·조명 유지, 헤어만 제거) | HairMapper |

- **A−B drop = 원본-헤어 외형 누설 의존도.** GT 스케치를 전 조건 고정하므로 구조 정보는 모든 구성에 동일 공급 → A−B는 순수하게 "배경 GT 헤어의 외형 참조" 의존만 측정.

### Harmony 축: C″ 전체적인 색톤 조절 (배경 교체 아님)

- **만드는 법**: 이미지 **전체(인물+배경)** 에 **알려진 전역 톤 변환 T** 적용 — WB/색온도 시프트 또는 LUT/gamma. **warm / cool / dim 3종** (+원본 = 4점). bright는 하이라이트 클리핑(포화)으로 hue 측정이 오염되어 **제외**.

#### 구체 변환 config (figure에서 눈에 띄는 수준, PI 확정)

| 변형 | sRGB 채널 gain (R, G, B) | 등가 느낌 | 목표 시프트 |
|------|--------------------------|----------|------------|
| **Warm** | ×1.18, ×1.03, ×0.82 | 6500K → 약 4300K (백열등/노을) | Lab Δb ≈ +12~15 |
| **Cool** | ×0.84, ×0.98, ×1.18 | 6500K → 약 9500K (그늘) | Lab Δb ≈ −12~15 |
| **Dim** | 전 채널 ×0.55 (≈ −1.2 stop) (선택: B만 ×1.05 추가 → 저녁 실내) | 어두운 실내 | ΔL ≈ −20~25 |

- warm/cool은 **G≈1 고정, R/B 반대 방향** → 밝기 보존·hue만 이동 (hue 축). dim은 hue 보존·L만 이동 (luminance 축). **두 축 분리 측정.**

#### 구현 주의 3가지
1. **linear 공간에서 적용**: sRGB → linear → gain → sRGB. (sRGB 직접 곱은 hue 비선형 왜곡)
2. **클리핑 방지**: warm의 R gain이 밝은 피부 하이라이트에서 포화 가능 → gain 전 전체 ×0.95 또는 soft-clip. 포화 픽셀은 hue가 무너져 측정 오염.
3. **🔴 tracking ratio 분모 T = 실측**: 변환을 **원본 이미지 헤어 영역에 적용했을 때 실제 생기는 Δhue/ΔL**을 재서 분모로 사용 (이론 gain 아님 — 클리핑·감마 효과 자동 반영).

#### Figure 구성 팁
- 1행 = 입력 타겟 [원본 | warm | cool | dim] 자체를 먼저 노출, 2~3행 = Ours / Ours+Gate 출력 → "장면이 이만큼 변했고, Ours는 따라가고 Gate는 고정"이 한눈에.

---

## 시나리오 (우선순위 순)

### T3. Leakage 행동 확인 — 🔴 최우선
- **입력**: 같은 GT 스케치 + GT matte를 타겟 **A vs B** 에 적용.
- **비교**: Sketch-only vs Ours.
- **측정**: A→B 에서 metric(Edge IoU, region IoU, 정성) **drop 폭**.
- **예측**: Sketch-only의 drop ≫ Ours → "GT-matched 평가가 sketch-only를 과대평가"의 단일-변수 직접 증거.

### T1. Region fill (raw-matte 역할) — 2순위
- **입력**: 타겟 **B**(bald+원본장면) + matte + **sparse 스케치**.
- **비교**: Ours / Matte-CNN-only / Raw-only / **Sketch-only(floor)** 4구성.
- **측정**: **region IoU**.
- **예측**: matte 있는 3구성 ≫ Sketch-only. (bald라 배경 도움 0 → 순수 fill 능력)

### T6. Harmony tracking (gate trade-off 정량) — 3순위 🔴 신설
- **입력**: 타겟 **C″ 리라이팅 세트**(warm/cool/dim 3종) + 같은 스케치·matte.
- **🔴 스케치 색 = 원본 그대로 전 변형 고정** (스케치는 한 번 만들어 4개 변형에 재사용. 스케치까지 T로 변환하면 지시-추종과 harmony가 confound되어 측정 불가).
- **비교**: Ours vs Ours+Gate.
- **측정**: **tracking ratio** — 출력 헤어(matte 내부)의 hue·luminance 이동량을 적용한 T로 나눈 값.
- **예측**: **Ours ≈ 1에 가깝게 추종**(scene harmonize) / **Ours+Gate 낮음**(배경 독립) → gate trade-off의 단일 정량 인덱스. 같은 세트가 figure(배경 톤별 나란히 비교)로 재활용됨.
- **해석 프레임**: 스케치 색 = 고유색(albedo) 지시 / 장면 톤 = 조명. 같은 고유색도 조명이 바뀌면 보이는 색이 ≈T만큼 이동하는 게 물리적으로 자연 → tracking ↑ 는 지시 위반이 아니라 **조명 적응(자연스러움)**, tracking ↓ 는 색 고정(배경-독립 옵션).

### T4. Shading 보존 (T6 보조)
- **입력**: 타겟 **A** + recolor 스케치. **비교**: Ours vs Ours+Gate.
- **측정**: **luminance 상관** — 원본 헤어 음영 맵 vs 출력 음영 맵 (matte 내부 상관계수).
- **예측**: Ours 높음(음영 보존·자연) / Gate 낮음.

### T2. Sparsity ladder (MatteCNN 역할)
- **입력**: 타겟 **B**, 같은 matte, 스트로크 수 3단계(촘촘→매우 sparse).
- **비교**: Ours vs Raw-only.
- **측정**: Edge IoU + **hair 영역 chroma 평균**(낮으면 muddy) + 동일 영역 zoom 정성.
- **예측**: sparse할수록 CNN 이득 확대; Raw-only는 색 탁하고 결 뭉개짐.

### T5. Stroke 배치 대응 (gate 백업, 여유 시)
- **입력**: 타겟 A 또는 B, 뚜렷한 다색 sparse 스케치. **비교**: Ours vs Ours+Gate.
- **측정**: **stroke-overlay**(스케치 반투명 오버레이) 정성 + 스트로크 주변 dominant hue 일치율 (**내부 확인용** — 논문 표에 넣지 않음).

---

## 주의사항

1. **B의 HairMapper inpaint는 불가피하며 대부분 무관** — M 내부는 어차피 생성으로 덮임. **유일한 관리 대상 = matte 경계 밖 밴드의 잔머리 잔존**(원본-헤어 누설을 도로 들여와 B의 목적 훼손 — 치명). 해법: ① 잔머리 없는 plate 선별, ② **matte를 소폭 dilate**해 잔머리를 M 안으로 흡수(과도 금지). 스킨 블러·톤 부자연은 경미.
2. **GT 스케치 고정의 한계**: 구조 누설(스케치=GT 파생)은 잔존 — 이 테스트는 **외형 의존만 분리**함을 보고서에 한 줄 명시.
3. bald base(T1·T2·T3-B)는 train/inference OOD → **절대 품질 해석 금지, 구성 간 상대 비교만** ("controlled diagnostic").
4. **T3·T1만 먼저 나와도 즉시 보고** 부탁 (WACV R1 일정상 우선).

---

## 논문 연결 (참고)

논문 핵심 thesis — **"complex strand(헤어)에서는 hair matte conditioning control이 중요하다"** — 의 증거 축 ⑤(controlled sanity-test):
- T3 → 평가 분석(leakage account)의 단일-변수 확정
- T1 → raw-matte 영역 fill 정량 (region IoU)
- T6 → gate trade-off(harmony↔독립)의 단일 수치(tracking ratio) + F1 figure 정량판
- T4 → shading 보존(자연스러움) 보조
- T2 → MatteCNN strand·색 순도 효과