# [0722] edge 타깃 정정 · flow val loss · 최종 학습 계획

## 0. 요약

1. **edge 타깃** — 코드 확인 결과 "**sketch stroke로 masked된 GT/구조 edge**"가 맞음. (§1)  
2. **flow val loss** — phase1·2 거치며 **상승**. 이는 **과적합의 직접 증거**이나 **품질 지표는 아님**. 품질 저하 = **원인=과적합(시변) + 조건=LPIPS·edge 무력화(상수)** 2단 구조. (§2)  
3. **최종 계획** — ①loss 밸런스 복원(정규화 **A:`.mean()` 되돌리기** / B:Eq.12 유지) + **loss ablation** → ②perceptual val 로깅 → ③과적합 관리(LR 인하) → ④replay(unbraid 훼손 억제). (§3)  

---

## 1. edge 타깃 정정

sketch는 **GT/구조 파생 조건 이미지** — 기하는 matte 실루엣, 색은 GT 헤어색. edge loss는 이 stroke를 **mask**로 써서 그 위치에 예측 edge presence를 강제.  
  
→ 타깃 = **sketch stroke로 masked된 GT/구조 edge.** OSDFace/EatGAN(GT edge)과 동일, [0721] 반영 완료.

---

## 2. flow val loss & 품질 저하 인과

학습 중 **세 가지가 서로 다르게 움직임**. 분리해서 보면 전부 설명 가능.  

| 무엇 | 추이 | 의미 |
|---|---|---|
| **① flow val loss** (held-out flow MSE) | 계속 **↑** (4.27→5.68) | 모델이 학습셋을 **외우기 시작**(과적합) |
| **② 실제 품질** (색·질감, 눈·지표) | p2e5까지 **↑** → 이후 **↓** | 우리가 원하는 것 |
| **③ LPIPS 효력** | 내내 **≈0** (flow의 1.8%) | 품질 방어 **브레이크 없음** |

**읽는 법:**
- **①은 품질이 아님.** flow val이 올라도 ep30 > ep10(눈으로 확인). flow MSE와 이미지 품질은 별개 → **정지 기준으로 쓰면 안 됨.**
- **①이 말해주는 건 "과적합 진행 중".** train은 내려가는데(3.71→2.44) val만 오름 = 과적합의 지문. (train이 안정 하강이므로 **발산 아님**.)
- **품질(②)을 무너뜨리는 원인이 바로 그 과적합(①).** 단 즉시가 아니라 임계점 이후 — p2e5까진 "배우는 것 > 외우는 것"이라 품질↑, 이후 역전되어 색저하·푸석.
- **왜 못 막았나 = ③.** LPIPS·edge가 무력화라 과적합이 품질을 갉아먹어도 막을 항이 없었음.  
  
**인과 정리 (2단):** **원인 = 과적합**(시변, 붕괴를 구동) / **조건 = LPIPS·edge 무력화**([0720] §3-2, 상수, 브레이크 없음). ③은 내내 상수라 "정점 후 붕괴"를 못 만듦 → 정점을 만든 건 과적합.  
  
**측정 결함(교정 필요)**: val seed 미고정(곡선 노이즈) · `eval_every=10`(초반 최저점 미측정) · 애초에 ①만 재고 ②를 안 잼.

---

## 3. 최종 학습 계획

|  | 조치 | 근거 |
|---|---|---|
| **0** | **loss 밸런스 복원 + loss ablation** (아래) | 논문 config에서 LPIPS·edge가 near-off(gradient 1.8%/0.009%) → **실효화 + 기여 입증 필요** |
| **1** | **perceptual val 로깅** — ΔE_GT·edge-IoU·LPIPS-GT, seed 고정·매 epoch | flow val이 못 보는 품질 곡선 확보 |
| **2** | **과적합 관리 = LR 인하** (phase2 2e-5→5e-6, phase1 유지) | 과적합(품질 저하의 **구동 원인**) 속도↓ |
| **3** | **replay/aug** (phase2에서 unbraid:braid = 1:1) | phase1(unbraid) **훼손(forgetting) 억제** |

**loss 밸런스 복원 — 두 갈래 (택1)**

| 갈래 | flow 정규화 | weight | 장단 |
|---|---|---|---|
| **A. `.mean()` 되돌리기** (권장) | 전체 원소 정규화(=mcs). 논문 Eq.12를 `.mean()` 형태로 개정 | **0.1 / 0.05 유지** (literature·mcs 실효) | hair-only gradient 유지, LPIPS 실효, weight 방어 불필요. **area-invariance 포기** |
| **B. Eq.12(area-norm) 유지** | 현행 유지(magnitude ~60×) | w_lpips·w_edge **재튜닝**(커짐, ~5급) | 수식 유지. 큰 값이라 **ablation으로만 정당화** |

> 두 갈래 모두 **hair 영역 제한(논문 핵심 주장)은 유지** — 바뀌는 건 정규화 상수·area-invariance뿐.  
  
**주의**: 순위 0 복원 후 **flow val이 더 올라도 정상** — LPIPS가 살아나 flow-MSE를 의도적으로 희생. 이 숫자를 낮추려 LPIPS를 다시 죽이면 안 됨. knob은 **한 번에 하나씩**, 판단은 **perceptual 기준**.
