# [0722] edge 타깃 정정 · flow loss 오버 경향 · 최종 학습 계획

## 0. 요약

1. **edge 타깃** — 코드 확인 결과 "**sketch stroke로 masked된 GT/구조 edge**"가 맞음. (§1)  
2. **flow loss 오버 경향** — flow가 ~60× 팽창해 LPIPS·edge를 **55×/11,068× 압도**(무력화) → **확인됨**. 품질 저하 = 원인=과적합 / 조건=이 무력화. (§2)  
3. **최종 계획** — ①loss 밸런스 복원(정규화 **A:`.mean()` 되돌리기** / B:Eq.12 유지) + **loss ablation** → ②perceptual val 로깅 → ③과적합 관리(LR 인하) → ④replay(unbraid 훼손 억제). (§3)  

---

## 1. edge 타깃 정정

sketch는 **GT/구조 파생 조건 이미지** — 기하는 matte 실루엣, 색은 GT 헤어색. edge loss는 이 stroke를 **mask**로 써서 그 위치에 예측 edge presence를 강제.  
  
→ 타깃 = **sketch stroke로 masked된 GT/구조 edge.** OSDFace/EatGAN(GT edge)과 동일, [0721] 반영 완료.

---

## 2. flow loss 오버 경향 확인 & 품질 저하 인과 (피드백 ②)

**질문 = flow loss가 과도하게 커져 LPIPS·edge가 억눌리는 경향** → **확인됨.**

flow 정규화 변경(‖matte‖₁로 나눔)으로 flow가 ~60× 팽창 → gradient 기준으로 flow가 나머지를 압도 ([0720] §3-2):

| 항 | 가중 gradient | flow 대비 |
|---|---|---|
| flow | 1.45e-1 | — |
| LPIPS | 2.63e-3 | **1.8%** (flow가 55× 압도) |
| edge | 1.31e-5 | **0.009%** (flow가 11,068× 압도) |

→ **flow가 LPIPS·edge를 각각 55×·11,068× 압도 = 사실상 무력화** — 색·질감 방어 항이 사라진 상태. → §3 순위 0(밸런스 복원)으로 교정.

**품질 저하 인과 (2단):** **원인 = 과적합**(시변, 학습이 길어질수록 붕괴 구동) / **조건 = 위 무력화**(상수, 방어 브레이크 없음). p2e5까진 품질↑, 이후 과적합이 우세해지며 색저하·푸석. 무력화는 상수라 정점을 만든 건 과적합.

**(별도 관찰) flow VAL loss도 상승** — held-out flow loss가 epoch마다 오르지만(4.27→5.68) 이건 **과적합 신호일 뿐 품질 지표는 아님**(ep30 > ep10). 정지 기준으론 부적합 → perceptual val 필요(§3 순위 1).

**측정 결함(교정 필요)**: val seed 미고정 · `eval_every=10`(초반 최저점 미측정).

---

## 3. 최종 학습 계획

|  | 조치 | 근거 |
|---|---|---|
| **0** | **loss 밸런스 복원 + loss ablation** (아래) | 논문 config에서 LPIPS·edge가 near-off(gradient 1.8%/0.009%) → **실효화 + 기여 입증 필요** |
| **1** | **perceptual val 로깅** — ΔE_GT·edge-IoU·LPIPS-GT, seed 고정·매 epoch마다 | flow val이 못 보는 품질 곡선 확보 |
| **2** | **과적합 관리 = LR 인하** (phase2 2e-5→5e-6, phase1 유지) | 과적합(품질 저하의 **구동 원인**) 속도↓ |
| **3** | **replay/aug** (phase2에서 unbraid:braid = 1:1) | phase1(unbraid) **훼손(forgetting) 억제** |

**loss 밸런스 복원 — 두 갈래 (택1)**

| 갈래 | flow 정규화 | weight | 장단 |
|---|---|---|---|
| **A. `.mean()` 되돌리기** | 전체 원소 정규화(=mcs). 논문 Eq.12를 `.mean()` 형태로 개정 | **0.1 / 0.05 유지** (literature·mcs 실효) | hair-only gradient 유지, LPIPS 실효, weight 방어 불필요. **area-invariance 포기** |
| **B. Eq.12(area-norm) 유지** | 헤어 면적(‖matte‖₁)으로 정규화 (magnitude ~60×) | w_lpips·w_edge **재튜닝**(커짐, ~5급) | 논문 수식 유지. **ablation으로 정당화** |

> 두 갈래 모두 **hair 영역 제한(논문 핵심 주장)은 유지** — 바뀌는 건 정규화 상수·area-invariance뿐.  
  
**주의**: loss 밸런스 복원 후 **flow val이 더 올라도 정상** — LPIPS가 살아나 flow-MSE를 의도적으로 희생. 이 숫자를 낮추려 LPIPS를 다시 죽이면 안 됨. knob은 **한 번에 하나씩**, 판단은 **perceptual 기준**.
