# [0721] Loss 설계 근거 정리 및 커리큘럼 러닝 방식 선택

## loss 설계
> 목적: HairDiT의 현재 loss 구성을 명시하고, **어떤 논문을 근거로 이렇게 구성했는지**, 그 논문들은 loss를 **어떤 방식으로 사용했는지**, 그리고 **우리 loss와 무엇이 다른지**를 정리.  
> 주 근거 논문: **PixelGen** (flow 백본 + perceptual 지연 투입), **OSDFace** (Sobel 기반 edge 감독).

---

### 1. 현재 Loss 구성

전체 목적함수는 항상 **flow matching loss를 백본**으로 하고, 여기에 **LPIPS(perceptual)** 와 **edge** 항을 가중합으로 얹음.

```
L_total = w_flow · L_flow  +  w_lpips(t) · L_LPIPS  +  w_edge · L_edge
          (w_flow = 1.0, w_lpips = 0.1, w_edge = 0.05)
```

| 단계 | flow | LPIPS | LPIPS warmup | edge | 비고 |
|---|---|---|---|---|---|
| **Phase 1** (unbraid / pretrain) | **1.0** | 0.1 | **0.3** (전체 step의 30% 이후 활성) | 0.0 | 구조·색 기반 형성, perceptual은 지연 투입 |
| **Phase 2** (braid) | **1.0** | 0.1 | 0.0 (즉시 활성) | **0.05** | flow 백본 유지 + edge로 경계 선명도 강화 |

각 항의 정의:

- **`L_flow`** — flow matching velocity 예측 L2 loss. 생성의 주 목적함수.
- **`L_LPIPS`** — perceptual loss. perceptual 총칭 중 **LPIPS**(학습된 채널 가중치로 보정된 지각 거리)를 구체적으로 채택. Phase 1에서는 학습 진행도 30% 이후 활성(`lpips_warmup_frac=0.3`), Phase 2에서는 즉시 활성.
- **`L_edge`** — 예측 이미지의 Sobel gradient 강도를 구한 뒤, **matte 내부에서 sketch stroke가 있는데 예측 edge가 약한 영역**을 penalize → 예측 머리카락 edge를 **sketch stroke로 masked된 GT/구조 edge에 정렬**. (sketch stroke의 기하는 matte 실루엣, 색은 GT 헤어색에서 파생 → stroke 자체가 GT/구조 파생물임에 유의.) Phase 2(braid)에서만 w_edge=0.05.

---

### 2. 근거 논문과 사용 방식

#### 2.1 PixelGen — "flow 백본 + perceptual, 지연 투입"

**논문에서 loss를 쓴 방식:**
- 예측 이미지를 velocity로 변환해 **flow-matching 목적함수를 주 loss로 유지**하고, 그 위에 perceptual 항(LPIPS + DINO)을 결합.
- 핵심: perceptual을 **처음부터 켜지 않는다.** 고노이즈 **상위 30% timestep에서는 perceptual을 끄고**, 저노이즈 70% 구간에서만 활성. 고노이즈 구간에 perceptual을 걸면 sample diversity(recall)가 떨어지기 때문. → "perceptual은 예측이 clean 영역에 들어온 뒤 가장 유용함."
- **가중치 명시**: `λ_LPIPS = 0.1` (P-DINO 0.01). ablation에서 "0.05는 너무 약, 0.5~1.0은 recall 저하 → **0.1이 최적**"이라 근거까지 제시 → 우리 `w_lpips=0.1`과 동일.

**우리가 가져온 것:**
- flow(w_flow=1)를 주 loss로 두고 perceptual(LPIPS)을 보조로 얹는 **백본 구조**.
- perceptual을 **초기에 끄고 나중에 켜는 지연 전략**.

**차이:** PixelGen은 **diffusion timestep** 기준으로 게이팅하고, 우리는 **학습 스케줄(전체 step의 30%)** 기준 warmup으로 단순화함. 구현 축은 다르지만 "perceptual은 초기를 지나 예측이 안정된 뒤 켠다"는 철학은 동일.

출처: [PixelGen, arXiv:2602.02493](https://arxiv.org/abs/2602.02493)

#### 2.2 OSDFace — "Sobel 기반 edge 감독으로 경계 선명도 강화"

**논문의 사용 방식:**
- one-step diffusion face restoration. perceptual로 LPIPS 대신 **DISTS** 채택(LPIPS는 diffusion에서 artifact 유발), **edge-aware 버전** 도입:
  ```
  L_EA-DISTS(Î, I) = L_DISTS(Î, I)  +  L_DISTS( S(Î), S(I) )
  ```
  `S(·)` = **Sobel operator**. 원본 이미지에 더해 **Sobel edge map에도 perceptual 거리**.
- 동형 **EA-LPIPS**도 정의 (이미지 Sobel 처리 후 LPIPS).
- 동기: "edge detail의 정확한 생성 = perceived sharpness에 필수" — 특히 **머리카락·피부**의 texture/경계 보존.
- **가중치**: edge 성분(Sobel)은 EA-DISTS 내 non-edge DISTS와 **1:1 합산, 독립 계수 없음**. 바깥 `L_gen = λ_dis·L_GAN + λ_ID·L_ID + λ_per·L_EA-DISTS + MSE`의 λ 수치(Stage2)는 **미기재**.

**우리가 가져온 것:**
- **Sobel 기반 edge 감독으로 경계 선명도 강화** — OSDFace와 동일 목적.
- 머리카락 경계/가닥 디테일 강화를 위한 독립 edge 항.

**차이:** 두 방식 모두 **타깃은 GT 파생 edge**로 동계열이다(우리 sketch stroke = GT/구조 파생). 차이는 형태뿐: OSDFace는 edge 감독을 **perceptual에 흡수** — `L_DISTS(S(Î), S(I))`로 GT Sobel edge map **전체와 perceptual 매칭**, non-edge와 **1:1**, 독립 가중치 없음. 우리는 edge를 **독립 항(`w_edge=0.05`)** 으로 분리하고, GT edge map 전체 매칭이 아니라 **sketch stroke로 masked된 영역에서 edge-presence만 단방향으로** 강제. 즉 ① perceptual 흡수 vs pixel-space 독립 항, ② full GT edge 매칭 vs stroke-mask 단방향 presence 두 축에서 상이. (선명도 정합만 높이려면 OSDFace식 EA-LPIPS 흡수도 가능.)

출처: [OSDFace (CVPR 2025), arXiv:2411.17163](https://arxiv.org/abs/2411.17163)

#### 2.3 (보조 근거) flow + perceptual 결합의 일반성

지연 투입/edge를 뺀 "flow + LPIPS" 결합 자체는 이미 여러 연구에서 검증됨 — 우리 백본 선택의 일반성 근거.

- **SSDD** — 주 loss `L_FM = ‖x − ε − D(x_t|t,z)‖²` + `L_LPIPS = LPIPS(x, x̂_0)`. [arXiv:2510.04961](https://arxiv.org/pdf/2510.04961)
- **Improving the Training of Rectified Flows** — rectified flow에 **LPIPS-Huber**가 few-step 품질에 결정적. [arXiv:2405.20320](https://arxiv.org/html/2405.20320v1)
- **RealisVSR** — rectified flow(`L_REC`) + **gradient(HOG) 기반 texture loss** + wavelet loss. flow+gradient 결합 선례. [arXiv:2507.19138](https://arxiv.org/pdf/2507.19138)

---

### 3. 우리 Loss vs 근거 논문 — 차이 비교

| 항목 | 본 연구 (HairDiT) | PixelGen | OSDFace |
|---|---|---|---|
| **주 loss(백본)** | flow matching (w_flow=1.0) | flow matching | one-step diffusion + **GAN(adversarial)** |
| **perceptual 종류** | **LPIPS** | LPIPS + DINO | **DISTS** (LPIPS는 artifact 우려로 회피) |
| **perceptual 가중치** | w_lpips=**0.1** | **λ_LPIPS=0.1** (ablation 명시) | λ_per 미기재 |
| **perceptual 지연** | ✅ 학습 스케줄 30% warmup | ✅ **timestep(노이즈)** gating | ✗ (지연 없음) |
| **edge 감독** | Sobel edge presence ↔ **stroke-mask 단방향** | ✗ | Sobel edge map에 **perceptual 거리**(EA-DISTS) |
| **edge 타깃** | **GT/구조 파생 sketch stroke로 masked된 edge** | — | **GT 이미지**의 Sobel edge |
| **edge 가중치** | w_edge=**0.05** (자체 선정, 근거 논문 없음) | — | 독립 계수 없음(EA-DISTS 내 1:1) |
| **추가 항** | — | DINO perceptual | GAN + identity(ArcFace) loss |

**핵심 차이 3가지:**

1. **지연 투입의 축이 다름** — PixelGen은 noise timestep 기준, 우리는 training-schedule 기준 warmup. (효과는 유사, 구현은 단순화)
2. **perceptual 선택** — OSDFace는 LPIPS를 피하고 DISTS를 썼지만, 우리는 **LPIPS를 그대로 채택**. (flow 백본 + 지연 투입 조합에서는 PixelGen처럼 LPIPS가 유효하다고 판단)
3. **edge loss의 형태가 다름 (타깃은 동계열)** — 두 방식 모두 **GT 파생 edge를 타깃**으로 한다(우리 sketch stroke도 GT/구조 파생). 차이는 형태: OSDFace는 **GT edge map 전체를 perceptual로 매칭**, 우리는 **sketch stroke로 masked된 영역에서 edge-presence만 pixel-space·단방향**으로 강제. "정답 edge를 따라가라"는 목적은 공유하되, 매칭 대상(전체 map vs stroke mask)과 방식(perceptual vs pixel-space)이 다름.

**요약:** 우리 loss는 **PixelGen(flow 백본 + perceptual 지연)** 의 골격에 **OSDFace(Sobel 기반 edge 감독)** 의 아이디어를 결합하되, ① perceptual을 LPIPS로 유지하고 ② 지연을 training-schedule warmup으로 단순화했으며 ③ edge 감독을 **GT edge map 전체 perceptual 매칭 대신, GT/구조 파생 sketch stroke로 masked된 영역의 pixel-space edge-presence**로 구현한 변형임.

---

## 커리큘럼 러닝 방식 선택 (Rehearsal / Replay)

> 문제: unbraid(phase1, 3000장) → braid(phase2, 1000장) 2단계 학습에서 "phase2 epoch↑ → phase1 훼손(catastrophic forgetting), epoch↓ → braid 언더러닝" 딜레마.  
> 결론: **epoch가 아니라 (1) 이전 데이터를 계속 섞는 replay 비율, (2) 신규/이전 loss 가중치로 해결.** 아래는 우리가 그대로 쓸 수 있는 diffusion 선례 2편 + 불균형 2단계 선례 1편.

### 1. 근거 (diffusion 선례)

| 연구 | 링크 | replay 방식 | **핵심 수치 (우리가 채택할 값)** | LR |
|---|---|---|---|---|
| **Latent Replay** ([2509.10529](https://arxiv.org/pdf/2509.10529)) | **Diffusion (T2I)** | loss 레벨 `L=(1−λ)L_new+λL_mem` | **λ_mem=0.5 (신규:이전=1:1)**, 매 step 신규와 **동일 크기**의 이전 latent를 replay, 이전 데이터 전량이 아니라 소량 반복 replay로 충분 | b1×accum4 |
| **DreamBooth** ([2208.12242](https://arxiv.org/pdf/2208.12242)) | Diffusion | prior-preservation(=자기생성 replay) | **λ_pp=1.0 (1:1)**, ~1000 step | **5e-6** |
| **Two-phase imbalance** ([2112.14491](https://arxiv.org/pdf/2112.14491)) | 분류 | 단계 분리 | 균형 단계→원분포 미세조정 (재샘플링 단독보다 F1 +6.1%) | — |

**공통 결론 3줄**
1. diffusion 두 선례 모두 **신규:이전 = 1:1** (λ=0.5~1.0)이 기본값. 이전이 훨씬 많아도 전량이 아니라 신규와 동등 비율만 섞음.
2. **LR은 5e-6급으로 낮게** (DreamBooth).
3. replay는 단독이 아니라 **낮은 LR과 함께** 써야 forgetting을 실제로 막음.

### 2. HairDiT 적용 (epoch 증가 대신)

**현재 상태**: phase1 = unbraid 3000 단독 → phase2 = **braid 1000 단독**(`phase2_braid`). phase2에 unbraid가 **전혀 들어가지 않아** phase1에서 형성한 **unbraid 능력이 훼손(catastrophic forgetting)** 되는 것이 핵심 문제. 선례의 "이전 데이터 replay"로 교정:

1. **phase2에 unbraid replay 추가 → 매 배치 braid:unbraid = 1:1** — phase2를 braid 단독이 아니라 **unbraid 3000 + braid 1000**으로 구성하고, `WeightedRandomSampler`로 braid를 3× 자주 뽑아 매 배치 **50:50**으로 노출(물리 복제 X). unbraid를 계속 replay해 forgetting 방지. (Latent Replay λ=0.5 / DreamBooth λ=1.0 근거 — 단, sampler 방식과 loss-가중 방식 중 **sampler 하나만** 사용)
2. **epoch 40 고정 폐기 → dual-val early-stopping** — unbraid val이 나빠지기 시작하는 지점 = forgetting 시작 = 정지점. joint_phase2 회귀 진단의 p2e5 정점과 일치.
3. **LR 2e-5 → 5e-6 실험** — braid 소수 단계에서 DreamBooth 수준으로 낮추는 값 검토.

> **요약:** 핵심 문제는 "phase2에서 unbraid 훼손". 답은 **"phase2를 braid 단독이 아니라 unbraid+braid 50:50으로 replay, LR 5e-6급, unbraid val 기준 early-stopping."** diffusion 직접 선례(Latent Replay·DreamBooth)와 정합.
