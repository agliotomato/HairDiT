# [0721] Loss 설계 근거 정리 및 커리큘럼 러닝 방식 선택

## loss 설계
> 목적: HairDiT의 현재 loss 구성을 명시하고, **어떤 논문을 근거로 이렇게 구성했는지**, 그 논문들은 loss를 **어떤 방식으로 사용했는지**, 그리고 **우리 loss와 무엇이 다른지**를 정리.
> 주 근거 논문: **PixelGen** (flow 백본 + perceptual 지연 투입), **OSDFace** (Sobel 기반 edge 감독).
> 관련 파일: `configs/joint_phase1.yaml`, `configs/joint_phase2.yaml`, `configs/phase2_braid.yaml`, `src/training/losses.py`

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
- **`L_edge`** — `SketchEdgeAlignmentLoss` (`src/training/losses.py`). 예측 이미지의 Sobel gradient 강도를 구한 뒤, **matte 내부에서 sketch stroke가 있는데 예측 edge가 약한 영역**을 penalize → 예측 머리카락 edge를 **입력 sketch stroke에 정렬**. Phase 2(braid)에서만 w_edge=0.05.

---

### 2. 근거 논문과 사용 방식 (why & how)

#### 2.1 PixelGen — "flow 백본 + perceptual, 단 지연 투입"

**그 논문에서 loss를 쓴 방식(how):**
- 예측 이미지를 velocity로 변환해 **flow-matching 목적함수를 주 loss로 유지**하고, 그 위에 perceptual 항(LPIPS + DINO)을 결합.
- 핵심: perceptual을 **처음부터 켜지 않는다.** 고노이즈 **상위 30% timestep에서는 perceptual을 끄고**, 저노이즈 70% 구간에서만 활성. 고노이즈 구간에 perceptual을 걸면 sample diversity(recall)가 떨어지기 때문. → "perceptual은 예측이 clean 영역에 들어온 뒤 가장 유용함."
- **가중치 명시**: `λ_LPIPS = 0.1` (P-DINO 0.01). ablation에서 "0.05는 너무 약, 0.5~1.0은 recall 저하 → **0.1이 최적**"이라 근거까지 제시 → 우리 `w_lpips=0.1`과 동일.

**우리가 가져온 것:**
- flow(w_flow=1)를 주 loss로 두고 perceptual(LPIPS)을 보조로 얹는 **백본 구조**.
- perceptual을 **초기에 끄고 나중에 켜는 지연 전략**.

**축의 차이(정직하게 명시):** PixelGen은 **diffusion timestep(노이즈 레벨)** 기준으로 게이팅하고, 우리는 **학습 스케줄(전체 step의 30%)** 기준 warmup으로 단순화함. 구현 축은 다르지만 "perceptual은 초기를 지나 예측이 안정된 뒤 켠다"는 철학은 동일.

출처: [PixelGen, arXiv:2602.02493](https://arxiv.org/abs/2602.02493)

#### 2.2 OSDFace — "Sobel 기반 edge 감독으로 경계 선명도 강화"

**해당 논문에서 loss를 쓴 방식(how):**
- one-step diffusion face restoration. perceptual loss로 LPIPS 대신 **DISTS**를 쓰되(LPIPS가 diffusion에서 artifact를 유발할 수 있어), **edge-aware 버전**을 도입:
  ```
  L_EA-DISTS(Î, I) = L_DISTS(Î, I)  +  L_DISTS( S(Î), S(I) )
  ```
  여기서 `S(·)`는 **Sobel operator**. 즉 원본 이미지에 더해 **Sobel로 뽑은 edge map에도 perceptual 거리를 건다.**
- 같은 방식의 **EA-LPIPS**도 정의: 이미지를 Sobel로 처리한 뒤 LPIPS 계산.
- 동기: "edge detail의 정확한 생성이 face restoration의 perceived sharpness에 필수" — 특히 **머리카락·피부 같은 영역**에서 texture/경계 디테일을 보존.
- **가중치**: edge 성분(Sobel)은 EA-DISTS 안에서 non-edge DISTS와 **1:1로 더할 뿐 독립 계수가 없음**. 바깥 `L_gen = λ_dis·L_GAN + λ_ID·L_ID + λ_per·L_EA-DISTS + MSE`의 λ 수치(Stage2)는 논문 **미기재**.

**우리가 가져온 것:**
- **Sobel 기반 edge 감독으로 경계 선명도(perceived sharpness)를 강화**한다는 아이디어와 동기 — OSDFace와 목적이 동일.
- 머리카락 경계/가닥의 디테일을 또렷하게 만들기 위해 edge 항을 별도로 두는 설계.

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
| **edge 감독** | Sobel edge 강도 ↔ **입력 sketch stroke 정렬** | ✗ | Sobel edge map에 **perceptual 거리**(EA-DISTS) |
| **edge 타깃** | **입력 sketch stroke** (matte 내부) | — | **GT 이미지**의 Sobel edge |
| **edge 가중치** | w_edge=**0.05** (자체 선정, 근거 논문 없음) | — | 독립 계수 없음(EA-DISTS 내 1:1) |
| **추가 항** | — | DINO perceptual | GAN + identity(ArcFace) loss |

**핵심 차이 3가지:**

1. **지연 투입의 축이 다름** — PixelGen은 noise timestep 기준, 우리는 training-schedule 기준 warmup. (효과는 유사, 구현은 단순화)
2. **perceptual 선택** — OSDFace는 LPIPS를 피하고 DISTS를 썼지만, 우리는 **LPIPS를 그대로 채택**. (flow 백본 + 지연 투입 조합에서는 PixelGen처럼 LPIPS가 유효하다고 판단)
3. **edge loss의 타깃이 근본적으로 다름** — OSDFace의 edge 항은 **GT 이미지 edge와의 perceptual 일치**(복원 태스크)인 반면, 우리 edge 항은 **입력 sketch stroke와의 정렬**(sketch-conditioned 생성 태스크). 즉 OSDFace는 "정답 edge를 따라가라", 우리는 "사용자가 그린 stroke를 따라가라". Sobel을 쓴다는 수단만 공유하고 목적함수의 의미는 다름.

**요약:** 우리 loss는 **PixelGen(flow 백본 + perceptual 지연)** 의 골격에 **OSDFace(Sobel 기반 edge 감독)** 의 아이디어를 결합하되, ① perceptual을 LPIPS로 유지하고 ② 지연을 training-schedule warmup으로 단순화했으며 ③ edge 타깃을 GT가 아닌 **입력 sketch stroke**로 바꾼 **sketch-conditioned 변형**임.

---

## 커리큘럼 러닝 방식 선택 (Rehearsal / Replay)

> 문제: unbraid(phase1, 3000장) → braid(phase2, 1000장) 2단계 학습에서 "phase2 epoch↑ → phase1 훼손(catastrophic forgetting), epoch↓ → braid 언더러닝" 딜레마.
> 결론: **epoch가 아니라 (1) 이전 데이터를 계속 섞는 replay 비율, (2) 신규/이전 loss 가중치로 해결.** 아래는 우리가 그대로 쓸 수 있는 diffusion 선례 2편 + 불균형 2단계 선례 1편.

### 1. 근거 (diffusion 직접 선례)

| 연구 | 매체 | replay 방식 | **핵심 수치 (우리가 채택할 값)** | LR |
|---|---|---|---|---|
| **Latent Replay** ([2509.10529](https://arxiv.org/pdf/2509.10529)) | **Diffusion (T2I)** | loss 레벨 `L=(1−λ)L_new+λL_mem` | **λ_mem=0.5 (신규:이전=1:1)**, 매 step 신규와 **동일 크기**의 이전 latent를 replay, buffer 100장이면 충분 | b1×accum4 |
| **DreamBooth** ([2208.12242](https://arxiv.org/pdf/2208.12242)) | Diffusion | prior-preservation(=자기생성 replay) | **λ_pp=1.0 (1:1)**, ~1000 step | **5e-6** |
| **Two-phase imbalance** ([2112.14491](https://arxiv.org/pdf/2112.14491)) | 분류 | 단계 분리 | 균형 단계→원분포 미세조정 (재샘플링 단독보다 F1 +6.1%) | — |

**공통 결론 3줄**
1. diffusion 두 선례 모두 **신규:이전 = 1:1** (λ=0.5~1.0)이 기본값. 이전이 훨씬 많아도 전량이 아니라 신규와 동등 비율만 섞음.
2. **LR은 5e-6급으로 낮게** (DreamBooth).
3. replay는 단독이 아니라 **낮은 LR과 함께** 써야 forgetting을 실제로 막음.

### 2. HairDiT 적용 (epoch 증가 대신)

현재 `joint_phase2 (dataset: both)`는 unbraid:braid = 3000:1000 = **3:1** → braid가 매 epoch **25%만** 등장(언더러닝). 선례의 1:1 기준으로 교정:

1. **braid를 sampler로 1:1까지 상향** — `WeightedRandomSampler`로 braid 3× 업웨이트(물리 복제 X). unbraid replay는 유지하되 braid를 매 step 동등 노출. (Latent Replay λ=0.5 / DreamBooth λ=1.0 근거)
2. **epoch 40 고정 폐기 → dual-val early-stopping** — unbraid val이 나빠지기 시작하는 지점 = forgetting 시작 = 정지점. joint_phase2 회귀 진단의 p2e5 정점과 일치.
3. **LR 2e-5 → 5e-6 실험** — braid 소수 단계에서 DreamBooth 수준으로 낮추는 값 검토.

> **한 줄:** "phase2 epoch 몇?" → **"braid replay를 1:1로, LR 5e-6급, unbraid val 기준 early-stopping."** diffusion 직접 선례(Latent Replay·DreamBooth)와 정합.

