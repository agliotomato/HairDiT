# [2026-07-28] 머릿결 방향 노이지(질감 방향성) 원인 분석

> `analysis.md`가 던진 질문(run1은 안 노이지, run2는 phase2부터 노이지, run3는 phase1부터 노이지)에 대해, 코드·학습 로그·결과 이미지를 직접 대조해 독립적으로 원인을 도출함. `[0726]`/`[0727]` 리포트의 원인분석·해결방안은 참고하지 않음.

## 요약

1. 문제를 색(§2, 이 리포트에서 다루지 않음)과 질감(본 주제)으로 분리.
2. 질감 노이지 후보는 두 가지(§3): (a) 마지막 DiT 블록 직접 residual 통로, (b) timestep 정규화 수정(색 저하와 같은 뿌리). 둘 다 mcs2→run2 사이 변경이며 mcs2엔 없음.
3. edge loss·scale-sync·LPIPS 세기·flow matte 가중 지수(m→m²)는 측정으로 배제(§4). matte 가중 지수는 m=1 내부 픽셀엔 영향이 없어 관찰된 "matte=1 내부에서도 어긋남"을 원리적으로 설명 못함.
4. `logs/run2_log.log` 확보로 phase1→phase2 전환의 사실관계 확정(§3-a): phase1은 epoch31 도중 수동 종료·weights-only 로드였고, phase2는 warmup 없이 즉시 풀파워 LPIPS가 걸리는 "리셋 겸 부스트" 지점 — 잔차 통로 가설을 뒷받침함. run2 phase1의 흐릿함은 `both_aug3x` 회전증강에 의한 회귀-평균 효과로, §3과 독립된 별도 후보(§5)로 남음.
5. 최종 확정은 재학습 기반 구분 실험 필요(§6) — 특히 마지막 블록 residual hook 제거 ablation.

---

## 1. 배경

### 1-1. run1/run2/run3 정의

**run1(=mcs2)** — phase1 unbraid 3000(LR 1e-4) / phase2 braid 1000(LR 2e-5).

**run2** — phase1 unbraid 3000+braid 증강 3000=6000(LR 1e-4) / phase2 동일 데이터(LR 1e-4). 논문 반영차 flow loss·matte 주입 구조 변경(상세 `[0713]training.md`). 문제점: phase2 진행할수록 색 재현·질감 저하(색 원인은 `[0726]`에서 별도 정리, 이 리포트에서 안 다룸).

**run3** — phase1 unbraid 3000(LR 1e-4) / phase2 replay(unbraid+braid 1000+1000, LR 5e-6). run2 문제 개선 목적(상세 `[0723]retrain_plan.md`). 문제점: mcs2 대비 색 학습 저조(§2에서 배제) + phase1부터 내내 푸석함 — matte=1 내부에서도 안 없어져 "머릿결 방향 노이지"로 재정의.

run3는 phase1·phase2 모두 노이지. run2는 phase1은 안 노이지하다가 phase2(epoch5~)부터 노이지. run1(mcs2)은 노이지 없음.

| | mcs2 | run2 | run3 |
|---|---|---|---|
| timestep → DiT | raw σ(0~1) | σ×1000 | σ×1000 |
| 아키텍처 | 17ch, 마지막 블록 residual 없음 | 32ch, **마지막 블록 residual 있음** | 32ch, 마지막 블록 residual 있음(run2와 동일) |
| phase1 데이터/스텝 | unbraid 3000, 187 step/epoch(batch16) | `both_aug3x`(unbraid 3000+braid 증강 3000=6000), 375 step/epoch | unbraid 3000, 187 step/epoch |
| phase2 데이터 | braid 1000 | `both_aug3x`(phase1과 동일) | replay(unbraid+braid 8:8) |
| phase1 실제 종료 | epoch 40 완주 | **epoch30**까지 진행 | epoch 40 완주 |
| 결과 이미지 렌더 시점 | **phase2 완료 후**만 존재 | phase1/2 각 epoch 렌더 존재 | phase1/2 각 epoch 렌더 존재 |

### 1-2. 이미지로 직접 재확인 — "정렬 유지"와 "설정이 옳음"은 다른 말

동일 샘플(`CM_1067`)·동일 20-step 추론 조건으로 4개 시점을 비교했음.

| | 가닥 선명도 | 가닥 방향 정렬 |
|---|---|---|
| mcs2 (phase2 이후) | 선명 | **정렬** |
| run2 phase1 ep10 / ep30 | **살짝 흐릿**(ep30에서 더 흐려짐) | 정렬 |
| run2 phase2 ep5 → ep20 | 급격히 선명해짐 | **어긋남** |
| run3 phase1 ep10 / ep40 | 선명 | **어긋남** |
| run3 phase2 ep5 | 선명 | **어긋남** |

<table>
<tr><th></th><th>mcs2</th><th>run2 phase1 ep10</th><th>run2 phase1 ep30</th><th>run2 phase2 ep20</th></tr>
<tr><th>CM_1067</th>
<td><img src="../outputs/figure/hair-dit_mcs2/gt/CM_1067.png" width="130"></td>
<td><img src="../outputs/results/joint_phase1_epoch10/sketch_gt/CM_1067.png" width="130"></td>
<td><img src="../outputs/results/joint_phase1_epoch30/sketch_gt/CM_1067.png" width="130"></td>
<td><img src="../outputs/results/joint_phase2_epoch20/sketch_gt/CM_1067.png" width="130"></td></tr>
</table>

<table>
<tr><th></th><th>run3 phase1 ep10</th><th>run3 phase1 ep40</th></tr>
<tr><th>CM_1067</th>
<td><img src="../outputs/0725_phase1/epoch10/seed42/paper/sketch_gt/CM_1067.png" width="150"></td>
<td><img src="../outputs/0725_phase1/epoch40/seed42/paper/sketch_gt/CM_1067.png" width="150"></td></tr>
</table>

선명도와 정렬은 원래 독립적으로 움직일 이유가 없음. run2 phase1의 흐릿함은 §5의 별도 후보로, phase2의 선명+어긋남은 §3-a의 잔차 통로 가설로 각각 설명됨.

---

## 2. 문제 정의

색 문제와 질감 문제는 서로 다른 결함이며, 이 리포트는 질감만 다룸.

**색 반영 저하**: `HairLoss`(flow+lpips+edge) 어디에도 색을 직접 겨냥하는 항이 없다는 사실로 정의하고 더 다루지 않음 — 상세 원인·해결 논증은 `[0727]color_texture_reanalysis.md` §3.

**머릿결 방향 노이지**: 가닥 결 방향이 스케치 의도와 다르게 헝클어지는 현상. 본 리포트의 주제이며, §3~§6에서 원인 후보를 좁힘.

---

## 3. 머릿결 노이지 문제 후보

mcs2(17ch, 마지막 블록 residual 없음, raw σ timestep)에는 없고 run2부터 들어온 변경 중 아래 두 가지가 후보로 남음. (flow loss의 matte 가중 지수 변경은 §4-4에서 배제.)

### a. 마지막 DiT 블록 residual 직행 통로

diffusers 기본 forward는 `block_samples[11]`을 block22에 주입하고 마지막 block23엔 주입하지 않는데, 코드는 block23 출력에 **같은 `block_samples[11]`을 한 번 더** 더함. block23 뒤에는 `norm_out→proj_out→unpatchify`뿐이라 attention/MLP로 섞이는 과정 없이 **ControlNet residual이 토큰 단위 그대로 출력 latent 직전까지 직행**하는 통로가 생김. mcs2(17ch)에는 이 통로가 없음.

LPIPS는 그레디언트가 가장 잘 통하는 경로를 따라가므로, 이 통로가 있으면 "자연스러운 결 방향으로 수렴" 대신 "이 통로를 통해 국소 고주파 텍스처를 직접 새겨 넣는" 지름길로 흡수될 유인이 됨.

**증거 1 — `logs/perceptual.log`(held-out 32장, LPIPS는 epoch12부터 활성):**

| epoch | lpips_unbraid | edge_iou_braid | dE_unbraid |
|---|---|---|---|
| 13 (LPIPS 활성 직후) | 0.3899 | 0.0586 | 10.25 |
| 17 | 0.3666 | 0.0631 | 9.60 |
| **22 (최저점)** | **0.3566** | 0.0655 | 8.91 |
| 28 | 0.3677 | 0.0724 | 8.47 |
| 33 | 0.3814 | 0.0779 | 8.39 |
| 39 | 0.3945 | 0.0835 | 8.29 |

epoch22 이후 `lpips_unbraid`는 **+10.6% 악화**되는 동안 `edge_iou_braid`는 **+27.5% 증가**, `dE_unbraid`(색)는 계속 개선됨 — 학습이 덜 된 게 아니라 **LPIPS가 걸린 채 계속 학습할수록 도달하는 수렴점 자체가 노이지한 방향**임을 뜻함. phase2 후반(epoch27~35)은 `lpips_unbraid`·`dE_unbraid` 모두 완전히 정체(더 돌려도 안 나아짐).

**증거 2 — `logs/run2_log.log`: phase1→phase2 전환이 이 통로를 급격히 키우는 조건임.** 이전 판까지는 run2 로그가 없어 phase2 구간이 미해결이었으나, `logs/run2_log.log`(39,517줄) 확보로 아래가 확정됨(phase2 중 CUDA 크래시가 1회 있었으나 EMA·optimizer·lr_scheduler까지 완전 복원되어 관측에 영향 없음).

phase1은 정상 종료가 아니라 epoch30 체크포인트 저장 후 epoch31 도중 수동 종료됐고, phase1→phase2 전환은 weights-only 로드였음 — 옵티마이저 상태가 이어지지 않고 재초기화됨. 게다가 phase2는 warmup 없이 첫 스텝부터 LPIPS가 풀파워로 걸림(`Epoch 1/40` 첫 스텝부터 `loss_lpips=0.1757`, 22727줄; phase1은 `lpips_warmup_frac=0.3×15000 step≈epoch13`부터 활성화, 9109줄). 전환 순간 모델은 ① cosine 감쇠로 낮아진 LR(≈1e-6대)에서 1.0e-4로 재부팅 ② 이미 흐릿하게 수렴한 체크포인트(§1-2) ③ warmup 없이 즉시 최대 강도 LPIPS, 이 세 조건을 동시에 처음 맞음.

이 통로가 LPIPS 그레디언트에 가장 민감한 지름길이라면, fresh 고LR + 즉시 풀파워 LPIPS를 만나는 순간부터 급격히 활성화되는 쪽이 "ep5부터 이미 선명+어긋남"이라는 관찰과 잘 맞음. run3 phase1(처음부터 같은 조건으로 시작하는 from-scratch 학습)이 같은 궤적을 보인 것과 합쳐보면, "선명+어긋남"은 특정 epoch가 쌓여야 나오는 게 아니라 **"1e-4대 LR + 활성 LPIPS + 32ch/hook 아키텍처"가 동시에 갖춰지는 순간 유도되는 수렴 방향**이라는 해석이 이 가설을 뒷받침함.

### b. timestep 정규화 수정 — 색 저하와 같은 뿌리

mcs2는 raw σ(0~1)를 timestep으로 넘겨 모든 노이즈 레벨을 t≈0으로 인식시킴 → SD3.5 prior가 사실상 무력화된 채 ControlNet residual이 생성을 지배(`[0727]` §3-2). run2/run3는 `timesteps_1d = sigmas × num_train_timesteps`(`src/training/trainer.py:608, 746`)로 정규화해 prior가 정상 작동함. prior가 살아나면 SD3.5의 자연 이미지 고주파 사전지식이 sketch stroke의 결 방향과 경쟁하게 됨 — 색 문제(`[0727]` §3)와 같은 메커니즘이 질감에도 적용될 수 있는 후보. run3 phase1 ep10 이미지가 색은 거의 없이 SD3.5가 그릴 법한 자연스러운 웨이브에 가깝다는 관찰이 이 해석과 부합함.

---

## 4. 측정으로 배제되는 것

### 4-1. edge loss — 배제

`logs/perceptual.log`, `logs/phase2.log` 전 구간에서 `R_edge = 0.0036~0.0071`(flow 항 gradient 대비 0.4~0.7%), `stroke_density = 0.03~0.05` — 전체 gradient의 1% 미만. threshold 변경(0.1→1e-3)은 실측으로도 영향 없음이 확인됨.

### 4-2. scale-sync — 배제

run3 flow 항을 전개하면(`src/training/losses.py:66-70`, `243-245`) `run3 = Σ(m²·d²)/Σm × Σm/N = Σ(m²·d²)/N`이고, mcs2 구 정규화(`git show 0033de3`)는 `Σ(m·d²)/N` — **scale-sync 적용 후 전체 스케일이 같음.** 로그 실측 `s_raw`는 phase1 32.9~46.0, phase2 45.5~65.5로 clamp[20,120]에 한 번도 안 걸림. 남는 차이는 matte 가중이 `m`이냐 `m²`이냐는 국소적 차이뿐(`[0727]` §2-2와 동일 결론) — **scale-sync 자체는 원인이 아니라 mcs2와 스케일을 맞춘 조치**로 봐야 함.

### 4-3. LPIPS(세기 자체) — 배제

mcs2 phase1(`0033de3`)과 run3 phase1(`e623ab0`)은 데이터(unbraid 3000)·batch(16)·step/epoch(187)·epochs(40)·LR(1e-4)·loss 가중치(flow 1.0/lpips 0.1/warmup 0.3/edge 0.0)가 전부 동일함. 남는 차이만 대조하면:

| | mcs2 phase1 | run3 phase1 |
|---|---|---|
| flow 항 스케일 | Σ(m·d²)/N | Σ(m²·d²)/N (§4-2, scale-sync 후 동일 자릿수) |
| 아키텍처 | 17ch, 마지막 블록 residual 없음 | 32ch, **마지막 블록 residual 있음**(§3-a) |
| timestep → DiT | raw σ(0~1) | σ×1000 |

결과는 mcs2=정렬, run3=어긋남으로 갈림. **LPIPS라는 신호의 존재 자체는 공통인데 결과가 다르므로, "LPIPS가 세서 문제"라는 설명만으론 불충분** — mcs2→run2 사이 구조 변경(§3-a·b)이 LPIPS 압력을 "정렬"이 아니라 "고주파 세선"으로 새게 만드는 통로 역할을 한다고 봐야 함.

### 4-4. flow loss의 matte 가중 지수(m→m²) — 배제

mcs2는 `Σ(m·d²)`(선형 가중), 현재 코드는 `(m·d)² = m²·d²`(제곱 가중). `m=1`인 완전 내부 픽셀은 `m²=m=1`이라 이 변경의 영향을 전혀 안 받고, `0<m<1`인 soft 경계에서만 감독이 더 깎임(예: m=0.4 → 가중 0.16). **두 근거로 배제함**: (1) analysis.md 자신이 "matte=1인 부분에서도 안 없어짐"을 관찰해 이미 배제한 후보, (2) 대수적으로 m=1 내부는 애초에 영향을 안 받으므로 관찰된 "matte=1 내부에서도 어긋남"을 원리적으로 설명 못함 — §3-a의 통로가 matte 값과 무관하게 전역적으로 작동하는 것과 대조적.

---

## 5. 참고: run2 phase1 흐릿함의 별도 후보 (미확정)

run2 phase1의 LPIPS 활성 스텝 수(epoch13~30, ≈6,375 step, `logs/run2_log.log`)는 run3 phase1(epoch13~40, ≈5,236 step)보다 짧지 않은데도 흐렸음 — "LPIPS 노출 부족" 설명은 이 수치로 반박됨. 남는 차이는 데이터셋: run2 phase1은 `both_aug3x`(unbraid 3000 + braid ±15° 회전증강 3000 = 6000, 375 step/epoch), run3 phase1은 unbraid 3000 단독(187 step/epoch). flow loss는 MSE 계열 회귀라, 회전 증강으로 target latent의 국소 방향 분산이 커지면 회귀가 여러 방향의 평균으로 수렴해 흐려지는 건 MSE형 손실에서 잘 알려진 성질(§3-a의 misalignment 기전과는 독립적인 별도 후보). 확정하려면 동일 아키텍처·스텝 수에서 `unbraid-only` vs `both_aug3x` 대조가 필요 — 이번 조사 범위에서는 **후보로만 기록**하며, §3의 노이지/어긋남 문제와는 별개임.

---

## 6. 다음 단계: 구분 실험 (우선순위 순)

| | 방법 | 비용 | 판정 대상 |
|---|---|---|---|
| A | 기존 run3 체크포인트를 hook 없는(diffusers 기본) forward로 추론 | 무비용 | §3-a — 질감이 정리되면 유력 후보로 승격 |
| B | 추론 시 timestep에 raw σ 전달 | 무비용 | §3-b — mcs2식 결이 재현되면 유력 후보로 승격 |
| C | phase1 짧은 재학습(10~15 epoch) — 마지막 블록 residual hook만 제거 | 저비용 | §3-a 독립 확정. `lpips_unbraid`/`edge_iou` 궤적 비교로 판정 |
| D (부차) | 동일 아키텍처로 `unbraid-only` vs `both_aug3x` phase1을 같은 스텝 수에서 대조 | 저비용 | §5 — run2 phase1의 흐릿함이 misalignment와 독립된 데이터 요인인지 확정 |

A·B는 학습·추론 조건이 달라 단독 결론엔 부족하지만 후보를 좁히는 덴 충분하며, **최종 확정은 C가 필요함**(flow 가중 지수는 §4-4에서 이미 배제되어 C에서 재테스트 불필요). D는 §5 확인용 부차 실험으로 C보다 우선순위 낮음.

---

## 7. Loss 재설계 — 착수 전 검토 항목 (색+방향 통합)

교수님 제안: §6의 hook ablation과는 별개로, 색과 방향(질감)을 loss 설계 단계에서 명시적으로 다루자는 방향. 아래는 최종 설계안이 아니라 **설계 착수 전에 확인·결정해야 할 항목**.

### 7-1. 색 항

- **어디서 잴지**: latent 공간(빠르지만 색 정밀도 낮음) vs decode된 RGB(LPIPS처럼 `x0_pred` 복원 필요, VAE decode 비용 추가).
- **무엇과 비교할지**: matte 내부 평균 Lab vs 스트로크 Lab(ΔE) — 픽셀별 비교 vs 스트로크 단위 비교.
- **증강과의 연결**: `StrokeColorSampler`(`src/data/augmentation.py`)는 GT 머리 픽셀에서만 색을 샘플링해 무지개색이 학습 데이터에 아예 없음(`[0727]` §3-3). 색 loss만 추가하고 증강을 그대로 두면 OOD 색은 여전히 못 배움 — loss와 증강 설계를 같이 검토해야 함.
- **스케일 밸런스**: 기존 flow/LPIPS와 gradient 스케일을 맞춰야 함(§4-2 scale-sync 사례처럼 항을 늘릴 때마다 재보정 필요). `compute_R` 같은 모니터링을 새 항에도 붙여야 같은 실수를 반복 안 함.

### 7-2. 방향(질감) 항

- **Gram loss는 원리적으로 안 맞는 도구** — Gram matrix는 정의상 공간(H,W)을 합쳐 채널 간 상관만 남기는 통계량이라 위치·방향 정보를 버림. "텍스처 통계는 맞는데 방향이 틀림"을 구별 못함.
- **방향을 직접 재려면** structure tensor나 steerable filter(Gabor bank)로 국소 gradient 방향을 뽑아 GT와 각도 차이를 벌점 주는 방식이 필요. 단 GT 방향장이 실제로 존재/추출 가능한지부터 확인해야 함 — 스케치 stroke 자체에 방향 정보가 있는지, 없다면 GT 이미지에서 새 전처리로 뽑아야 하는지.
- latent vs pixel 공간 중 어디서 계산할지도 결정 필요.

### 7-3. 색+방향 동시 도입 시 상호작용

새 항의 그레디언트도 결국 §3-a의 마지막 블록 residual 통로를 거쳐 흐름. hook을 안 고친 채 loss만 추가하면 같은 통로로 또 다른 아티팩트가 새어나갈 수 있음 — §6의 hook ablation과 독립적으로 진행 가능하지만, 결과 해석 시 이 얽힘을 염두에 둬야 함.

### 7-4. 검증 순서

- **재학습 없이 먼저 되는 것**: 기존 체크포인트에 색/방향 지표만 계산해 현재 실패 정도를 정량화(§4-3에서 LPIPS 배제할 때 쓴 방식과 동일 — 무비용).
- **재학습이 필요한 것**: 새 loss 항을 실제로 넣어 수렴 여부·`lpips_unbraid`/`edge_iou` 궤적 변화 확인.
- §6과의 순서·병행 여부 결정 필요 — 특히 hook ablation(C) 결과가 방향 항의 필요 여부·우선순위에 영향을 줄 수 있음.

### 7-5. 선행연구 조사 필요 항목

- hair/strand orientation loss 선례(헤어 모델링·렌더링 분야 방향장 매칭 사례).
- color-aware perceptual loss 사례(지각 손실에 색 항을 결합한 선행 연구).
