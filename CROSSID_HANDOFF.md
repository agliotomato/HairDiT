# Cross-Identity "알록달록" 실험 — 핸드오프 문서

다른 머신/에이전트가 이 문서만 읽고 실험을 이어갈 수 있도록 정리.
(repo: `github.com/agliotomato/HairDiT`, 작업 경로: `/home/agliotomato/HairDiT`)

---

## 0. 한 줄 요약

SketchHair-DiT(SD3.5 + ControlNet, 스케치로 헤어 편집) 모델의 **외형 누설(leakage)** 을
진단하는 추론-only 실험. 테스트 이미지가 무지개색 헤어라 "알록달록".

---

## 1. 모델 (mcs1~6) — 무엇이 다른가

SD3.5-medium 백본 + HairControlNet. 6구성은 **matte conditioning** 방식만 다름:

| 코드 | 논문명 | MatteCNN | raw matte(1ch) | gate |
|---|---|:---:|:---:|:---:|
| mcs1 | Ours | ✅ | ✅ | ❌ |
| mcs2 | Ours+Gate | ✅ | ✅ | ✅ |
| mcs3 | Sketch-only | ❌ | ❌ | ❌ |
| mcs4 | Sketch-only+Gate | ❌ | ❌ | ✅ |
| mcs5 | Raw-only | ❌ | ✅ | ❌ |
| mcs6 | Matte-CNN-only | ✅ | ❌ | ❌ |

ControlNet 입력 17ch = `cat([sketch_latent(16) + MatteCNN_feat, raw_matte(1)])`.
구성별 차이는 config(`configs/mcs{N}_phase2.yaml`)의 `zero_matte_cond` / `zero_matte_feat` / `zero_raw_matte` / `schedule(gate)` 로 제어.

---

## 2. 실험 설계 (Cross-TYPE)

**목적**: 배경(face/img)을 **다른 인물**로 바꿔도 출력 헤어가 sketch/GT에 충실한가
= "모델이 배경 헤어 외형을 베끼나(leakage)?"

**Cross-TYPE 매핑** (유형 교차, derangement seed 42 고정):
- braid sketch(A) → **unbraid** 이미지(B)
- unbraid sketch(A) → **braid** 이미지(B)
- `sketch = matte = A`(해당 유형), `face/img = B`(반대 유형, 다른 인물)
- 개수: braid 107, unbraid 466 → 비대칭이라 **중복 허용**(unbraid→braid는 braid 이미지 ~4.4회 재사용)

**핵심 메커니즘 (BLD)**: 추론 중 매 스텝 matte 바깥을 face(B)의 noised latent로 주입.
즉 헤어 영역(A matte) = 생성, 배경 = B. → 출력 stem = **A_id** (face symlink trick).

**sketch 2버전**:
- `gt_sketch`: A 자신의 (img×matte)로 재착색 → sketch가 A 실제 헤어색을 담음
- `original_sketch`: dataset 원본 스케치 그대로 (재착색 X)

**노이즈 시드 3개**: 42, 1, 2 (매핑 고정, 노이즈만 변동 → mean±std로 재현성 통계)

**매트릭스**: 2 type × 6 mcs × (버전) × 3 seed.
- gt_sketch: 36 run (완료)
- original_sketch: 36 run (진행/예정)

---

## 3. 파이프라인 (스크립트)

순서대로:

```bash
# ① 매핑+symlink+gt-recolor 사전생성 (1회, torch 필요)  — gt_sketch만 해당
.venv/bin/python scripts/preprocess/gen_cross_id_map.py
#   → experiment_cross_id/{tripletmap_*.json, face_links/, gt_recolored_sketch/}

# ② 체크포인트 strip (1회) — 20GB ckpt에서 ControlNet만 → _infer.pth(~6GB)
.venv/bin/python scripts/preprocess/strip_checkpoint.py
#   → checkpoints/mcs{N}_phase2/epoch_40_infer.pth

# ③ 추론 매트릭스
VERSIONS=original_sketch bash scripts/run_cross_id_experiment.sh
#   → outputs/crossid/{type}/mcs{N}_{version}/seed{S}/{A_id}.png

# ④ 평가 (3seed 집계)
.venv/bin/python scripts/eval/eval_crossid.py
#   → eval_results/crossid/summary_3seed.csv
```

핵심 파일:
- `scripts/preprocess/gen_cross_id_map.py` — cross-type derangement, face symlink, gt-recolor
- `scripts/preprocess/strip_checkpoint.py` — ControlNet 추출 (mmap)
- `scripts/run_cross_id_experiment.sh` — 추론 매트릭스. env: `VERSIONS`, `MCS`, `SEEDS`, `TYPES`, `THROTTLE`, `TEMP_LIMIT`, `COOLDOWN`, `NPROC`, `FORCE`
- `scripts/eval/eval_crossid.py` — `scripts/eval_metrics.py` 재사용 driver, 3seed 집계
- `scripts/infer_custom.py` — 추론 본체. `--seed`, `--throttle` 옵션 있음

---

## 4. 평가 지표 (헤어 영역만, GT matte 마스킹)

| 지표 | 방향 | 의미 | 비고 |
|---|---|---|---|
| Sketch LPIPS | ↓ | 구조(vs sketch, Canny 엣지) | braid/unbraid 분리 |
| Edge IoU | ↑ | 구조(엣지 정렬, binary IoU) | 분리 |
| Hair FID | ↓ | 리얼리즘 | **통합 573만** (2048-dim 공분산) |
| LPIPS(GT) | ↓ | 외형(vs A GT) | 분리 |
| PSNR | ↑ | 화질(vs A GT) | 분리 |
| ΔE2000(GT) | ↓ | 색(CIEDE2000+음영정규화 mean hue) | **gt_sketch만** |

- 마스킹: `dataset/{split}/matte/test`(A 기준). LPIPS·FID·PSNR·ΔE = soft-alpha, Edge IoU = binary.
- 출력 stem = A_id 이므로 eval_metrics가 자동으로 잡는 GT/sketch/matte 전부 A 기준.
- ΔE2000은 **gt_sketch에만** (original은 sketch가 A색이 아니라 의미 없음).

---

## 5. 환경 (⚠️ 버전 중요)

- **venv**: `/home/agliotomato/HairDiT/.venv` (python 3.10)
- **torch 2.3.1+cu118**, torchvision 0.18.1
- **diffusers 0.32.2** ← ⚠️ **0.31.0은 안 됨** (`SD3ControlNetModel`이 `dual_attention_layers` 못 받음. SD3.5는 dual attention 필수)
- transformers **4.44.2** (5.x는 torch≥2.4 요구해서 PyTorch 비활성화됨), huggingface_hub 0.24.6, tokenizers 0.19.1, accelerate 0.34.2
- numpy **<2** (cv2/skimage 호환), opencv-python-headless, scikit-image, pytorch-fid, matplotlib, lpips, kornia, einops
- **SD3.5-medium**: HF 캐시 `~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium` 에 있어야 함 (model_id는 `configs/base.yaml`)

재현 설치:
```bash
python3 -m venv .venv && .venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install "numpy<2" diffusers==0.32.2 transformers==4.44.2 \
  huggingface_hub==0.24.6 tokenizers==0.19.1 accelerate==0.34.2 \
  lpips kornia einops PyYAML tqdm opencv-python-headless scikit-image pytorch-fid matplotlib
```

---

## 6. 함정 / 트러블슈팅

1. **diffusers 버전**: 0.31 → `dual_attention_layers` TypeError. **0.32.2 써라.**
2. **체크포인트 20GB OOM**: 추론이 full ckpt를 통째로 로드하면 RAM 부족.
   - `strip_checkpoint.py`로 `_infer.pth`(6GB) 미리 생성 → infer_custom이 자동으로 그걸 우선 로드.
   - strip 시 `torch.load(mmap=True)` 사용. RAM<21GB면 `sudo sysctl vm.overcommit_memory=1` 필요(아니면 mmap ENOMEM).
   - `--checkpoint`엔 `epoch_40.pth`를 넘기면 코드가 옆의 `_infer.pth`를 자동 탐색.
3. **GPU 셧다운 (이전 머신)**: 추론 중 순간 전력 스파이크(~187W)로 PSU OCP 추정 → PC 전체 셧다운.
   - WSL에선 `nvidia-smi -pl`(전력제한) **불가**(Insufficient Permissions).
   - 대책: ① Windows MSI Afterburner Power Limit 68% (확실), ② 하나씩 + 쿨다운, ③ `--throttle 1.5`(이미지당 sleep, in-run 발열 완화, 단 스파이크는 못 막음).
   - run 스크립트 env: `TEMP_LIMIT`, `COOLDOWN`, `THROTTLE`, `NPROC`로 조절.
   - skip 로직: 출력 폴더에 png 있으면 자동 SKIP → 셧다운 후 재개 안전. (잘린 run = 부분 png는 지우고 재실행)
4. **WSL vs Windows Git Bash**: 작업은 반드시 **WSL** 터미널(`agliotomato@...:~/HairDiT$`)에서. MINGW64(Windows)엔 `/home/...` 경로 없음.
5. **git**: outputs/·checkpoints(심볼릭)·.pyc·로그·데이터(unbraid_new 등)는 커밋 금지(.gitignore 처리됨). 코드+보고서만.

---

## 7. 현재 상태 (2026-06-19 기준)

- ✅ **gt_sketch 36 run 완료** (10,314장) + eval 완료 → `reports/[0618][서현택]3seed.md`
- 🔄 **original_sketch 36 run** 시작/진행 (새 컴퓨터로 교체, 발열 재확인 중)
- 체크포인트 6개 strip 완료(`_infer.pth`)
- 미커밋: cross-id 스크립트 4종 + infer_custom 수정 + 보고서 (push 대기)

### gt_sketch 결과 핵심 (주의)
가설("matte conditioning이 누설 막아 Ours가 A에 더 충실")이 **깔끔히 입증 안 됨**:
- ΔE2000·Hair FID·Sketch LPIPS는 **Sketch-only가 더 좋음**
- Ours 우위(LPIPS-GT·PSNR·Edge IoU)는 노이즈 수준
- 원인: gt_sketch는 sketch에 A색이 적혀 있어 Sketch-only가 트리비얼하게 색 맞힘.
  **cross-id 절대값으론 누설 측정 불가 → drop(same-id vs cross-id) 비교 필요.**

### 다음 후보
1. original_sketch 완료 → gt_sketch와 비교 (sketch에 정답색 없을 때 패턴 바뀌나)
2. **same-id baseline**(face=A 본인, 573장) → drop으로 누설 정량화 ← 핵심
3. cross-id 설계 재검토

---

## 8. 데이터 레이아웃 (참고)

```
dataset/{braid,unbraid}/{img,sketch,matte}/test/*.png   (braid 107, unbraid 466)
experiment_cross_id/
  tripletmap_{braid,unbraid}.json        {A_id: B_id}
  face_links/{type}/{A_id}.png -> dataset/{반대type}/img/test/{B_id}.png
  gt_recolored_sketch/{type}/{A_id}.png
checkpoints/ -> (심볼릭) Hair-DiT-Forward/checkpoints/mcs{N}_phase2/{epoch_40.pth, epoch_40_infer.pth}
outputs/crossid/{type}/mcs{N}_{version}/seed{42,1,2}/{A_id}.png
eval_results/crossid/{mcs{N}_{version}_seed{S}_summary.csv, summary_3seed.csv}
```
