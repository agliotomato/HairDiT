#!/usr/bin/env bash
# Cross-identity 추론 매트릭스 실행.
#   2 유형(braid/unbraid) × 6 mcs × 3 노이즈시드 = 36 run  (gt_sketch만)
# NPROC(기본 1) 프로세스로 실행. 로컬 단일 GPU면 1, 여유 있으면 NPROC=2~3.
#
# 사전: scripts/preprocess/gen_cross_id_map.py 먼저 실행 (face_links/, gt_recolored_sketch/).
# 출력: outputs/crossid/{type}/mcs{N}_{version}/seed{S}/{A_id}.png
#
# 사용:
#   bash scripts/run_cross_id_experiment.sh
#   NPROC=2 SEEDS="0 1 2" bash scripts/run_cross_id_experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# venv python 우선 (PATH에 python 없을 수 있음). PY 환경변수로 override 가능.
PY="${PY:-$(ls .venv/bin/python 2>/dev/null || echo python3)}"
NPROC="${NPROC:-1}"
SEEDS="${SEEDS:-42 1 2}"
TYPES="${TYPES:-braid unbraid}"
MCS="${MCS:-1 2 3 4 5 6}"
NUM_STEPS="${NUM_STEPS:-20}"
THROTTLE="${THROTTLE:-0}"   # 이미지당 sleep 초 (in-run 발열/전력 완화). 예: THROTTLE=1.0
XID_ROOT="experiment_cross_id"
OUT_ROOT="outputs/crossid"

# (version_name, sketch_dir_template) — {t}=type 치환.
#   gt_sketch       = A 실제색 재착색 (experiment_cross_id/gt_recolored_sketch)
#   original_sketch = 원본 스케치 그대로 (dataset/{t}/sketch/test)
# VERSIONS 환경변수로 선택 가능 (기본: 둘 다). 예: VERSIONS=original_sketch
declare -A SKETCH_DIR_ALL=(
  [gt_sketch]="${XID_ROOT}/gt_recolored_sketch/{t}"
  [original_sketch]="dataset/{t}/sketch/test"
)
VERSIONS="${VERSIONS:-gt_sketch original_sketch}"
declare -A SKETCH_DIR=()
for v in $VERSIONS; do SKETCH_DIR[$v]="${SKETCH_DIR_ALL[$v]}"; done

JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT

for t in $TYPES; do
  matte_dir="dataset/${t}/matte/test"
  face_dir="${XID_ROOT}/face_links/${t}"
  for n in $MCS; do
    cfg="configs/mcs${n}_phase2.yaml"
    ckpt="checkpoints/mcs${n}_phase2/epoch_40.pth"
    for ver in "${!SKETCH_DIR[@]}"; do
      sdir="${SKETCH_DIR[$ver]/\{t\}/$t}"
      for s in $SEEDS; do
        out="${OUT_ROOT}/${t}/mcs${n}_${ver}/seed${s}"
        echo "$PY scripts/infer_custom.py" \
             "--config ${cfg} --checkpoint ${ckpt}" \
             "--sketch ${sdir} --matte ${matte_dir} --face ${face_dir}" \
             "--seed ${s} --num_steps ${NUM_STEPS} --throttle ${THROTTLE} --output_dir ${out}" >> "$JOBS"
      done
    done
  done
done

total=$(wc -l < "$JOBS")
echo "총 ${total} run · 동시 ${NPROC} 프로세스"
# 각 라인을 mkdir 후 실행
xargs -P "$NPROC" -I {} bash -c '
  out=$(echo "{}" | sed -n "s/.*--output_dir \([^ ]*\).*/\1/p")
  # 이미 결과 png가 있으면 skip (재실행/이어돌리기 안전). FORCE=1 이면 무시하고 덮어씀.
  if [ "${FORCE:-0}" != "1" ] && ls "$out"/*.png >/dev/null 2>&1; then
    echo "=== SKIP (이미 있음): $out ==="
    exit 0
  fi
  # 온도 게이트: TEMP_LIMIT 설정 시 GPU가 그 아래로 식을 때까지 대기 (발열 셧다운 방지)
  if [ -n "${TEMP_LIMIT:-}" ]; then
    while t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); \
          [ -n "$t" ] && [ "$t" -ge "$TEMP_LIMIT" ]; do
      echo "    GPU ${t}C >= ${TEMP_LIMIT}C → 쿨다운 대기..."; sleep 30
    done
  fi
  mkdir -p "$out"
  echo ">>> $out"
  {} > "${out}/run.log" 2>&1
  # run 사이 쿨다운 (초). COOLDOWN 미설정 시 0.
  [ -n "${COOLDOWN:-}" ] && sleep "${COOLDOWN}"
' < "$JOBS"

echo "완료. 결과: ${OUT_ROOT}/"
