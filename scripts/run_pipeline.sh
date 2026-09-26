#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh
# End-to-end orchestration for the Amazon ML 2026 pipeline.
# Usage:  bash scripts/run_pipeline.sh [--config path/to/config.yaml]
# ============================================================

set -euo pipefail   # exit on error, unset var, pipe failure

# ── Defaults ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/config.yaml"
PYTHON="${PYTHON:-python}"

# ── Argument parsing ────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "============================================="
echo " Amazon ML Hackathon 2026 — Entity Resolution"
echo " Config : ${CONFIG}"
echo " Root   : ${PROJECT_ROOT}"
echo "============================================="
cd "${PROJECT_ROOT}"

# ── Helper: timed stage runner ───────────────────────────────
run_stage() {
  local stage_name="$1"
  local module="$2"
  local extra_args="${3:-}"
  echo ""
  echo "[$(date '+%H:%M:%S')] >>> Starting: ${stage_name}"
  ${PYTHON} -m "${module}" --config "${CONFIG}" ${extra_args}
  echo "[$(date '+%H:%M:%S')] <<< Finished: ${stage_name}"
}

# ── Ensure output sub-directories exist ─────────────────────
mkdir -p output/cache output/models

# ── Phase 2: Preprocessing ───────────────────────────────────
run_stage "Phase 2 · Preprocessing" "src.preprocessor"

# ── Phase 3: Blocking ────────────────────────────────────────
run_stage "Phase 3 · Blocking" "src.blocker"

# ── Phase 4: Feature Engineering ────────────────────────────
run_stage "Phase 4 · Feature Engineering" "src.features"

# ── Phase 5: Modelling ───────────────────────────────────────
run_stage "Phase 5 · Modelling" "src.model"

# ── Phase 6: Post-processing ─────────────────────────────────
run_stage "Phase 6 · Post-processing" "src.optimizer"

# ── Phase 7: Output Validation ───────────────────────────────
run_stage "Phase 7 · Validation" "src.utils" "--validate"

echo ""
echo "============================================="
echo " Pipeline complete."
echo " candidate_pairs.tsv  -> output/candidate_pairs.tsv"
echo " matching_results.tsv -> output/matching_results.tsv"
echo "============================================="
