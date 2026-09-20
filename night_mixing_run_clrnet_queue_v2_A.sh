#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}

STUDY_ROOT=/root/autodl-tmp/night_mixing_study_v2
CLRNET_ROOT=/root/autodl-tmp/CLRNet_laneA_v2
LOG_ROOT="$STUDY_ROOT/logs"
mkdir -p "$LOG_ROOT"
cd "$CLRNET_ROOT"

for exp_name in "$@"; do
  echo "===== START ${exp_name} $(date -Iseconds) lane=A =====" | tee -a "$LOG_ROOT/queue_A.log"
  rm -rf cache
  python main.py "$STUDY_ROOT/configs/${exp_name}.py" --gpus 0 > "$LOG_ROOT/${exp_name}_train.log" 2>&1
  python "$STUDY_ROOT/scripts/night_mixing_eval_clrnet_v2.py" --exp-name "${exp_name}" > "$LOG_ROOT/${exp_name}_eval.log" 2>&1
  echo "===== DONE ${exp_name} $(date -Iseconds) lane=A =====" | tee -a "$LOG_ROOT/queue_A.log"
done
