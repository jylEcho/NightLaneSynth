#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

STUDY_ROOT=/root/autodl-tmp/night_mixing_study_v2
LOG_ROOT="$STUDY_ROOT/logs"
mkdir -p "$LOG_ROOT"

for pair in "CLRNet CLRNet_laneA_v2" "CLRNet CLRNet_laneB_v2" "CLRNet CLRNet_laneC_v2"; do
  src=$(echo "$pair" | awk '{print $1}')
  dst=$(echo "$pair" | awk '{print $2}')
  if [ ! -d "/root/autodl-tmp/${dst}" ]; then
    cp -a "/root/autodl-tmp/${src}" "/root/autodl-tmp/${dst}"
  fi
done

nohup bash /root/autodl-tmp/night_mixing_run_clrnet_queue_v2_A.sh \
  exp1_day_only \
  exp3_mix_syn75_real25 \
  exp4_real_budget_100 \
  exp4_real_budget_500_plus_syn_full \
  > "$LOG_ROOT/queue_A_launcher.log" 2>&1 < /dev/null &
echo $! > "$LOG_ROOT/queue_A.pid"

nohup bash /root/autodl-tmp/night_mixing_run_clrnet_queue_v2_B.sh \
  exp1_day_plus_syn_night_full \
  exp3_mix_syn100_real0 \
  exp3_mix_syn25_real75 \
  exp4_real_budget_500 \
  exp4_real_budget_1000_plus_syn_full \
  > "$LOG_ROOT/queue_B_launcher.log" 2>&1 < /dev/null &
echo $! > "$LOG_ROOT/queue_B.pid"

nohup bash /root/autodl-tmp/night_mixing_run_clrnet_queue_v2_C.sh \
  exp1_day_plus_real_night_full \
  exp3_mix_syn50_real50 \
  exp3_mix_syn0_real100 \
  exp4_real_budget_100_plus_syn_full \
  exp4_real_budget_1000 \
  > "$LOG_ROOT/queue_C_launcher.log" 2>&1 < /dev/null &
echo $! > "$LOG_ROOT/queue_C.pid"

echo "launched"
