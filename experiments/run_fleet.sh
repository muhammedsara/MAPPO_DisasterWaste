#!/usr/bin/env bash
# Fleet-size sensitivity: retrain + eval MAPPO for K in {5,15,20}; K=10 uses the base S2 model.
set -e
cd "$(dirname "$0")/.."
export TF_CPP_MIN_LOG_LEVEL=3
LOG=experiments/logs; MDL=experiments/models
mkdir -p "$LOG" "$MDL"
for K in 5 15 20; do
  echo ">>> FLEET train K=$K $(date)"
  python3 experiments/train_mappo_dr.py --scenario S2_MEDIUM --timesteps 500000 \
      --n-vehicles $K --out-dir "$MDL/fleet_K$K" > "$LOG/fleet_train_K$K.log" 2>&1
  python3 experiments/eval_mappo_only.py --scenario S2_MEDIUM \
      --model "$MDL/fleet_K$K/mappo_best.pt" --n-vehicles $K --n-seeds 30 --tag "fleet_K$K" \
      > "$LOG/fleet_eval_K$K.log" 2>&1
done
echo ">>> FLEET eval K=10 (base model) $(date)"
python3 experiments/eval_mappo_only.py --scenario S2_MEDIUM \
    --model "$MDL/S2_MEDIUM/mappo_best.pt" --n-seeds 30 --tag "fleet_K10" > "$LOG/fleet_eval_K10.log" 2>&1
echo "===== FLEET SENSITIVITY DONE $(date) ====="
