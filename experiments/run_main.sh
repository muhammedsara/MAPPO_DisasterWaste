#!/usr/bin/env bash
# Train the four scenario policies and run the paired 30-seed benchmark.
set -e
cd "$(dirname "$0")/.."          # repo root
export TF_CPP_MIN_LOG_LEVEL=3
LOG=experiments/logs; MDL=experiments/models
mkdir -p "$LOG" "$MDL"

for SC in S2_MEDIUM S1_SMALL S4_SEVERE S3_LARGE; do
  echo ">>> TRAIN $SC"
  python3 experiments/train_mappo.py --scenario "$SC" --timesteps 500000 \
      --out-dir "$MDL/$SC" > "$LOG/train_$SC.log" 2>&1
done

for SC in S1_SMALL S2_MEDIUM S3_LARGE S4_SEVERE; do
  echo ">>> EVAL $SC"
  python3 experiments/eval_multiseed.py --scenario "$SC" \
      --model "$MDL/$SC/mappo_best.pt" --n-seeds 30 --base-seed 1000 \
      --ga-pop 30 --ga-gen 50 --skip-milp > "$LOG/eval_$SC.log" 2>&1
done
echo "Done. Results in experiments/results/, then run: python3 experiments/analyze.py"
