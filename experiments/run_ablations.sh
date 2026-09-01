#!/usr/bin/env bash
# Ablation, reward-weight sensitivity, and fleet-capacity studies.
set -e
cd "$(dirname "$0")/.."          # repo root
export TF_CPP_MIN_LOG_LEVEL=3
LOG=experiments/logs; MDL=experiments/models; T=500000
mkdir -p "$LOG" "$MDL"

# Trained single-agent PPO ablation
python3 experiments/train_eval_single_ppo.py --scenario S2_MEDIUM --timesteps $T --n-seeds 30 --tag 500k

# Reward-weight sensitivity (cost,time,emission,recycling)
run_rw () { python3 experiments/train_mappo.py --scenario S2_MEDIUM --timesteps $T --reward-weights "$2" --out-dir "$MDL/rw_$1" > "$LOG/rw_$1.log" 2>&1
            python3 experiments/eval_mappo_only.py --scenario S2_MEDIUM --model "$MDL/rw_$1/mappo_best.pt" --reward-weights "$2" --n-seeds 30 --tag "rw_$1"; }
run_rw costHeavy "0.70,0.10,0.10,0.10"; run_rw timeHeavy "0.10,0.70,0.10,0.10"
run_rw emisHeavy "0.10,0.10,0.70,0.10"; run_rw recyHeavy "0.10,0.10,0.10,0.70"
run_rw pe20 "0.60,0.10,0.20,0.10"; run_rw pe40 "0.40,0.10,0.40,0.10"; run_rw pe60 "0.20,0.10,0.60,0.10"

# Fleet-capacity scaling
run_cap () { python3 experiments/train_mappo.py --scenario S2_MEDIUM --timesteps $T --capacity-scale "$1" --out-dir "$MDL/cap_x$1" > "$LOG/cap_$1.log" 2>&1
             python3 experiments/eval_mappo_only.py --scenario S2_MEDIUM --model "$MDL/cap_x$1/mappo_best.pt" --capacity-scale "$1" --n-seeds 30 --tag "cap_x$1"; }
run_cap 3; run_cap 6; run_cap 12

# Action-masking ablation
python3 experiments/train_mappo.py --scenario S2_MEDIUM --timesteps $T --mask-mode none --out-dir "$MDL/mask_none" > "$LOG/mask_none.log" 2>&1
python3 experiments/train_mappo.py --scenario S2_MEDIUM --timesteps $T --mask-mode soft --out-dir "$MDL/mask_soft" > "$LOG/mask_soft.log" 2>&1
python3 experiments/eval_masking.py --scenario S2_MEDIUM --mask-mode hard --model "$MDL/S2_MEDIUM/mappo_best.pt" --tag hard --n-seeds 30
python3 experiments/eval_masking.py --scenario S2_MEDIUM --mask-mode none --model "$MDL/mask_none/mappo_best.pt" --tag none --n-seeds 30
python3 experiments/eval_masking.py --scenario S2_MEDIUM --mask-mode soft --model "$MDL/mask_soft/mappo_best.pt" --tag soft --n-seeds 30
python3 experiments/eval_masking.py --scenario S2_MEDIUM --mask-mode hard --model "$MDL/S2_MEDIUM/mappo_best.pt" --corrupt-prob 0.10 --tag hard_corrupt10 --n-seeds 30
python3 experiments/eval_masking.py --scenario S2_MEDIUM --mask-mode hard --model "$MDL/S2_MEDIUM/mappo_best.pt" --corrupt-prob 0.20 --tag hard_corrupt20 --n-seeds 30
echo "Done."
