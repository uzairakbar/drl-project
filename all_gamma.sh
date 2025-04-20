#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 SEED"
  exit 1
fi

SEED="$1"
SESSION="humanoid_run"
WINDOW="0"
GAMMAS=(0.0 0.01 0.001 0.0001 0.00001 0.000001)

# 1) Check if session exists already
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Using existing tmux session '$SESSION'"
else
  echo "Creating new tmux session '$SESSION'"
  tmux new-session -d -s "$SESSION"

  # split into 5 panes (pane 0 exists)
  for i in {1..5}; do
    tmux split-window -h -t "${SESSION}:$WINDOW.$((i-1))"
  done

  # tile evenly
  tmux select-layout -t "${SESSION}:$WINDOW" tiled
fi

# 2) Send commands into each pane
# 2) In each pane: first activate, then run
for idx in "${!GAMMAS[@]}"; do
  GAM="${GAMMAS[$idx]}"
  TARGET="${SESSION}:$WINDOW.$idx"

  # activate conda env
  tmux send-keys -t "$TARGET" "conda activate drl" C-m
  # run training
  tmux send-keys -t "$TARGET" \
    "python train.py \
      --config configs/R3GAIL.yaml \
      --env_name Humanoid-v5 \
      --log_dir humanoid_run \
      --seed ${SEED} \
      --gamma_d ${GAM} \
      --gamma_g ${GAM}" C-m
done

# 3) Attach to session
echo "Attaching to tmux session '$SESSION'"
tmux attach -t "$SESSION"
