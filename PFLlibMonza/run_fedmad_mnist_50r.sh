#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
SEED=42
CLIENTS=20
MALICIOUS=4
ROUNDS=50

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$HOME/venvs/fedmad-run/bin/python" ]]; then
    PYTHON_BIN="$HOME/venvs/fedmad-run/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

mkdir -p "$RESULTS_DIR"
CC=0
while [[ -e "$RESULTS_DIR/detection_log_MNIST_MAD_cc${CC}.json" || \
        -e "$RESULTS_DIR/MNIST_MAD_${CC}_100_${MALICIOUS}_test_0.h5" ]]; do
    CC=$((CC + 1))
done

RUN_ID="$(date +%Y%m%d_%H%M%S)_mnist_${CLIENTS}c_${ROUNDS}r_seed${SEED}"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"

cat > "$RUN_DIR/config.txt" <<EOF
Python: $PYTHON_BIN
Dataset: MNIST (20-client, Dirichlet non-IID partition)
Algorithm: MAD (adaptive FedMAD)
Total clients: $CLIENTS
Malicious clients: $MALICIOUS (20%)
Attack: cyclic label flip y -> (y + 1) mod 10, active from round 0
Communication rounds: $ROUNDS (main.py -gr 49 because the loop includes round 0)
Client participation: 100% each round
Local epochs: 1
Device: CPU
Seed: $SEED
Output discriminator (-cc): $CC
EOF

cd "$SCRIPT_DIR/system"
set -o pipefail
"$PYTHON_BIN" -u main.py \
    -algo MAD \
    -data MNIST \
    -m CNN \
    -nc "$CLIENTS" \
    -nmc "$MALICIOUS" \
    -atk label \
    -rfake 1 \
    -ria -1 \
    -gr "$((ROUNDS - 1))" \
    -jr 1.0 \
    -ls 1 \
    -lbs 10 \
    -lr 0.005 \
    -cc "$CC" \
    --seed "$SEED" \
    -dev cpu \
    -mad_agents all \
    -mad_byzantine_f "$MALICIOUS" \
    -slm_e False \
    2>&1 | tee "$RUN_DIR/run.log"

cp "$RESULTS_DIR/MNIST_MAD_${CC}_100_${MALICIOUS}_test_0.h5" "$RUN_DIR/metrics.h5"
cp "$RESULTS_DIR/detection_log_MNIST_MAD_cc${CC}.json" "$RUN_DIR/detection_log.json"
cp "$SCRIPT_DIR/system/models/MNIST/MAD_server.pt" "$RUN_DIR/MAD_server.pt"

echo "Execucao concluida. Artefatos: $RUN_DIR"
