#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
SEED=42
CLIENTS=20
MALICIOUS=4
ROUNDS=50
CC=5

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
RUN_ID="$(date +%Y%m%d_%H%M%S)_fedavg_mnist_${CLIENTS}c_${ROUNDS}r_seed${SEED}"
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"

GOAL=test
H5_PATH="$RESULTS_DIR/MNIST_FedAvg_${CC}_100_${MALICIOUS}_${GOAL}_0.h5"
if [[ -e "$H5_PATH" ]]; then
    GOAL="test_${RUN_ID}"
    H5_PATH="$RESULTS_DIR/MNIST_FedAvg_${CC}_100_${MALICIOUS}_${GOAL}_0.h5"
fi

cat > "$RUN_DIR/config.txt" <<EOF
Python: $PYTHON_BIN
Dataset: MNIST (20-client, Dirichlet non-IID partition)
Algorithm: FedAvg (plain weighted averaging, no client filtering)
Total clients: $CLIENTS
Malicious clients: $MALICIOUS (20%)
Attack: cyclic label flip y -> (y + 1) mod 10, active from round 0
Communication rounds: $ROUNDS (main.py -gr 49 because the loop includes round 0)
Client participation: 100% per round
Local epochs: 1
Device: CPU
Seed: $SEED
Cluster comparison mode: $CC (mode 5 disables extra filtering)
EOF

FPR_CSV="$SCRIPT_DIR/system/f.csv"
FPR_CSV_BACKUP="$RUN_DIR/f.csv.before"
FPR_CSV_EXISTED=0
if [[ -e "$FPR_CSV" ]]; then
    cp "$FPR_CSV" "$FPR_CSV_BACKUP"
    FPR_CSV_EXISTED=1
fi
restore_fpr_csv() {
    if [[ "$FPR_CSV_EXISTED" -eq 1 ]]; then
        cp "$FPR_CSV_BACKUP" "$FPR_CSV"
    else
        rm -f "$FPR_CSV"
    fi
}
trap restore_fpr_csv EXIT

cd "$SCRIPT_DIR/system"
"$PYTHON_BIN" -u main.py \
    -algo FedAvg \
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
    -go "$GOAL" \
    --seed "$SEED" \
    -dev cpu \
    -mad_byzantine_f "$MALICIOUS" \
    -slm_e False \
    2>&1 | tee "$RUN_DIR/run.log"

cp "$H5_PATH" "$RUN_DIR/metrics.h5"
cp "$SCRIPT_DIR/system/models/MNIST/FedAvg_server.pt" "$RUN_DIR/FedAvg_server.pt"
if [[ -e "$FPR_CSV" ]]; then
    cp "$FPR_CSV" "$RUN_DIR/fedavg_diagnostics.csv"
fi
restore_fpr_csv
trap - EXIT

echo "Execucao FedAvg concluida. Artefatos: $RUN_DIR"
