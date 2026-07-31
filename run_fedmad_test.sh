#!/usr/bin/env bash
# ============================================================
# FedMAD - Script de Testes (dataset: MNIST)
# Uso:
#   bash run_fedmad_test.sh [mode] [num_atks] [num_rounds] [num_clientes]
#
# Modes:
#   single        : uma execucao MAD baseline (-atk all)   [padrao]
#   attack_label  : uma execucao MAD com -atk label (BA/ASR)
#   robustness    : serie -atk label com nmc = 3,5,8,10 (curva % maliciosos)
#   comparison    : FedAvg vs agentes isolados vs MAD full (-atk label)
#   all           : attack_label + robustness + comparison
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/PFLlibMonza/system"

# Valores padrao (leves para nao sobrecarregar o servidor)
MODE=${1:-single}
if [[ "$MODE" =~ ^[0-9]+$ ]]; then
    # retrocompatibilidade: primeiro argumento era num_atks
    NM=$MODE
    MODE=single
else
    NM=${2:-5}          # numero de clientes maliciosos
fi
GR=${3:-50}         # rounds globais
NC=${4:-50}         # total de clientes

# Ativar ambiente virtual (prioriza venv_wsl, depois .venv)
PYTHON="python"
if [ -d "$SCRIPT_DIR/venv_wsl" ]; then
    PYTHON="$SCRIPT_DIR/venv_wsl/bin/python"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

cd "$SYSTEM_DIR"

echo "============================================"
echo " FedMAD Tests - MNIST | mode=$MODE"
echo "============================================"
echo " Clientes totais : $NC"
echo " Clientes maliciosos (default): $NM"
echo " Rounds globais  : $GR"
echo "============================================"

run_experiment() {
    local desc="$1"; shift
    echo ""
    echo ">>>>> $desc"
    $PYTHON main.py "$@"
}

run_mad() {
    # $1 = descricao; demais = flags extras de main.py
    local desc="$1"; shift
    run_experiment "$desc" \
        -algo MAD \
        -nc "$NC" \
        -gr "$GR" \
        -jr 1.0 \
        -eg 5 \
        -cc 6 \
        -rfake 1 \
        -data MNIST \
        -m CNN \
        -t 1 \
        -ls 1 \
        -dev cuda \
        -did 0 \
        -slm_e False \
        "$@"
}

run_fedavg() {
    # $1 = descricao; demais = flags extras de main.py
    local desc="$1"; shift
    run_experiment "$desc" \
        -algo FedAvg \
        -nc "$NC" \
        -gr "$GR" \
        -jr 1.0 \
        -eg 5 \
        -cc 6 \
        -rfake 1 \
        -data MNIST \
        -m CNN \
        -t 1 \
        -ls 1 \
        -dev cuda \
        -did 0 \
        "$@"
}

# ---------- single ----------
if [ "$MODE" = "single" ]; then
    run_mad "FedMAD baseline (-atk all)" -nmc "$NM" -atk all \
        -agent_em True -agent_fedre True -agent_bhv True -agent_flg True

# ---------- attack_label ----------
elif [ "$MODE" = "attack_label" ]; then
    run_mad "FedMAD -atk label" -nmc "$NM" -atk label \
        -agent_em True -agent_fedre True -agent_bhv True -agent_flg True

# ---------- robustness ----------
elif [ "$MODE" = "robustness" ]; then
    for nmc in 3 5 8 10; do
        run_mad "FedMAD -atk label, nmc=$nmc (robustez)" -nmc "$nmc" -atk label \
            -agent_em True -agent_fedre True -agent_bhv True -agent_flg True
    done

# ---------- comparison ----------
elif [ "$MODE" = "comparison" ]; then
    run_fedavg "FedAvg vulnerable (-atk label)" -nmc "$NM" -atk label
    run_mad "FedMAD - so EmInspector" -nmc "$NM" -atk label \
        -agent_em True -agent_fedre False -agent_bhv False -agent_flg False
    run_mad "FedMAD - so FedREDefense" -nmc "$NM" -atk label \
        -agent_em False -agent_fedre True -agent_bhv False -agent_flg False
    run_mad "FedMAD - so Behavior" -nmc "$NM" -atk label \
        -agent_em False -agent_fedre False -agent_bhv True -agent_flg False
    run_mad "FedMAD - so FedLLMGuard" -nmc "$NM" -atk label \
        -agent_em False -agent_fedre False -agent_bhv False -agent_flg True
    run_mad "FedMAD - full (todos os agentes)" -nmc "$NM" -atk label \
        -agent_em True -agent_fedre True -agent_bhv True -agent_flg True

# ---------- all ----------
elif [ "$MODE" = "all" ]; then
    bash "$0" attack_label "$NM" "$GR" "$NC"
    bash "$0" robustness "$NM" "$GR" "$NC"
    bash "$0" comparison "$NM" "$GR" "$NC"

else
    echo "Modo desconhecido: $MODE" >&2
    echo "Modos: single, attack_label, robustness, comparison, all" >&2
    exit 1
fi

echo ""
echo "============================================"
echo " Concluido em $(date)"
echo "============================================"
