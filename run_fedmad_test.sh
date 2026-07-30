#!/usr/bin/env bash
# ============================================================
# FedMAD - Script de Teste Baseline
# Uso: bash run_fedmad_test.sh [num_atks] [num_rounds] [num_clientes]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/PFLlibMonza/system"

# Valores padrão (medianos para baseline)
NM=${1:-5}          # numero de clientes maliciosos
GR=${2:-150}        # rounds globais
NC=${3:-50}         # total de clientes

echo "============================================"
echo " FedMAD Test - Baseline"
echo "============================================"
echo " Clientes totais : $NC"
echo " Clientes maliciosos : $NM"
echo " Rounds globais  : $GR"
echo "============================================"

# Ativar ambiente virtual (prioriza venv_wsl, depois .venv)
PYTHON="python"
if [ -d "$SCRIPT_DIR/venv_wsl" ]; then
    PYTHON="$SCRIPT_DIR/venv_wsl/bin/python"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

cd "$SYSTEM_DIR"

$PYTHON main.py \
    -algo MAD \
    -nc "$NC" \
    -nmc "$NM" \
    -gr "$GR" \
    -jr 1.0 \
    -atk all \
    -cc 6 \
    -rfake 1 \
    -data Cifar10 \
    -m CNN \
    -t 1 \
    -ls 1 \
    -dev cuda \
    -did 0 \
    -agent_em True \
    -agent_fedre True \
    -agent_bhv True \
    -agent_flg True \
    -slm_e False

echo ""
echo "============================================"
echo " Teste concluido em $(date)"
echo "============================================"
