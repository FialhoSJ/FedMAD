#!/usr/bin/env bash
# ============================================================
# Gera o dataset MNIST federado (PFLlib format) com 50 clientes
# Uso: bash generate_mnist_dataset.sh
# Requer: python com torch, torchvision, sklearn, ujson
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="$SCRIPT_DIR/tmp_gen"

# Python: prioriza venv_wsl, depois .venv, depois python3
PYTHON="python3"
if [ -d "$SCRIPT_DIR/venv_wsl" ]; then
    PYTHON="$SCRIPT_DIR/venv_wsl/bin/python"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

DOWNLOAD=""
if command -v curl >/dev/null 2>&1; then
    DOWNLOAD="curl -fsSL -o"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD="wget -q -O"
else
    echo "ERRO: precisa de curl ou wget" >&2
    exit 1
fi

BASE_URL="https://raw.githubusercontent.com/TsingZ0/PFLlib/master/dataset"

mkdir -p "$GEN_DIR/utils"
cd "$GEN_DIR"

echo "Baixando generate_MNIST.py..."
$DOWNLOAD generate_MNIST.py "$BASE_URL/generate_MNIST.py"
echo "Baixando utils/dataset_utils.py..."
$DOWNLOAD utils/dataset_utils.py "$BASE_URL/utils/dataset_utils.py"

# Ajusta para 50 clientes (padrao do script e 20)
sed -i 's/num_clients = 20/num_clients = 50/' generate_MNIST.py
grep -n "num_clients = " generate_MNIST.py | head -1

echo "Gerando dataset MNIST (noniid unbalance dir, 50 clientes)..."
"$PYTHON" generate_MNIST.py noniid unbalance dir

# Move para o local esperado pela simulacao
mkdir -p "$SCRIPT_DIR/PFLlibMonza/dataset"
rm -rf "$SCRIPT_DIR/PFLlibMonza/dataset/MNIST"
mv MNIST "$SCRIPT_DIR/PFLlibMonza/dataset/MNIST"
cd "$SCRIPT_DIR"
rm -rf "$GEN_DIR"

# Splits envenenados (label flip) para o ataque 'label' (BA/ASR)
echo "Gerando splits envenenados (train_mal/test_mal, label flip)..."
"$PYTHON" "$SCRIPT_DIR/generate_malicious_split.py" "$SCRIPT_DIR/PFLlibMonza/dataset/MNIST" 10

echo "============================================"
echo " Dataset pronto em: PFLlibMonza/dataset/MNIST"
echo " (inclui train_mal/ e test_mal/ para -atk label)"
echo " Agora rode: bash run_fedmad_test.sh"
echo "============================================"
