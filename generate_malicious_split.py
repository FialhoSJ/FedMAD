#!/usr/bin/env python
# ============================================================
# Gera os splits envenenados (train_mal/ e test_mal/) para o
# ataque "label" (label flipping) no formato PFLlib.
# Flip usado: y_mal = (y + 1) % num_classes (shift ciclico).
#
# Uso: python generate_malicious_split.py <dataset_dir> [num_classes]
# ============================================================
import os
import sys
import numpy as np


def load_npz(path):
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=True)["data"].tolist()


def save_npz(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, data=data)


def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "PFLlibMonza/dataset/MNIST"
    num_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    for src_name, dst_name in (("train", "train_mal"), ("test", "test_mal")):
        src_dir = os.path.join(dataset_dir, src_name)
        dst_dir = os.path.join(dataset_dir, dst_name)
        if not os.path.isdir(src_dir):
            print(f"[skip] {src_dir} nao existe")
            continue
        files = sorted(
            f for f in os.listdir(src_dir) if f.endswith(".npz")
        )
        total = 0
        for f in files:
            data = load_npz(os.path.join(src_dir, f))
            y = data["y"]
            y_mal = (y + 1) % num_classes
            save_npz(os.path.join(dst_dir, f), {"x": data["x"], "y": y_mal})
            total += len(y_mal)
        print(f"[ok] {src_name} -> {dst_name}: {len(files)} arquivos, {total} amostras")

    # Verificacao rapida
    check_dir = os.path.join(dataset_dir, "train_mal")
    files = sorted(f for f in os.listdir(check_dir) if f.endswith(".npz"))
    if files:
        clean = load_npz(os.path.join(dataset_dir, "train", files[0]))
        mal = load_npz(os.path.join(check_dir, files[0]))
        flipped = int(np.sum((mal["y"] - clean["y"]) % num_classes != 1))
        expected = int(np.sum((mal["y"] == (clean["y"] + 1) % num_classes)))
        print(f"[check] {files[0]}: amostras={len(clean['y'])}, "
              f"flip aplicado em {expected}/{len(clean['y'])} (nao-shift={flipped})")
        assert expected == len(clean["y"]), "verificacao de flip falhou"


if __name__ == "__main__":
    main()
