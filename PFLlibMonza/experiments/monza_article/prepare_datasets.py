#!/usr/bin/env python3
"""Prepare isolated 100-client, Dirichlet alpha=0.2 datasets for the MONZA protocol."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO / "dataset"
ALPHA = 0.2
CLIENTS = 100
SPECS = (
    ("MNIST", "MONZA_MNIST_A02", "generate_MNIST"),
    ("Cifar10", "MONZA_Cifar10_A02", "generate_Cifar10"),
    ("Cifar100", "MONZA_Cifar100_A02", "generate_Cifar100"),
)


def main() -> None:
    sys.path.insert(0, str(DATASET_DIR))
    from utils import dataset_utils

    dataset_utils.alpha = ALPHA
    manifest = {
        "source": "MONZA article replication",
        "num_clients": CLIENTS,
        "dirichlet_alpha": ALPHA,
        "non_iid": True,
        "balanced": False,
        "partition": "dir",
        "datasets": {},
    }

    for source_name, experiment_name, module_name in SPECS:
        output_dir = DATASET_DIR / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        source_raw = DATASET_DIR / source_name / "rawdata"
        target_raw = output_dir / "rawdata"
        if source_raw.is_dir() and not target_raw.exists():
            shutil.copytree(source_raw, target_raw)

        generator = importlib.import_module(module_name)
        generator.generate_dataset(
            str(output_dir) + "/", CLIENTS, True, False, "dir"
        )
        config_path = output_dir / "config.json"
        with config_path.open(encoding="utf-8") as source:
            config = json.load(source)
        expected = {
            "num_clients": CLIENTS,
            "non_iid": True,
            "partition": "dir",
            "alpha": ALPHA,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"{experiment_name}: config {key}={config.get(key)!r}, "
                    f"expected {value!r}"
                )
        manifest["datasets"][experiment_name] = {
            "path": str(output_dir.relative_to(REPO)),
            "config": str(config_path.relative_to(REPO)),
        }
        print(f"Prepared {experiment_name}: {config_path}", flush=True)

    manifest_path = Path(__file__).with_name("dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
