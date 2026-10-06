"""Prepare isolated IID/Dirichlet splits from locally available PFLlib NPZ data."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[2]


def prepared_name(source, clients, distribution, seed, max_examples=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source):
        raise ValueError("source must be a dataset name")
    partition = "IID" if distribution == "iid" else "A" + str(float(distribution)).replace(".", "p")
    suffix = f"_sample{max_examples}" if max_examples else ""
    return f"FEDMAD_V2_{source}_{partition}_{clients}c_seed{seed}{suffix}"


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_dataset(source, clients, distribution, seed, max_examples=None,
                    validation_fraction=0.1, batch_size=10, dataset_root=None):
    root = Path(dataset_root) if dataset_root else PROJECT / "dataset"
    name = prepared_name(source, clients, distribution, seed, max_examples)
    destination = root / name
    source_dir = root / source
    files = [path for split in ("train", "test") for path in sorted((source_dir / split).glob("*.npz"), key=lambda p: int(p.stem))]
    if not files:
        raise FileNotFoundError(f"prepare source dataset locally first: {source_dir}")
    if clients < 2 or not 0 < validation_fraction < 0.5:
        raise ValueError("require clients >= 2 and 0 < validation_fraction < 0.5")
    if distribution != "iid" and float(distribution) <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    specification = {"source": source, "clients": clients, "distribution": distribution,
                     "seed": seed, "max_examples": max_examples, "validation_fraction": validation_fraction,
                     "batch_size": batch_size, "source_sha256": {str(path.relative_to(root)): _digest(path) for path in files}}
    manifest_path = destination / "preparation.json"
    if destination.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8"))["specification"] != specification:
            raise ValueError(f"existing dataset has a different preparation; choose another name: {destination}")
        expected = [destination / "validation.npz"] + [destination / split / f"{cid}.npz" for split in ("train", "test") for cid in range(clients)]
        if not all(path.is_file() for path in expected):
            raise ValueError(f"incomplete prepared dataset: {destination}")
        return name, destination / "validation.npz"

    counts = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            counts.append(len(data["data"].item()["y"]))
    total = sum(counts)
    rng = np.random.default_rng(seed)
    selected_ids = np.sort(rng.choice(total, min(max_examples or total, total), replace=False))
    pieces_x, pieces_y = [], []
    offset = 0
    for path, count in zip(files, counts):
        ids = selected_ids[(selected_ids >= offset) & (selected_ids < offset + count)] - offset
        if len(ids):
            with np.load(path, allow_pickle=True) as archive:
                data = archive["data"].item()
                pieces_x.append(np.asarray(data["x"])[ids])
                pieces_y.append(np.asarray(data["y"])[ids])
        offset += count
    x, y = np.concatenate(pieces_x), np.concatenate(pieces_y)
    labels = np.unique(y)
    local_ids = np.arange(len(y))
    train_test_ids, validation_ids = train_test_split(local_ids, test_size=validation_fraction, stratify=y, random_state=seed)
    if len(train_test_ids) < clients * max(2, 2 * batch_size):
        raise ValueError("not enough examples for client training and held-out tests")

    spec = importlib.util.spec_from_file_location("fedmad_dataset_utils", PROJECT / "dataset/utils/dataset_utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    utils.alpha = float(distribution) if distribution != "iid" else 1.0
    utils.batch_size = batch_size
    np.random.seed(seed)
    # The established partitioner works on IDs; no duplicate image pool is needed.
    with contextlib.redirect_stdout(io.StringIO()):
        xx, yy, statistics = utils.separate_data((train_test_ids[:, None], y[train_test_ids]), clients, len(labels),
                                                niid=distribution != "iid", balance=True, partition="dir" if distribution != "iid" else "pat", class_per_client=len(labels))
        train, test = utils.split_data(xx, yy)
    for split, partitions in (("train", train), ("test", test)):
        (destination / split).mkdir(parents=True, exist_ok=True)
        for cid, data in enumerate(partitions):
            ids = np.asarray(data["x"]).reshape(-1).astype(int)
            np.savez_compressed(destination / split / f"{cid}.npz", data={"x": x[ids], "y": y[ids]}, sample_ids=selected_ids[ids])
    np.savez_compressed(destination / "validation.npz", x=x[validation_ids], y=y[validation_ids], sample_ids=selected_ids[validation_ids])
    config = {"num_clients": clients, "num_classes": int(len(labels)), "non_iid": distribution != "iid",
              "balance": True, "partition": "dir" if distribution != "iid" else "pat", "alpha": utils.alpha,
              "batch_size": batch_size, "Size of samples for labels in clients": statistics}
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {"specification": specification, "dataset": name, "examples": len(y),
                "validation_examples": len(validation_ids), "validation_disjoint_from_train_test": True,
                "sample_ids": "original pooled source-file row IDs; present in every NPZ for disjointness checks"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return name, destination / "validation.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="MNIST")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--distributions", default="iid,1.0,0.5,0.1")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for seed in map(int, args.seeds.split(",")):
        for value in args.distributions.split(","):
            distribution = "iid" if value == "iid" else float(value)
            if args.dry_run:
                print(prepared_name(args.source, args.clients, distribution, seed, args.max_examples))
            else:
                print(prepare_dataset(args.source, args.clients, distribution, seed, args.max_examples)[0])


if __name__ == "__main__":
    main()
