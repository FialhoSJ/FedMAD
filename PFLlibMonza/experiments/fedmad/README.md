# FedMAD redesign experiments

Run from the repository root with a Python environment containing the PFLlib dependencies:

```bash
python PFLlibMonza/experiments/fedmad/run_matrix.py PFLlibMonza/experiments/fedmad/configs/smoke_grid.json --resume
python PFLlibMonza/experiments/fedmad/analyze_results.py PFLlibMonza/results/fedmad_redesign_smoke/manifest.jsonl
```

The smoke grid is one seed, one attack, three methods, and two training updates. It checks the pipeline only. The original PFLlib FedAvg server cannot summarize time after just one update, so the paired smoke uses two. `configs/pilot_grid.json` declares three attacks, nine methods/ablations and five seeds (135 jobs). Inspect it with `--dry-run` before launching it; use `--resume` to skip completed runs with the same configuration hash. Each job writes a console log, a unique detection JSON for MAD/MADStatic, an HDF5 learning curve, and a post-update final evaluation JSON. The analyzer uses the post-update value as final accuracy and reports mean and sample standard deviation across seeds.

`--mad_ablate_history` removes historical features and reputation from the decision, so it is broader than `--mad_ablate_temporal`. `--mad_ablate_meta` uses the configured fixed defense, with validator and rollback still available. `--mad_ablate_validator` accepts the first aggregated candidate. Static baselines use `MADStatic`; FedAvg uses the original PFLlib server.

The matrix does not yet compute attack success rate, peak memory, or a properly isolated validation/test split. The available `model_replacement` option scales a local update; it is a proxy and must not be presented as a full targeted model replacement attack. FEMNIST and external paper baselines require separate preparation. See [the redesign plan](../../../docs/FEDMAD_REDESIGN.md) for protocol and limitations.
