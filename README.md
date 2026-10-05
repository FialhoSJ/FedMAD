# FedMAD — Adaptive Multi-Agent Defense for Federated Learning

FedMAD is an experimental framework for defending federated learning against model poisoning. Its current redesign has two active agents: a Sentinel that audits every round and a rule-based Meta-Agent that chooses defenses. Compact memory, reputation, risk assessment, aggregation, and validation are deterministic modules. The design and migration record are in [docs/FEDMAD_REDESIGN.md](docs/FEDMAD_REDESIGN.md).

The implementation lives in PFLlibMonza (PFLlibMonza/README.md) and extends PFLlib.

## Adaptive round loop

1. Clients train locally and upload model updates.
2. The **FedMAD Sentinel** extracts magnitude, direction, population distance, and temporal deviation from client updates, then produces an anomaly score.
3. Compact per-client memory updates EMA features and reputation. The risk engine estimates client and round risk. Repeated suspicious behavior is required before quarantine.
4. The rule-based meta-agent chooses an ordered defense policy:
   - **LOW** — FedAvg;
   - **MEDIUM** — Trimmed Mean, coordinate Median, or update clipping;
   - **HIGH or isolated strong outliers** — Multi-Krum, Krum, trimmed mean, median, and other feasible candidates.
5. The global validator checks candidate parameters and held-out loss/accuracy. If a candidate fails, the meta-agent tries another defense. If all candidates fail, FedMAD restores the last trusted model.
6. Round decisions, Sentinel signals, reputation, risk, active attack labels, and defense attempts are saved as JSON under PFLlibMonza/results/.

## Run FedMAD

From the workspace root:

```bash
cd PFLlibMonza/system
python main.py -data Cifar10 -m CNN -algo MAD -nc 20 -gr 200 -nmc 2 -atk random
```

Risk boundaries, temporal smoothing, defense order, Byzantine estimate, and validation limits can be changed with the -mad_* arguments. For example:

```bash
python main.py -data Cifar10 -m CNN -algo MAD -nc 20 -gr 200 -nmc 2 -mad_byzantine_f 2 -mad_validation_clients 5
```

The former `-mad_agents` setting is deprecated in MAD mode. The Sentinel always runs. `-mad_history_alpha`, `-mad_reputation_penalty`, `-mad_reputation_recovery`, and `-mad_max_defense_attempts` control the compact memory and decision loop.

For a JSON-controlled pilot, run from `PFLlibMonza/system`:

```bash
python main.py --config ../experiments/fedmad/configs/cifar10_pilot.json
```

An explicit CLI flag overrides the matching JSON value. The pilot needs the dataset prepared locally.

To inspect the planned comparison grid without running its 135 jobs:

```bash
python PFLlibMonza/experiments/fedmad/run_matrix.py PFLlibMonza/experiments/fedmad/configs/pilot_grid.json --dry-run
```

The single-run smoke grid is `smoke_grid.json`. The runner writes a manifest and unique logs under `PFLlibMonza/results/`; `analyze_results.py` summarizes completed seeds as mean and standard deviation. It evaluates all methods once after their last aggregation so the reported final accuracy is for the final model.

## Research comparisons

Run the same dataset, client partition, attack, and random seed for:

- FedAvg as the undefended baseline;
- static defenses with -algo MADStatic -mad_fixed_defense bulyan (replace bulyan with median, trimmed_mean, clipping, krum, multi_krum, or foolsgold);
- MAD for adaptive monitoring, selection, and validation.

For example, run a fixed Trimmed Mean baseline with:

```bash
python main.py -data Cifar10 -m CNN -algo MADStatic -mad_fixed_defense trimmed_mean -nc 20 -gr 200 -nmc 2 -atk random
```

In adaptive MAD mode, the per-round log records four Sentinel signals, raw anomaly scores, reputation, client and round risk, decision reason, defense attempts, validation outcomes, quarantine, rollback, active attack ground truth, detection counts, time, and estimated upload/download bytes. Static runs record the fixed defense and communication/time estimates. Long benchmarks and ablations described in the redesign document remain experimental work; no comparative claim is implied by these logs alone.

## Validation data

The current simulator builds a fixed validation subset from the first configured clients' held-out test batches. This enables candidate-model acceptance and rollback in experiments. A deployment or privacy-preserving study should replace this simulator validation source with a separately governed public or server-side validation set.

The initial meta-agent is deterministic and rule-based. The prior SLM score combiner remains in the codebase for comparison, but it does not control adaptive defense selection.
