# FedMAD — Adaptive Multi-Agent Defense for Federated Learning

FedMAD is an experimental framework for defending federated learning against model poisoning. It monitors client updates, keeps temporal client profiles, estimates client and round risk, selects defenses with a rule-based meta-agent, and validates each candidate global model before accepting it.

The implementation lives in PFLlibMonza (PFLlibMonza/README.md) and extends PFLlib.

## Adaptive round loop

1. Clients train locally and upload model updates.
2. Five monitoring agents score the updates:
   - **Gradient** — update norms and outliers;
   - **Similarity** — direction agreement between client updates;
   - **Statistical** — robust outliers in layer and update statistics;
   - **Performance** — loss impact on held-out client examples;
   - **History** — change from each client's recent update directions.
3. Risk assessment combines the agent scores with an exponential temporal profile for each client. An unusual update raises risk, while repeated HIGH risk is required before quarantine.
4. The rule-based meta-agent chooses an ordered defense policy:
   - **LOW** — FedAvg;
   - **MEDIUM** — Trimmed Mean, coordinate Median, or update clipping;
   - **HIGH** — Bulyan, Multi-Krum, Krum, FoolsGold, and robust mean candidates.
5. The global validator checks candidate parameters and held-out loss/accuracy. If a candidate fails, the meta-agent tries another defense. If all candidates fail, FedMAD restores the last trusted model.
6. Round decisions and per-agent scores are saved as JSON under PFLlibMonza/results/.

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

Use -mad_agents all to enable all five monitors, or pass a comma-separated subset such as -mad_agents gradient,similarity,history.

## Research comparisons

Run the same dataset, client partition, attack, and random seed for:

- FedAvg as the undefended baseline;
- static defenses with -algo MADStatic -mad_fixed_defense bulyan (replace bulyan with median, trimmed_mean, clipping, krum, multi_krum, or foolsgold);
- MAD for adaptive monitoring, selection, and validation.

For example, run a fixed Trimmed Mean baseline with:

```bash
python main.py -data Cifar10 -m CNN -algo MADStatic -mad_fixed_defense trimmed_mean -nc 20 -gr 200 -nmc 2 -atk random
```

In adaptive MAD mode, the per-round detection log records agent scores, raw anomaly scores, temporally smoothed client risks, risk level, defense attempts, validation outcomes, quarantine, rollback, monitor/aggregation time, and estimated upload/download bytes. Static runs record the fixed defense and communication/time estimates. These logs support analysis of accuracy, attack success, detection quality, and defense cost.

## Validation data

The current simulator builds a fixed validation subset from the first configured clients' held-out test batches. This enables candidate-model acceptance and rollback in experiments. A deployment or privacy-preserving study should replace this simulator validation source with a separately governed public or server-side validation set.

The initial meta-agent is deterministic and rule-based. The prior SLM score combiner remains in the codebase for comparison, but it does not control adaptive defense selection.
