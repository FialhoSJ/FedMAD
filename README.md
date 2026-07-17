# FedMAD
FedMAD It is a defense framework against poisoning (poisoning/backdoor) attacks in Federated Learning (FL), based on a collaborative multi-agent architecture with self-supervised learning (SSL).


# FedMAD — Federated Multi-Agent Defense (em desenvolvimento)

Defesa contra ataques de envenenamento em Federated Learning usando arquitetura multi-agente e aprendizado auto-supervisionado.

## Status

Em desenvolvimento. Código baseado no [PFLlib](https://github.com/VeigarGit/PFLlibMonza.git).

## O que foi feito até agora

### Ambiente e dados
- [x] Repositório clonado e configurado (`PFLlibMonza`)
- [x] Ambiente Python com PyTorch (CPU/CUDA), scikit-learn, hdbscan
- [x] Dataset STL10 com splits pré-processados (treino, teste, servidor)
- [x] Encoder pré-treinado SimCLR para STL10 (`fedsimclr_stl10_encoder200.pth`)

### Implementação base (experimentos EmInspector)
- [x] Script `backdoor_fssl.py` funcional com ataque backdoor em FL
- [x] Agregadores implementados: FedAvg e EmInspector
- [x] Defesa EmInspector: detecção por similaridade cosseno no espaço de embeddings
- [x] Fix permanente no `backdoored_dataset.py` (eager load numpy com context manager)

### Experimentos
- [x] FedAvg baseline com STL10 (25 usuários, 2 atacantes, 5 épocas)
- [x] Geração de gráficos BA (acurácia) e ASR (ataque)
- [x] Logs CSV com métricas por época

### PFLlibMonza — base para FedMAD
- [x] Fork do PFLlib com suporte a ataques (label, random, zero, shuffle)
- [x] Cálculo de similaridade cosseno entre clientes
- [x] Clusterização e remoção de outliers
- [x] Métricas FPR/FRR
- [x] Mecanismo de quarentena

## Próximos passos
- [ ] Client MAD com encoder SSL (SimCLR)
- [ ] Agent 1 — EmInspector (similaridade de embedding)
- [ ] Agent 2 — FedREDefense (erro de reconstrução)
- [ ] Agent 3 — Behavior (análise temporal)
- [ ] Aggregator Agent (fusão de scores)
- [ ] Server MAD (orquestração)
- [ ] Experimentos comparativos: FedAvg, EmInspector, FedREDefense, FedMAD

## Como executar

```bash
cd system
python main.py -data Cifar10 -m CNN -algo FedAvg -nc 20 -gr 200
Referências
- EmInspector — detecção por embedding (arXiv)
- FedREDefense — detecção por erro de reconstrução (ICML 2024)
- SimCLR — aprendizado contrastivo (NeurIPS 2020)
- PFLlib — biblioteca FL (JMLR 2025)

---
