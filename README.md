# FedMAD — Federated Multi-Agent Defense 

Defesa contra ataques de envenenamento em Federated Learning usando arquitetura multi-agente.

## Status

Em desenvolvimento. Código baseado no [PFLlib](https://github.com/TsingZ0/PFLlib).

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
- [x] Client MAD com encoder SSL (SimCLR)
- [x] Agent L2Norm — norma L2 do desvio `||W_i - W_global||_2`
- [x] Agent L3Norm — norma L3 do desvio `||W_i - W_global||_3`
- [x] Agent Cosine — 1 - similaridade de cosseno com o modelo global
- [x] Agent Entropy — entropia de Shannon dos pesos
- [x] Aggregator Agent (fusão aritmética de scores — sem SLM)
- [x] Server MAD (orquestração: detecção sem SLM + agregação com SLM)
- [ ] Experimentos comparativos: FedAvg, cada defesa isolada, FedMAD completo

## Como executar

```bash
cd system
python main.py -data Cifar10 -m CNN -algo MAD -nc 20 -gr 200 -slm_e False
```

## Referências

- SLMFORGE — SLMs em FL para cybersecurity (IEEE BigData 2025)
- PFLlib — biblioteca FL (JMLR 2025)
