# Experimentos FedMAD V2

V2 combina contexto populacional e histórico individual para investigar heterogeneidade Non-IID e mudanças suspeitas. O plano e a documentação dos arquivos estão em [FEDMAD_REDESIGN_V2.md](../../../docs/FEDMAD_REDESIGN_V2.md). Configurações antigas usam V1 por padrão; seus resultados foram preservados.

## Verificação rápida

Execute na raiz, com dependências PFLlib/PyTorch e MNIST local em `PFLlibMonza/dataset/MNIST/{train,test}`:

```bash
python -m unittest discover -s PFLlibMonza/tests -p "test_fedmad*.py" -v
python PFLlibMonza/experiments/fedmad/run_v2.py PFLlibMonza/experiments/fedmad/configs/v2_smoke_grid.json --dry-run
python PFLlibMonza/experiments/fedmad/run_v2.py PFLlibMonza/experiments/fedmad/configs/v2_smoke_grid.json --resume
python PFLlibMonza/experiments/fedmad/analyze_results.py PFLlibMonza/results/fedmad_v2_smoke/manifest.jsonl
python PFLlibMonza/experiments/fedmad/plot_results.py PFLlibMonza/results/fedmad_v2_smoke/manifest.jsonl
```

Esse smoke tem 1 seed, 5 clientes, 1.000 exemplos, seis atualizações e sign-flipping iniciado na rodada 4. Compara FedMAD, Krum fixo, FedAvg sem defesa e MONZA. `v2_attack_smoke_grid.json` tem oito atualizações, Dirichlet α=0.1 e cinco ataques, com FedMAD full e population-only. Troque a configuração e o diretório para `fedmad_v2_attack_smoke` nos comandos acima para executá-lo/analisá-lo. Smoke verifica integração; não comprova superioridade científica.

`v2_seed_smoke_grid.json` executa FedMAD/Dirichlet 0.1/sign-flipping com as seeds 1–5, seis atualizações e o mesmo limite de 1.000 exemplos. Seu diretório é `fedmad_v2_seed_smoke`; permite conferir média/SD com cinco execuções independentes.

## Matrizes científicas

| Configuração | Objetivo | Jobs | Seeds | Atualizações por job |
|---|---|---:|---|---:|
| `v2_noniid_grid.json` | IID/Dirichlet 1.0/0.5/0.1, clean/sign, sete métodos | 280 | 1–5 | 150 |
| `v2_poisoning_grid.json` | Cinco ataques, Dirichlet 0.5/0.1, sete métodos | 350 | 1–5 | 150 |
| `v2_ablation_grid.json` | Full e seis ablações, IID/0.1, dois ataques | 140 | 1–5 | 150 |
| `v2_history_grid.json` | EMA 0.5/0.7/0.9/0.95, clean/sign | 40 | 1–5 | 150 |
| `v2_generalization_grid.json` | Calibração, ataques não observados e proxy | 200 | 1–5 | 150 |

Inspecione e execute uma matriz, por exemplo:

```bash
python PFLlibMonza/experiments/fedmad/run_v2.py PFLlibMonza/experiments/fedmad/configs/v2_poisoning_grid.json --dry-run
python PFLlibMonza/experiments/fedmad/run_v2.py PFLlibMonza/experiments/fedmad/configs/v2_poisoning_grid.json --resume
```

São matrizes configuradas; os testes curtos da migração não substituem sua execução. `--resume` pula jobs completos com mesmo hash de configuração/código e arquivos finais presentes. Falhas ficam no manifest e fazem o runner terminar com código diferente de zero.

## Dados e configuração

O runner prepara novos datasets `FEDMAD_V2_` a partir dos NPZ locais. Retira 10% estratificados para validação, reutiliza o particionador IID/Dirichlet e divide cada cliente em treino/teste. IDs das amostras permitem conferir disjunção. `preparation.json` registra parâmetros/seeds/SHA256 da origem, que permanece intacta. Esta preparação redivide o pool local; não preserva o split oficial original do MNIST. Protocolos que exijam esse split devem fornecer os dados correspondentes.

`v2_base.json` contém pesos, EMA, reputação, limiares, janela, meta política, validação e avaliação. Grids variam distribuição, ataque, método, seed e `overrides`. Os métodos compartilham partição e agenda por seed/cliente/rodada. Treino determinístico é sequencial, com um thread PyTorch. O cohort planejado antes da quarentena é registrado; MONZA pode excluir clientes por sua política.

`rounds: 150` significa exatamente 150 atualizações, índices 0–149. No CLI legado, `-gr 149` produz essa contagem. `attack.start_round: 30` inicia no índice 30. Flags CLI explícitas prevalecem sobre JSON. No uso direto de `main.py`, V2 com validator ativo exige `validator.data`, NPZ separado com `x`, `y`; o runner preenche esse caminho. `data_distribution` no main registra metadados; preparar a partição cabe ao preparador/runner.

Label Flipping direcionado usa fonte/alvo configurados e o mesmo número de passos locais que treino benigno em V2. O `all` legado mantém seu mix antigo; as matrizes V2 enumeram os ataques separadamente.

## Ablações e ataques não observados

`history` desativa perfil individual, reputação e componentes temporais do risco: detector population-only. Conserva confirmação por repetição para entrada em DEFENSE; `risk_detection` permite comparar o threshold de risco sem essa espera. `historical_deviation` remove a contribuição temporal e mantém reputação. `population` elimina contribuições populacionais ao score/desvio temporal, mantendo norma/direção relativas ao próprio perfil. `meta` aplica defesa fixa sem filtragem/ponderação adaptativa; `validator` aceita o primeiro candidato.

Generalização usa Gaussian/sign-flipping em `calibration`; scaling e label direcionado em `unseen_test`, com parâmetros congelados. Não há ajuste automático de thresholds. Ajustes manuais devem usar somente calibration e ser salvos antes de observar unseen_test. Model Replacement é o proxy existente de update escalado, equivalente ao scaling, e fica em `proxy_test`: não representa comportamento novo ou backdoor completo. Baselines externos FLDetector/FLOW/CONTRA/AdaAggRL/FedRoLA exigem implementações/protocolos próprios.

## Artefatos e métricas

- Manifest, configuração efetiva e stdout por run; snapshots por hash do código com versões Python/PyTorch/NumPy.
- JSON completo, JSONL incremental e CSV por cliente: evidências, estado, reputação, risco e decisão. FedAvg/MONZA geram CSV de FPR/FNR; `FRR` permanece como nome legado da coluna FNR.
- `evaluation_*.json`: episódios por cliente/ataque, latência e censura. Latência vai do primeiro upload efetivamente atacado à primeira entrada em DEFENSE, ou remoção em MONZA. Clientes sem upload não entram na matriz de confusão da rodada.
- `final_eval_*.json`: modelo global exato após última agregação. `clean_accuracy` usa teste limpo; `robust_accuracy` é essa medida sob ataque, comparada ao cenário clean pareado. ASR existe só para label poisoning direcionado com fonte/alvo; demais ataques têm `null`. AUC não é calculada pelo hook global V2.
- `summary.csv`: média, SD amostral e seeds disponíveis por métrica, separados por origem/distribuição/ataque/método/fase/EMA/código/protocolo. SD vazia para uma seed. Não mistura códigos/protocolos diferentes; `failures.json` lista incompletos.
- `figures/`: PNG/PDF de curvas, acurácia final e detecção/ativação. Curvas V2 usam índices reais das rodadas. Barras de erro exigem múltiplas seeds.

FedAvg e agregadores estáticos não classificam clientes: TPR/FNR de predição sempre negativa não medem sua agregação; compare robust accuracy. No FedMAD, `detection` mede DEFENSE e `risk_detection` mede threshold de suspeita. Ativação robusta inclui filtragem, clipping, alternativas rejeitadas e defesas fixas. Taxa desnecessária usa rodadas sem uploads efetivamente atacados.

Pico de memória é RSS/working set do processo em bytes, incluindo dados/modelos. Tempos são segundos; Sentinel overhead é soma dos tempos Sentinel dividida pela soma dos tempos das rodadas. Memória/risco, meta e validator têm tempos separados. Hardware e concorrência afetam essas medidas; não representam memória GPU ou apenas custo da defesa.

## Caminho legado V1

```bash
python PFLlibMonza/experiments/fedmad/run_matrix.py PFLlibMonza/experiments/fedmad/configs/smoke_grid.json --resume
```

O runner e suas configurações foram preservados. V1 usa validação em batches de teste e não possui todas as métricas/splits V2. Resultados anteriores permanecem identificados como V1.
