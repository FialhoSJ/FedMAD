# FedMAD: auditoria contínua e defesa adaptativa

Estado: plano de migração baseado na inspeção do repositório em 2026-10-05. Este documento precede a implementação da nova arquitetura. O código pré-existente, inclusive alterações locais ainda não commitadas, é tratado como baseline e não como resultado experimental validado.

## Current Architecture

O projeto estende PFLlib em `PFLlibMonza/`. `system/main.py` seleciona `MAD` ou `MADStatic`, define argumentos CLI e seeds. `flcore/servers/servermad.py` herda de `Server` (`serverbase.py`), instancia `ClientMAD` para clientes benignos e `ClientMaliciousAVG` para IDs maliciosos, distribui o modelo, executa treino local em threads, recebe modelos e registra resultados. `ClientMAD` acrescenta um caminho SSL opcional. `serverbase.py` mantém seleção, pesos por amostras, avaliação, quarentena e saída HDF5.

O modo `MAD` atual combina cinco classes em `madsystem/monitoring.py`: `GradientAgent`, `SimilarityAgent`, `StatisticalAgent`, `PerformanceAgent` e `HistoryAgent`. Cada classe recebe os modelos completos e calcula seu próprio sinal; vários recalculam os deltas. `RiskAssessment` faz média dos scores, EMA do risco e define risco da rodada como o **máximo** dos riscos individuais. `RuleBasedMetaAgent` escolhe uma lista por nível LOW/MEDIUM/HIGH. `defenses.py` contém FedAvg, mediana, média aparada, clipping, Krum, Multi-Krum, Bulyan e FoolsGold, mas chama cada estratégia de `DefenseAgent`. `GlobalValidator` testa finitude, norma do delta, loss e accuracy em batches retidos de clientes; o servidor tenta alternativas e restaura o último modelo aceito. `MADStatic` usa uma defesa fixa e não faz a auditoria/validação adaptativa.

`clientmaliciousavg.py` e `attack.py` implementam ou simulam zeros, parâmetros aleatórios, permutações, label flipping e, nas alterações locais atuais, ruído gaussiano e permutação de canais. A comparação `experiments/monza_article/` adiciona preparação de MNIST/CIFAR-10/CIFAR-100, matriz pareada, seeds e análise. `results/` contém saídas; `system/utils/result_utils.py` resume HDF5. Há `test_mad.sh` como execução exploratória, mas nenhum teste Python unitário encontrado. A configuração geral é CLI, com scripts shell e argumentos da matriz experimental; não existe ainda uma configuração declarativa FedMAD em YAML/JSON. Os logs por rodada ficam em JSON e incluem scores, risco, tentativas, tempo e bytes estimados.

## Problems

1. Cinco monitores e oito defesas recebem o nome de agente. A topologia não corresponde à hipótese de dois agentes. Módulos legados em `madsystem/agents/` incluem heurísticas, autoencoder e SLM; não controlam a decisão adaptativa atual e exigem separação explícita de baselines opcionais.
2. O sinal temporal atual guarda sketches de direções e `RiskAssessment` guarda EMA de risco. Faltam um estado compacto único por cliente, EMA das features, reputação com recuperação e uma distinção formal entre anomalia, reputação e risco.
3. A direção usa similaridade média entre clientes, sensível a conluio; magnitude só pontua valores altos. Falta distância a uma referência robusta e detecção bilateral de magnitude. Cada monitor extrai novamente o update completo.
4. O maior risco individual define sozinho o nível da rodada. Isso não distingue outlier isolado, vários outliers e drift populacional. A política usa apenas o nível, não a distribuição de risco nem a reputação.
5. O servidor avalia impactos de performance por cliente antes dos demais sinais, com custo de inferência adicional. A validação usa batches de teste de clientes dentro do simulador; isso pode contaminar a avaliação final se o mesmo conjunto for reportado como teste. É necessário um conjunto de validação distinto para resultados publicáveis.
6. A lista de tentativas de defesa não tem limite explícito configurado, embora seja finita pelo catálogo. Trocas de defesa quando uma estratégia é inviável precisam registrar exatamente o algoritmo executado.
7. O rótulo `malicious_ground_truth` marca IDs designados como maliciosos, mesmo quando o ataque só é ativado em parte das rodadas. Detection TPR/FPR por rodada deve usar o estado efetivo do ataque, além de uma análise separada por ID adversarial.
8. Há mudanças locais preexistentes em arquivos de ataque, cliente, validador, servidor e `main.py`. A migração deve preservar essas mudanças e os experimentos MONZA, sem reivindicar reprodutibilidade já alcançada.

## Proposed Architecture

```text
PFLlib server/client/attack pipeline
  -> Sentinel (único agente observador; roda toda rodada)
  -> Memory Manager (módulo determinístico)
  -> Risk Engine (módulo determinístico)
  -> Meta-Agent (único agente decisor, política por regras)
  -> Defense Pool (funções/estratégias determinísticas)
  -> Global Validator (módulo determinístico)
  -> aceite ou rollback -> logs
```

Não há classificador obrigatório de nomes de ataques. O Sentinel mede incompatibilidade com a população **e** com o histórico próprio. Nenhum módulo de memória, reputação, agregação ou validação é um agente. `MADStatic` e a execução PFLlib `FedAvg` permanecem controles experimentais.

### Relação entre código existente e destino

| Atual | Destino | Ação |
|---|---|---|
| `servermad.py`, `clientmad.py`, `serverbase.py` | orquestração PFLlib | reutilizar o laço e a interface de modelos; reduzir a lógica de política no servidor |
| `_updates`, `_robust_high_scores` em `monitoring.py` | extração/normalização do Sentinel | reutilizar ideias, computar o delta uma vez por rodada; testar zeros, NaN e conluio |
| `HistoryAgent`, `RiskAssessment.profiles` | memória e risco | migrar para estado compacto por ID; descontinuar agentes separados |
| `meta_agent.py` | Meta-Agent | preservar política determinística; passar estado da rodada completo |
| `defenses.py` | Defense Pool | preservar implementações, retirar a semântica de agente; checar pré-condições e equivalência com baselines |
| `global_validator.py` | validação | reutilizar validação e rollback; isolar dados de validação e limitar tentativas |
| `madsystem/agents/*`, `aggregator_agent.py` | `legacy/` ou baselines opcionais | fora do caminho padrão; não instanciar como novos agentes |
| `attack.py`, `clientmaliciousavg.py` | ataques experimentais | reaproveitar os ataques presentes; implementar faltantes como cenários independentes |
| `main.py`, `experiments/monza_article/*` | configuração e experimentos | preservar CLI; introduzir configuração JSON/YAML reproduzível sem quebrar scripts atuais |
| `result_utils.py`, JSON/HDF5 atual | métricas e relatórios | ampliar logs, indicadores de detecção e custo com definições auditáveis |

### Árvore de arquivos alvo, incremental

```text
docs/FEDMAD_REDESIGN.md
PFLlibMonza/system/flcore/madsystem/
  sentinel.py             # Sentinel Agent e quatro features
  memory.py               # ClientState, EMA, reputação
  risk.py                 # risco individual e populacional
  meta_agent.py           # Meta-Agent baseado em regras
  defenses.py             # pool de estratégias, sem agentes
  global_validator.py     # validação e decisão de rollback
  monitoring.py           # compatibilidade temporária / legado
  agents/                 # legado, fora do caminho MAD principal
PFLlibMonza/system/flcore/servers/servermad.py
PFLlibMonza/system/flcore/clients/clientmad.py
PFLlibMonza/system/flcore/attack/attack.py
PFLlibMonza/experiments/fedmad/
  configs/                # arquivos JSON/YAML de cenários
  run_matrix.py
  analyze_results.py
PFLlibMonza/tests/        # testes unitários e de integração curtos
```

Não há motivo para duplicar `serverbase.py`, `clientmad.py` ou cada defesa em uma nova árvore `fedmad/`. Arquivos alvo só serão criados quando houver comportamento implementado. Ataques, datasets e baselines externos adicionais dependerão de protocolo e licença.

## Sentinel Design

Para os modelos recebidos, calcular uma única vez `u_i^t = vec(W_i^t - W_{t-1})`, somente para tensores floating point com shapes compatíveis. A referência populacional inicial é a mediana por coordenada `m^t = median_i(u_i^t)`. A mediana exige memória `O(nd)` para `n` clientes e `d` parâmetros; para modelos grandes, a extração atual já materializa `O(nd)` e deverá ser perfilada antes de optar por chunking. O vetor zero e referências com norma zero exigem uma convenção explícita de cosseno.

Features cruas: `N_i = ||u_i||_2`, `C_i = cos(u_i,m)`, `D_i = ||u_i-m||_2 / (||m||_2+ε)` e `T_i = ||z_i - h_i^{t-1}||_2`, onde `z_i` usa features comparáveis e `h_i` é a EMA anterior do cliente. No primeiro encontro `T_i=0` e o estado é inicializado **depois** do score, impedindo vazamento do update atual para a referência temporal. Para magnitude, pontuar desvios nos dois sentidos em relação à mediana da rodada e à própria EMA. Para direção usar `1-C_i` e para distância e tempo pontuar caudas altas. Normalização robusta: mediana/MAD com fallback documentado quando MAD=0, limites `[0,1]` e saída finita. A anomalia é `A_i = (s_N+s_C+s_D+s_T)/4`, com cada `s` em `[0,1]`. Pesos iguais são ponto inicial, não verdade universal. Perfil, impacto e features por camada ficam fora da versão mínima.

## Memory Design

`ClientState = {norm_ema, cosine_ema, distance_ema, anomaly_ema, reputation, risk, consecutive_suspicious_rounds, total_suspicious_rounds, last_seen_round, observations}`. Não guardar modelos completos por cliente. Para cada feature comparável `x`, `H_i^t = α H_i^{t-1}+(1-α)x_i^t`, com `0<α<1`. Primeiro encontro usa `H_i^t=x_i^t`. A reputação começa em 1 e segue `Rep_i^t = clip(Rep_i^{t-1} - λ_p A_i^t + λ_r(1-A_i^t),0,1)`, com `λ_p>λ_r≥0`. Aplicar limiar de alerta, streak e recuperação gradativa; uma única anomalia não deve gerar quarentena. Testar `α∈{0.5,0.7,0.9,0.95}` e sensibilidades de `λ_p, λ_r`. Ausência em rodadas não altera a EMA; `last_seen_round` permite avaliar lacunas.

## Risk Engine Design

Definir `R_i^t = clip(β_A A_i^t + β_R(1-Rep_i^{t-1}) + β_T s_T,i^t,0,1)`, `β_A+β_R+β_T=1`. Usar a reputação **anterior** evita aplicar a penalidade duas vezes no mesmo update; só depois avançar a memória. `R_round = clip(γ_mean mean_i R_i + γ_max max_i R_i + γ_frac (#{R_i>τ_c}/n),0,1)` com pesos `γ` somando 1. Registrar separadamente `mean`, `max`, `fraction_high` e dispersão para a decisão. Sem clientes, risco 0 e nenhum agregado. Esses scores são índices de evidência, não probabilidades calibradas. Testar thresholds e pesos em validação separada, inclusive Non-IID.

## Meta-Agent Design

Estado: risco individual e da rodada, reputações, fração de suspeitos, dispersão, número de participantes e viabilidade de cada defesa. Política inicial auditável: risco baixo → FedAvg; poucos outliers fortes e Krum viável → Multi-Krum/Krum; muitos desvios de coordenada → média aparada/mediana; suspeita difusa → agregação ponderada por reputação ou clipping; fallback robusto se a primeira candidata falhar. Registrar motivo, ordem de tentativas, algoritmo de fato aplicado e custo. Sem LLM, SLM ou RL no caminho padrão. Decisões e thresholds devem ser congelados antes do teste de ataque não visto.

## Defense Pool

Reaproveitar FedAvg, median, trimmed mean, clipping, Krum e Multi-Krum de `defenses.py` como estratégias comuns. Bulyan e FoolsGold podem permanecer como baselines opcionais, sujeitos a pré-condições e validação do método. Krum/Multi-Krum requerem limite de Byzantine e número suficiente de clientes; nunca rotular uma execução de Krum se o código efetivamente executou FedAvg. `f` é parâmetro/estimativa experimental, não o ground truth fornecido ao método. Verificar buffers não floating point e pesos em cada estratégia. Não copiar código externo sem checar licença.

## Validator

Validar finitude, norma do update, perda/acurácia e, quando houver base adequada, degradação por classe. Comparar sempre com o último modelo aceito. Aceite faz commit atômico de modelo e métricas; rejeição tenta a próxima defesa até `max_attempts` e então restaura `W_safe`. Ausência de dados de validação precisa de estado explícito: somente verificações estruturais podem ser feitas, e o log deve marcar `validation_available=false`. Para artigo, separar validação de teste final e documentar a origem e disponibilidade da amostra confiável.

## Pseudocódigo de uma rodada

```text
selected <- select_clients(seed, round)
send(W_safe); train_local(selected); received <- receive_updates()
if received is empty: log(no_updates); continue
features, anomaly <- Sentinel.inspect(received, W_safe, memory.before_round)
client_risk, round_state <- RiskEngine.assess(anomaly, memory.before_round)
memory.update(features, anomaly, client_risk, round)
defense_order <- MetaAgent.select(round_state, feasible_defenses)
for defense in defense_order[0:max_attempts]:
    candidate <- DefensePool[defense].aggregate(received, W_safe)
    valid, reason, metrics <- Validator.check(candidate, W_safe)
    log_attempt(defense, reason, metrics)
    if valid: W_safe <- candidate; break
else: restore(W_safe); log(rollback)
log_features_risk_reputation_selection_runtime_bytes()
```

## Metrics

Registrar por rodada e por seed: accuracy, loss, ASR (somente para ataque direcionado com conjunto acionado), TPR/FPR/FNR/F1 de detecção com rótulos de ataque efetivamente ativo, tempo total, Sentinel, Meta-Agent, agregação+validação, pico de memória e bytes estimados. Reportar `DefenseActivationRate = rodadas com defesa robusta aceita / rodadas com updates`; também reportar tentativas robustas rejeitadas. `SelectedDefenseDistribution` usa defesa efetivamente aplicada, não a solicitada. `DefenseSelectionAccuracy` só se houver oracle pré-definido justificável. Agregar média ± desvio padrão e intervalos por seed, sem misturar rodadas e seeds como observações independentes.

## Experiments

Antes de novos benchmarks, congelar baseline e registrar hash do código, dados, seed, configuração e hardware. Usar pelo menos CIFAR-10 e FEMNIST quando preparados e auditados; a matriz MONZA existente também suporta MNIST/CIFAR-100 e não deve ser apresentada como FEMNIST. Testar 0%, 10%, 20%, 30% maliciosos, diferentes valores Dirichlet, ataques presentes e ataques adicionais implementados com semântica explícita. Generalização: calibrar em Gaussian, sign flipping e label flipping; congelar parâmetros e testar model replacement não visto. Hoje sign flipping e model replacement ainda não constam como cenários prontos.

Comparar FedAvg, median, trimmed mean, Krum/Multi-Krum, defesa fixa versus seleção adaptativa. FLDetector, CONTRA, FLOW e AdaAggRL só entram na tabela numérica após implementações reproduzidas/licenciadas e protocolo equivalente; até lá são referências, não baselines executados. Ablations: Full, sem History, sem Reputation, sem TemporalDeviation, sem MetaDecision e sem Validator. O par `current` versus `history` é central; medir FPR em Non-IID, TPR/F1 e robust accuracy. Testar mudança de ataque durante o treino, cliente temporariamente anômalo, retorno ao normal e ataque coordenado. RQ1–RQ6 do pedido são respondidas por essas comparações, sem presumir superioridade geral.

## Testing Strategy

Testes unitários determinísticos para features, mediana robusta, zero/NaN, EMA, reputação/recuperação, risco individual/populacional, política, pré-condições da defesa, rejeição e rollback. Teste curto de integração com modelos minúsculos e ataques sintéticos, cobrindo zero adversários, updates iguais, outlier isolado, conluio, heterogeneidade benigna, anomalia transitória e recuperação. Conferir que o modo estático e a comparação MONZA continuam carregando; executar benchmark longo somente após os testes curtos e dados validados.

## Migration Plan

0. Congelar comandos e saídas existentes, distinguir modificações locais e registrar baseline reproduzível.
1. Adicionar Sentinel com quatro features e testes; manter adaptação temporária do formato de log.
2. Introduzir memória compacta, EMA e reputação com teste de recuperação.
3. Separar Risk Engine e distinguir risco individual/populacional.
4. Fazer Meta-Agent consumir o estado completo com política configurável.
5. Adaptar Defense Pool, pré-condições e nomes; retirar `DefenseAgent` do caminho principal.
6. Integrar validador e rollback limitado, mantendo conjunto de validação distinto.
7. Completar métricas, configuração JSON/YAML e logs por rodada.
8. Rodar benchmark pareado com múltiplas seeds.
9. Rodar ablations com mesma agenda de clientes/ataques.
10. Rodar ataque não visto após congelar parâmetros.

Em cada fase, registrar problema, referência, ideia, adaptação FedMAD, arquivos, entrada/saída, complexidade, teste e impacto esperado. Implementação e resultado empírico são coisas distintas: este plano não declara que a hipótese foi confirmada.

## Implementation record (primeira entrega)

As fases 1–7 têm um núcleo funcional, configurações JSON e instrumentação inicial. As fases 8–10 possuem runner e plano, mas a matriz grande **não foi executada**. As funções auxiliares legadas em `monitoring.py` e `agents/` permanecem no repositório para compatibilidade/comparação; o caminho `MAD` instancia somente `FedMADSentinel` e `RuleBasedMetaAgent` como agentes.

| Componente | Problema e referência | Ideia e adaptação FedMAD | Código; entrada → saída | Complexidade; teste; impacto esperado |
|---|---|---|---|---|
| Sentinel | Cinco monitores recalculavam deltas; FLDetector/FLOW/CONTRA inspiram histórico e alinhamento | Mediana coordenada, quatro sinais em um observador; `s_N` usa `abs(log1p(N_i)-log1p(median N))`, `s_C` usa `1-C_i`, `s_D` usa distância relativa, `s_T` usa EMA anterior. Cauda alta: `1-exp(-max(0,(x-median(x))/scale))`, `scale=max(1.4826 MAD,0.1 abs(median),1e-6)` | `sentinel.py`; modelos, IDs, modelo global, memória anterior → features, sinais e anomalia | Tempo `O(nd)` para extrair e mediana, memória `O(nd)`; testes de igualdade, outlier e temporal; espera-se sensibilidade a mudanças sem classificação de ataque |
| Memória/reputação | Histórico espalhado entre `HistoryAgent` e `RiskAssessment`; CONTRA/FLOW inspiram estado e confiança | `ClientState` compacto, EMA de três features e anomalia; penalidade `0.08`, recuperação `0.02`, limiar de anomalia `0.45` como defaults calibráveis | `memory.py`; features/anomalia/risco por ID → estado atualizado | `O(1)` por cliente observado além do estado; teste de EMA, penalidade, recuperação e streak; espera-se menor reação a eventos isolados |
| Risco | Máximo individual era o risco global; AdaAggRL inspira decisão a partir de estado, sem RL | `R_i=0.65A_i+0.20(1-Rep_{i,t-1})+0.15T_i`; `R_round=0.30 mean(R)+0.40 max(R)+0.30 fraction(R>=0.35)`. Fração anômala adicional informa a política. Os coeficientes são hipóteses iniciais | `risk.py`; observações + memória anterior → riscos e componentes populacionais | `O(n)` tempo e espaço; teste outlier versus anomalia disseminada; espera-se separar padrões de risco |
| Meta-Agent/defesas | Política dependia somente do nível; defesas eram chamadas agentes | Regras para outlier isolado, anomalia disseminada e LOW/MEDIUM/HIGH; peso de reputação só fora do caminho LOW; teto de quatro tentativas por default; nome executado corresponde à estratégia usada | `meta_agent.py`, `defenses.py`, `servermad.py`; estado → lista de candidatas e modelo aceito | Seleção `O(n)`, agregadores mantêm seus próprios custos (Krum `O(n²d)`); teste de política e fallback; espera-se menor ativação robusta em rodadas normais |
| Validação/rollback | Aceite de candidato poderia incorporar regressão; validador pré-existente já verificava modelo | Preservado e ligado a tentativas limitadas; ablação pode desativar a checagem para medir sua contribuição | `global_validator.py` reutilizado, `servermad.py` modificado; candidato + último modelo aceito → aceite/rejeição e rollback | Custo proporcional aos batches de validação por tentativa; teste com todas as candidatas rejeitadas; espera-se menor incorporação de modelos degradados |
| Ataques | Faltavam opções do protocolo de generalização | Gaussian existente exposto no cliente; sinal invertido e escala aplicados ao **delta** relativo ao modelo recebido, sem mutar a referência | `attack.py`, `clientmaliciousavg.py`, `main.py`; modelo local + referência + fator → modelo enviado | `O(d)` tempo/memória; teste de sign flip e escala; fornece cenários, não prova generalização |
| Configuração/log/matriz | CLI e JSON por rodada não definiam uma grade com seeds e ablações | JSON base com override CLI, cinco ablações, IDs únicos de execução, manifesto com hash da configuração/revisão git, CSV com média/desvio por seed e avaliação depois da última agregação para todos os métodos | `main.py`, `servermad.py`, `experiments/fedmad/*`; config/grade → logs, HDF5, `final_eval_*.json`, resumo | Custo de I/O `O(rodadas × clientes)`; smoke de uma seed e dry run de 135 combinações; permite comparação pareada sem assumir resultado |

Ablações disponíveis: `--mad_ablate_history` (remove uso do histórico e da reputação na decisão), `--mad_ablate_reputation`, `--mad_ablate_temporal`, `--mad_ablate_meta` (defesa fixa com rollback se rejeitada) e `--mad_ablate_validator`. Como History inclui reputação, a ablação `-History` é mais ampla; reportar essa sobreposição na análise. A reputação continua registrada nas ablações, mas não entra no cálculo quando desativada. Thresholds atuais são defaults de engenharia, ainda não calibrados em validação independente.

### Verificação executada

- `python -m unittest discover -s PFLlibMonza/tests -p 'test_fedmad_redesign.py' -v`: 7 testes passaram.
- Rodada MNIST com 5 clientes, 1 cliente aleatório malicioso e seed 17: Multi-Krum foi rejeitado por perda, Krum foi aceito; o JSON registrou sinal, reputação, motivo, labels de ataque ativo, detecção e ativação. É smoke, não resultado científico.
- Rodada CIFAR-10 com 5 clientes, ruído gaussiano em 1 cliente e seed 17: Multi-Krum foi aceito; houve avaliação pós-agregação e log completo. Continua sendo apenas smoke.
- `python PFLlibMonza/experiments/fedmad/run_matrix.py PFLlibMonza/experiments/fedmad/configs/smoke_grid.json --resume` e `analyze_results.py .../manifest.jsonl`: uma seed de sign flipping e duas atualizações concluíram para FedMAD, Krum fixo e FedAvg. A acurácia pós-agregação foi, respectivamente, 0,8604 / 0,8344 / 0,8752; o F1 de detecção FedMAD foi 0. Com uma única seed e duas atualizações, esses números verificam o pipeline, sem demonstrar vantagem de método. Em um smoke anterior de uma atualização, a acurácia pós-agregação FedMAD foi 0,8236 e o último valor pré-update do HDF5 foi 0,0355; o runner agora avalia todos os métodos após o último agregado. O FedAvg legado lança `ZeroDivisionError` ao resumir tempo com somente uma atualização, então o smoke pareado usa duas.
- `pilot_grid.json --dry-run` produziu 135 comandos (3 ataques × 9 métodos × 5 seeds). Nenhum desses 135 benchmarks foi executado.

### Limitações abertas para publicação

O `model_replacement` disponível agora é um **proxy de escala do update** (`W_global + γΔW_local`), sem objetivo de backdoor ou modelo alvo; não pode ser usado como teste de generalização para um ataque de model replacement completo. FEMNIST ainda não foi preparado. A validação continua usando batches de teste de clientes no simulador, por isso precisa ser separada do teste final antes de qualquer conclusão de acurácia robusta/ASR. ASR, memória de pico, degradação por classe, comunicação observada e baseline externo FLDetector/CONTRA/FLOW/AdaAggRL não são produzidos pela matriz atual. Os metadados registram revisão Git e `git_dirty`; uma reprodução exata de um workspace sujo exige também preservar um snapshot ou patch.

## References

- Zhang et al., [FLDetector](https://arxiv.org/abs/2207.09209), KDD 2022, DOI 10.1145/3534678.3539231: inconsistência temporal como inspiração, sem copiar o algoritmo.
- Liu et al., [FLOW](https://ieeexplore.ieee.org/document/10362974/), IEEE IoT Journal 2024, DOI 10.1109/JIOT.2023.3341811: updates atuais e históricos.
- Awan et al., [CONTRA](https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf), ESORICS 2021, DOI 10.1007/978-3-030-88418-5_22: reputação e alinhamento.
- Wang et al., [AdaAggRL](https://ojs.aaai.org/index.php/AAAI/article/view/34733), AAAI 2025, DOI 10.1609/aaai.v39i24.34733: estado observável e decisão adaptativa; FedMAD não reproduz a política RL/MMD.
- Yan et al., [FedRoLA](https://kdd.org/kdd2024/research-track-papers/), KDD 2024, DOI 10.1145/3637528.3671906: futura análise por camada.
- Han et al., [FedCPA](https://openaccess.thecvf.com/content/ICCV2023/html/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.html), ICCV 2023: futura característica de parâmetros críticos.
- Cao et al., [FLTrust](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/), NDSS 2021: direção confiável somente com dados confiáveis no servidor.
- Nguyen et al., [FLAME](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen), USENIX Security 2022: defesa opcional para backdoor, não implementada aqui.

Contribuição a investigar: integração auditável de monitoramento contínuo, memória temporal compacta, reputação, risco em dois níveis, seleção adaptativa e validação posterior. Nenhuma dessas técnicas isoladas é reivindicada como invenção FedMAD.
