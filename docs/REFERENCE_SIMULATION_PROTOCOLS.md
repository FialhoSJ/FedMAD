# Protocolos experimentais das referências FedMAD

Extração dos experimentos descritos nas oito referências do prompt. A tabela resume o que cada artigo efetivamente varia; campos que não aparecem no texto acessível são marcados como não especificados, em vez de serem preenchidos por suposição. Resultados publicados não são diretamente comparáveis entre artigos quando mudam dados, ataques, partição, arquitetura ou participação.

## Simulações publicadas

| Trabalho | Dados/modelo | Clientes, partição e participação | Ataques avaliados | Métricas e nota para reprodução |
|---|---|---|---|---|
| **FLDetector (KDD 2022)** | MNIST, CIFAR-10 e FEMNIST; CNN/VGG conforme conjunto/experimento | 28, 28 e 84 clientes maliciosos, respectivamente; varia heterogeneidade Dirichlet `α=0.1…0.9`; detector começa na rodada 50 e usa janela temporal de 10 | Fang untargeted model poisoning, Scaling, DBA/backdoor distribuído e ALIE | Acurácia, ASR e métricas de detecção (DACC/FPR/FNR); compara FedAvg, Krum, Trimmed Mean, Median e variantes de detecção. A configuração padrão e o momento de início do detector importam para a reprodução. |
| **FLOW (IEEE IoT J. 2024)** | Benchmarks de FL; detalhes de datasets/modelos devem ser conferidos no texto integral | usa atualização atual e histórica; cenários com dados imprevisíveis/non-IID; população e participação não confirmadas na página pública do resumo | Ataques untargeted, targeted e adaptativos | Taxa de defesa e acurácia global; compara defesas Byzantine-robust. O resumo confirma cosine distance entre updates e penalização gradual de suspeitos. O acesso público consultado não expõe tabela suficiente para fechar os valores de cada cenário. |
| **CONTRA (ESORICS 2021)** | MNIST, CIFAR-10 e Loan | população de 100 clientes; maliciosos `m=5,10,20,33,50%`; Dirichlet `α∈{0.05,1,10,100,1000}`. Testa `E∈{1,5}` épocas locais e participação `C∈{0.1,0.2,0.4}` (10/20/40 participantes por rodada) | Label flipping, backdoor e ataques coordenados/sybil | Main-task accuracy e ASR, com comparação principal a FoolsGold. Tabelas reportam convergência. A descrição do artigo implementa reputação que afeta recrutamento de clientes. |
| **AdaAggRL (AAAI 2025)** | MNIST, Fashion-MNIST, EMNIST e CIFAR-10; CNN e ResNet-18 | distribuição por grupos de classe parametrizada por `q`; em MNIST, `q=0.5` heterogêneo e `q=0.1` uniforme. Curvas chegam a 500 épocas/rodadas de comunicação | Explicit Boosting (EB), Inner Product Manipulation (IPM), Local Model Poisoning (LMP) e ataque aprendido por RL | Acurácia de teste; compara Clipping, C-Median, FLTrust, Median e Krum. Para CIFAR-10 também compara FedDefender/FedVal. `q` é uma partição por grupos, não Dirichlet; não deve ser tratado como `α` equivalente. |
| **FedRoLA (KDD 2024)** | Implementação dos autores tem exemplo executável em Fashion-MNIST/AlexNet; configuração de exemplo: 128 clientes, 32 participantes, 32 controlados/maliciosos, 3 épocas locais, `α=0.5`, 200 iterações | não-IID (`isIID=False`); 25% de participação por rodada no exemplo | exemplo usa MinMax com conhecimento completo do ataque; código também prevê label attack, MinSum, MPHM e SimAttack | Agrega por camada usando updates atuais e parâmetros globais, sem histórico nem validação. A configuração acima é a do exemplo público do repositório e não deve ser confundida com todas as tabelas do artigo. |
| **FedCPA (ICCV 2023)** | CIFAR-10, SVHN e TinyImageNet no suplemento; ResNet-18 default | principal implementação dos autores mostra CIFAR-10 com 20 clientes e attacker ratio 20%; varia número de maliciosos/participantes e non-IID Dirichlet `β` | Ataque direcionado com trigger 5×5 no canto inferior direito; label flipping e Gaussian untargeted (σ=0.05); taxas de poisoning variam | Acurácia/robustez e custo por rodada; compara sem defesa, Median, Trimmed Mean, Multi-Krum, FoolsGold, Norm Bound, RFA e ResidualBase. Os resultados são média dos últimos 10 rounds, com média e desvio padrão. |
| **FLTrust (NDSS 2021)** | Seis tarefas de domínios diversos, incluindo MNIST, CIFAR-10, Fashion-MNIST, HAR, CH-MNIST e Reddit | experimentos incluem 100 clientes e fração maliciosa de 20%; varia população (50–400), percentagem maliciosa, viés do root set e cenários de dados iid/non-IID; exige pequeno conjunto limpo no servidor | Label flipping, Krum/Trim attacks, scaling/model replacement e adaptativos; backdoor por padrão/target | Erro de teste, ASR e impacto do viés/tamanho do root set. O root dataset e o update de referência limpo fazem parte do método e precisam ser idênticos na comparação. |
| **FLAME (USENIX Security 2022)** | CIFAR-10 (ResNet-18 leve), MNIST, Tiny-ImageNet, Reddit e tráfego IoT | CIFAR-10: 100 clientes, PMR 20%; também mede variação de PMR e participação dinâmica; estuda grau de non-IID | Backdoors Constrain-and-Scale, DBA, Edge-Case, PGD; inclui cenário untargeted adicional | Main-task accuracy (MA), backdoor accuracy (BA/ASR), TPR/TNR e runtime; usa clustering + clipping + ruído. No CIFAR-10, o cenário principal de imagem usa ResNet-18 leve. É uma defesa voltada a backdoors. |

## Cenário comum recomendado

Para comparar métodos em condições idênticas, recomendo começar por **CIFAR-10, 100 clientes, 20% maliciosos, Dirichlet `α=0.2`, participação de 20 clientes por rodada (`C=0.2`), uma época local e sign-flipping aplicado ao update em todas as rodadas**. CIFAR-10 aparece em quase todas as referências, `20%` coincide com configurações explícitas de FLTrust/FLAME, e o `α=0.2` está disponível como partição preparada no repositório. A participação parcial reduz o custo e aproxima `C=0.2` do protocolo CONTRA.

Use seeds pareadas `1…5` para a comparação final; mantenha o conjunto de clientes maliciosos fixo dentro de cada seed e compartilhe a mesma agenda de participação/ataque entre métodos. Registre a configuração e os IDs sorteados. Métricas primárias: acurácia limpa por rodada e no final, queda de acurácia contra FedAvg sem ataque, TPR/FPR/F1 de detecção quando o método produz decisões por cliente, e tempo por rodada. Sign-flipping não é backdoor: ASR não se aplica. Para estudar ataques direcionados, execute depois um segundo experimento separado de backdoor com trigger e label-alvo fixos e reporte ASR + acurácia limpa.

### Piloto executado neste repositório

O cenário está em `PFLlibMonza/experiments/fedmad/configs/cifar10_common_scenario_10r.json` e a matriz em `.../cifar10_common_scenario_10r_grid.json`. Executei quatro métodos (FedMAD adaptativo, Krum fixo, Trimmed Mean fixo e FedAvg), uma seed e **10 atualizações de treino** em CPU. Saída CSV e figuras estão em `PFLlibMonza/results/cifar10_common_scenario_10r_seed1/`. Na seed 1, a acurácia pós-treino foi FedMAD 0,266; Trimmed Mean 0,247; Krum 0,231; FedAvg 0,216. FedMAD teve F1 0,408, TPR 0,313 e FPR 0,043. As curvas de avaliação pré-atualização oscilam e terminam entre 0,135 e 0,171; a acurácia pós-atualização é avaliação extra do modelo final. Uma seed e 10 rounds são um piloto para comparar o pipeline e validar os gráficos, não bastam para afirmar superioridade. O desempenho baixo também indica que é necessário treinar por horizonte maior antes de interpretar acurácia.

Comandos para repetir e gerar os arquivos:

```powershell
python experiments/fedmad/run_matrix.py experiments/fedmad/configs/cifar10_common_scenario_10r_grid.json --resume
python experiments/fedmad/analyze_results.py results/cifar10_common_scenario_10r_seed1/manifest.jsonl
python experiments/fedmad/plot_results.py results/cifar10_common_scenario_10r_seed1/manifest.jsonl
```

### Limite de implementação atual

Este repositório tem FedMAD, FedAvg e defesas fixas Krum/Trimmed Mean/Median/Clipping no Defense Pool. Não contém implementações dos artigos FLDetector, FLOW, CONTRA, AdaAggRL, FedRoLA, FedCPA, FLTrust ou FLAME como baselines reproduzidos. Portanto, a comparação imediatamente executável é **FedMAD vs FedAvg vs agregadores fixos disponíveis** no mesmo simulador; incorporar os oito métodos requer portar/reimplementar cada método e validar seu protocolo. Os resultados do smoke pré-existente em MNIST (5 clientes, 2 atualizações, uma seed) só validam o pipeline e não sustentam comparação científica.

## Fontes primárias

- [FLDetector (PDF dos autores)](https://zaixizhang.github.io/ZaixiZhang_files/FLDetector.pdf)
- [FLOW (IEEE Xplore)](https://ieeexplore.ieee.org/document/10362974/)
- [CONTRA (PDF dos autores)](https://cae.ittc.ku.edu/papers/Awan2021ESORICS.pdf)
- [AdaAggRL (AAAI Open Access)](https://ojs.aaai.org/index.php/AAAI/article/download/34733/36888)
- [FedRoLA (repositório dos autores)](https://github.com/GYan58/KDD-2024-FedRoLA)
- [FedCPA (ICCV Open Access)](https://openaccess.thecvf.com/content/ICCV2023/html/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.html)
- [FLTrust (NDSS Open Access)](https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/)
- [FLAME (USENIX Open Access)](https://www.usenix.org/system/files/sec22-nguyen.pdf)
