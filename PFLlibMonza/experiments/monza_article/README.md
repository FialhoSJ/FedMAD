# Comparação FedMAD × MONZA com parâmetros do artigo

Este conjunto de scripts compara **FedMAD** com o **MONZA do repositório**, usando os parâmetros do artigo *MONZA: A Score System for Malicious Clients Detection* ([JISA, DOI 10.5753/jisa.2026.7098](https://journals-sol.sbc.org.br/index.php/jisa/article/download/7098/4181/42713)).

## Protocolo

- 100 clientes, participação de 100% e 30 clientes maliciosos.
- MNIST, CIFAR-10 e CIFAR-100, com partição Dirichlet não IID `α=0,2`.
- CNN do PFLlib, uma época local, batch 10, SGD `lr=0,005`, decaimento exponencial `γ=0,99`.
- Quatro ataques com chance igual por cliente malicioso em cada rodada: pesos zerados, parâmetros embaralhados, valores aleatórios e inversão cíclica de rótulos. O código começa a aplicar ataques depois da primeira rodada (`-ria 0`).
- Dez seeds pareadas (`0` a `9`) por método e dataset; cada seed escolhe os mesmos 30 IDs maliciosos nos dois métodos. O tipo de ataque e os sorteios de parâmetros/permutação também são fixos por seed, cliente e rodada, com agenda gravada em `metrics.json`.
- Exatamente 150 rodadas, avaliação a cada iteração e intervalo de confiança de 95%. Como o laço do PFLlib é `range(global_rounds + 1)`, o script passa `-gr 149` e produz 150 atualizações e avaliações, indexadas de 0 a 149. Esse horizonte reduzido foi escolhido para esta simulação; o artigo apresenta resultados até 500 rodadas.

O comparador gera curvas até o horizonte disponível de 150 rodadas, além de FPR/FRR por rodada, tempo total das rodadas, média do tempo local por chamada e uma estimativa de MFLOP/s.

## Executar

No diretório `PFLlibMonza`, use o ambiente virtual que já existe no projeto:

```bash
source .venv/bin/activate
python experiments/monza_article/prepare_datasets.py

# Piloto curto para conferir a execução
python experiments/monza_article/run_matrix.py \
  --datasets MONZA_MNIST_A02 \
  --methods fedmad,monza \
  --seeds 42 \
  --rounds 2

# Simulação: FedMAD × MONZA, 3 datasets × 10 seeds, 150 no argumento de rounds
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/monza_article/run_matrix.py \
  --methods fedmad,monza \
  --rounds 150 \
  --malicious 30 \
  --attack-profile article \
  --attack-noise-snr 1.0 \
  --label-flip-source 0 \
  --label-flip-target 1 \
  --device cpu \
  --torch-threads 1 \
  --resume

# Gerar gráficos e tabelas depois que as execuções terminarem
python experiments/monza_article/analyze_results.py \
  --results results/monza_article_replication/rounds_150_article_attacks_batched_eval_seeded_cpu
```

`--device auto` escolhe CUDA se o PyTorch instalado tiver GPU disponível; também aceita `--device cpu` ou `--device cuda`. O nome da pasta termina em `_cpu` ou `_cuda` para manter execuções em dispositivos diferentes separadas. Neste computador, o PyTorch é CPU-only; em uma máquina com CUDA, use `rounds_150_article_attacks_batched_eval_seeded_cuda` no comando de análise.

Os resultados por seed ficam em `results/monza_article_replication/`. O exemplo compara somente FedMAD e MONZA.

O modo padrão usa avaliação em lotes de 256 exemplos para reduzir o custo da avaliação repetida. Ele mantém a acurácia ponderada e a cross-entropy ponderada e não altera os dados ou passos de treino; a AUC adicional do PFLlib é omitida. Validei o avaliador padrão contra o original no mesmo seed: acurácia idêntica, parâmetros do modelo final idênticos e diferença máxima de `5,8×10⁻⁵` na loss por arredondamento float32. Para executar também as rotinas originais por cliente, use `--exact-eval` no `run_matrix.py`. Os embaralhamentos de treino recebem uma seed por cliente e rodada, evitando disputas do gerador global entre as 100 threads e assegurando o pareamento dos métodos.

## Perfil de ataques com base nos nomes da tabela do artigo

O comando principal acima usa este perfil para comparar somente FedMAD e MONZA com as três datasets e dez seeds:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/monza_article/run_matrix.py \
  --methods fedmad,monza \
  --rounds 150 \
  --malicious 30 \
  --attack-profile article \
  --attack-noise-snr 1.0 \
  --label-flip-source 0 \
  --label-flip-target 1 \
  --device auto \
  --torch-threads 1 \
  --resume

python experiments/monza_article/analyze_results.py \
  --results results/monza_article_replication/rounds_150_article_attacks_batched_eval_seeded_cpu
```

Esse perfil zera parâmetros, adiciona ruído Gaussiano por tensor com SNR linear 1, permuta canais de saída por camada Conv/Linear e troca rótulos da classe 0 para a classe 1. A tabela informa os nomes dos ataques, mas não especifica SNR nem classes de origem/destino; esses valores ficam explícitos e configuráveis no comando e em cada `metrics.json`. A permutação de canais é uma interpretação operacional de “Shuffled-Layer”/mudança na ordem de canais. O ramo MONZA do repositório é `-cc 3`, baseado em scores; a opção de clusterização explícita com dois grupos (`-cc 2`) seleciona outro ramo do código, portanto não a aplico como se fosse o MONZA do artigo.

## Limites da comparação

O artigo identifica este repositório como implementação do MONZA. Mesmo assim, há diferenças entre o pseudocódigo publicado e o ramo `-cc 3` atual: o texto descreve a troca para L2Grad quando `σ < 0,001`, enquanto `serverbase.calculate_similarity_scores()` dispara essa alternativa em `σ < 0,01`; o cálculo do código usa quadrados dos parâmetros (com `λ=0,01`), não gradientes calculados. Os resultados gerados aqui reproduzem o MONZA **como implementado no repositório**, e essas diferenças ficam registradas no relatório para que a comparação não pareça uma réplica exata de cada linha do pseudocódigo.

O texto de resultados também descreve ataques “simultâneos”, mas a seção de metodologia e `-atk all` especificam uma escolha aleatória, equiprovável, de um ataque por cliente malicioso a cada rodada. O projeto tinha um erro em `model_zeros`: apesar do nome, enviava tensores preenchidos com 1. Corrigi essa função para enviar zeros, como define o ataque do artigo. No perfil `repository`, as demais rotinas usam ruído uniforme em `[0,1]`, permutação independente dos elementos de cada tensor e rotação cíclica dos rótulos. O perfil `article` usa as interpretações descritas acima. Os parâmetros exatos dos ataques não estão totalmente especificados no artigo. A Tabela 4 e a narrativa têm números diferentes para CIFAR-10; o relatório conserva os valores tabulares como referência e sinaliza a divergência.

O artigo usou CPU i9-13900K, 128 GB RAM e duas RTX 4090. Esta máquina tem um Ryzen 5 5600G, PyTorch CPU-only e nenhum CUDA disponível. O piloto pareado executado com `--rounds 50`, seed 42 e MNIST levou 56,0 min para FedMAD, 29,3 min para FedAvg e 25,8 min para MONZA. Uma extrapolação linear desse custo para a grade completa dá cerca de 23 dias, possivelmente mais com CIFAR e dependendo da carga da máquina. O script grava cada execução separadamente; `--resume` pula execuções concluídas, mas não retoma uma execução interrompida no meio. O tempo de parede absoluto não deve ser comparado entre máquinas.

O MFLOP/s local é estimado contando multiplicações e somas das convoluções e camadas lineares, com três passagens equivalentes para forward/backward e dividido pelo tempo local medido. O artigo publica uma fórmula simplificada cuja unidade declarada não coincide dimensionalmente com o produto descrito; use a estimativa para comparar os métodos selecionados neste mesmo hardware, e não como equivalência direta ao número do artigo.
