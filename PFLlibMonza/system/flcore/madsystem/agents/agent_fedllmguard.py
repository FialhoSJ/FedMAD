"""
AgentFedLLMGuard — Agente de deteccao baseado em LLM leve.

Referencia:
    Rezaei, Taheri & Shojafar (2025). "FedLLMGuard: A federated large language
    model for anomaly detection in 5G networks". Computer Networks, 269, 111473.

Adaptacao para FedMAD:
    Usa um encoder transformer leve (Tiny-BERT-style) para analisar a
    representacao estrutural dos model updates de cada cliente e detectar
    comportamentos anomalos (backdoor / envenenamento).

    Diferenca para o artigo original:
    - O artigo usa Tiny-BERT + soft prompt tuning como modelo LOCAL em cada
      cliente para detectar anomalias em trafego de rede 5G.
    - Aqui usamos o mesmo principio (encoder transformer leve + soft prompts)
      como detector SERVER-SIDE que analiza os modelos recebidos.

Arquitetura:
    1. Extrai 8 features estatisticas por camada do modelo (media, std, norma,
       min, max, skew, esparsidade, entropia) — similar a "tokenizacao"
    2. Transformer encoder (2 camadas, 4 heads, 128 hidden) processa a
       sequencia de tokens e produz embeddings semanticos
    3. Soft Prompt Tuning: classifier com prompts treinaveis (encoder congelado)
    4. Score final = fusao da saida do soft prompt + distancia para os demais
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from flcore.madsystem.agents.agent_base import AgentBase


# =============================================================================
# Componentes do FedLLMGuard
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Codificacao positional sinusoidal (Vaswani et al., 2017).
    Permite que o transformer use a ordem da sequencia de tokens.
    """
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TinyBERTEncoder(nn.Module):
    """
    Encoder transformer leve inspirado no Tiny-BERT do artigo FedLLMGuard.
    
    Caracteristicas (mesmo do artigo):
    - 2 camadas transformer
    - 128 dimensoes ocultas
    - 4 heads de atencao
    - GELU activation
    - Batch-first para compatibilidade
    """
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation='gelu',
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return F.normalize(x, dim=1)


class SoftPromptClassifier(nn.Module):
    """
    Classificador com Soft Prompt Tuning (artigo FedLLMGuard, Sec 4.4).
    
    Em vez de fine-tunar o encoder inteiro, apenas N prompts treinaveis
    sao aprendidos. Os prompts sao prependidos a sequencia de entrada
    (como tokens especiais) e o encoder permanece congelado.
    
    Isso reduz o custo de ~100x comparado a fine-tuning completo,
    permitindo execucao eficiente ate em CPU.
    """
    def __init__(self, hidden_dim, n_prompts=10):
        super().__init__()
        self.n_prompts = n_prompts
        self.soft_prompts = nn.Parameter(
            torch.randn(1, n_prompts, hidden_dim) * 0.02
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, encoder_output):
        batch_size = encoder_output.size(0)
        prompts = self.soft_prompts.expand(batch_size, -1, -1)
        x = torch.cat([encoder_output.unsqueeze(1), prompts], dim=1)
        x = x.mean(dim=1)
        return torch.sigmoid(self.classifier(x).squeeze(-1))


# =============================================================================
# Agente FedLLMGuard
# =============================================================================

class AgentFedLLMGuard(AgentBase):
    """
    Detector de anomalias baseado em LLM leve (FedLLMGuard).
    
    Fluxo de deteccao:
        1. Para cada modelo de cliente, extrai features estruturais
           (media, std, norma, min, max, skew, sparsity, entropia por camada)
        2. Converte em sequencia de tokens e passa pelo TinyBERTEncoder
           para obter um embedding semantico do model update
        3. SoftPromptClassifier analisa o embedding e produz score de anomalia
        4. Treina soft prompts online com base no modelo global (referencia)
        5. Score final combina LLM + distancia para os demais clientes

    Hiperparametros (via args):
        flg_prompts (int): numero de soft prompts (default: 10)
        flg_hidden (int): dimensao oculta do transformer (default: 128)
        flg_layers (int): camadas do transformer (default: 2)
        flg_epochs (int): epocas de treino dos prompts (default: 5)
        flg_threshold (float): threshold para anomalia (default: 0.6)
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, name="FedLLMGuard")
        self.device = args.device

        # Hiperparametros (com defaults seguros para CPU)
        self.n_prompts = getattr(args, 'flg_prompts', 10)
        self.hidden_dim = getattr(args, 'flg_hidden', 128)
        self.num_layers = getattr(args, 'flg_layers', 2)
        self.train_epochs = getattr(args, 'flg_epochs', 5)

        # --- Componentes do FedLLMGuard ---
        # Encoder transformer leve (Tiny-BERT style) — congelado
        self.encoder = TinyBERTEncoder(
            input_dim=8,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        self._freeze_encoder()

        # Classificador com soft prompts — apenas isso e treinado
        self.classifier = SoftPromptClassifier(
            hidden_dim=self.hidden_dim,
            n_prompts=self.n_prompts,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(), lr=1e-3, weight_decay=1e-5
        )
        self.prompt_trained = False

    def _freeze_encoder(self):
        """Congela todos os parametros do encoder (soft prompt tuning)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    # ------------------------------------------------------------------
    # Extracao de features estruturais
    # ------------------------------------------------------------------

    def _extract_layer_signature(self, param_tensor, name=""):
        """
        Extrai 8 features estatisticas de um tensor de parametro.

        Retorna: tensor (8,) com [media, std, norma, min, max,
                 skew, sparsity, entropy]

        Isso funciona como uma "tokenizacao" do model update:
        cada camada vira um token com 8 dimensoes.
        """
        p = param_tensor.flatten().float()
        if p.numel() < 2:
            return torch.zeros(8, device=self.device)

        mean = p.mean()
        std = p.std()
        norm = p.norm()
        p_min = p.min()
        p_max = p.max()

        # Skewness (terceiro momento padronizado)
        skew = ((p - mean).pow(3).mean() /
                (std.pow(3) + 1e-8)) if std > 1e-8 else torch.tensor(0.0, device=self.device)

        # Esparsidade: fracao de valores proximos de zero
        sparsity = (p.abs() < 1e-4).float().mean()

        # Entropia aproximada via histograma de 10 bins
        bins = torch.histc(p, bins=10)
        probs = bins / (bins.sum() + 1e-10)
        entropy = -(probs * (probs + 1e-10).log()).sum()

        return torch.stack([mean, std, norm, p_min, p_max,
                            skew, sparsity, entropy])

    def _model_to_token_sequence(self, state_dict):
        """
        Converte state_dict do modelo em sequencia de tokens.
        
        Cada camada com parametros (weight, bias) vira um token
        de 8 dimensoes. A sequencia e limitada a 64 tokens.

        Retorna: tensor (seq_len, 8)
        """
        tokens = []
        for name, param in state_dict.items():
            if param.numel() < 2:
                continue
            sig = self._extract_layer_signature(param, name)
            tokens.append(sig)

        if len(tokens) == 0:
            return torch.zeros((1, 8), device=self.device)

        seq = torch.stack(tokens)

        # Trunca ou faz padding para 64 tokens
        max_len = 64
        if seq.size(0) > max_len:
            indices = torch.linspace(0, seq.size(0) - 1, max_len).long()
            return seq[indices]
        elif seq.size(0) < max_len:
            pad = torch.zeros(max_len - seq.size(0), 8, device=self.device)
            return torch.cat([seq, pad], dim=0)
        return seq

    # ------------------------------------------------------------------
    # Embedding com Tiny-BERT
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_model(self, state_dict):
        """
        Codifica um modelo cliente em embedding semantico.
        
        Pipeline:
            state_dict -> token sequence (64x8) -> TinyBERTEncoder -> embedding (128,)
        """
        tokens = self._model_to_token_sequence(state_dict).to(self.device)
        embedding = self.encoder(tokens.unsqueeze(0))
        return embedding.squeeze(0)

    # ------------------------------------------------------------------
    # Treino dos soft prompts
    # ------------------------------------------------------------------

    def _train_soft_prompts(self, client_embeddings, global_embedding):
        """
        Treina apenas os soft prompts usando o modelo global como
        referencia benigna (pseudo-labeling).

        Soft Prompt Tuning (FedLLMGuard Sec 4.4):
        - Encoder congelado (sem gradientes)
        - Apenas SoftPromptClassifier e treinado
        - Clientes mais proximos do global sao pseudo-rotulados como benignos

        Isso permite adaptacao ao dataset especifico sem custo alto.
        """
        n = client_embeddings.size(0)
        if n < 3:
            self.prompt_trained = True
            return

        # Pseudo-labels: clientes com embedding similar ao global = benigno
        cos_sim = F.cosine_similarity(
            global_embedding.unsqueeze(0), client_embeddings
        )
        threshold = cos_sim.median()
        pseudo_labels = (cos_sim < threshold).float().to(self.device)

        # Garante que ha ambas as classes
        if pseudo_labels.sum() == 0 or pseudo_labels.sum() == n:
            pseudo_labels[0] = 1.0 if pseudo_labels.sum() == 0 else 0.0

        self.classifier.train()
        for _ in range(self.train_epochs):
            self.optimizer.zero_grad()
            logits = self.classifier(client_embeddings)
            loss = F.binary_cross_entropy(logits, pseudo_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.classifier.parameters(), max_norm=1.0
            )
            self.optimizer.step()

        self.classifier.eval()
        self.prompt_trained = True

    # ------------------------------------------------------------------
    # Calculo de score de anomalia
    # ------------------------------------------------------------------

    def _anomaly_score_from_embeddings(self, embedding, all_embeddings):
        """
        Calcula score de anomalia de um cliente baseado em seu embedding.
        
        Combina:
          - Distancia de cosseno para a media dos demais (pond. 0.6)
          - Distancia euclidiana para o vizinho mais proximo (pond. 0.4)
        
        Clientes anomalos tendem a ficar isolados no espaco de embeddings.
        """
        n = all_embeddings.size(0)
        if n < 2:
            return 0.5

        # Mascara para selecionar todos exceto o proprio cliente
        mask = torch.ones(n, dtype=torch.bool, device=all_embeddings.device)
        mask[0] = False  # embedding passado como batch 1, e o primeiro
        others = all_embeddings[mask]

        # Distancia de cosseno para a media dos demais
        other_mean = others.mean(dim=0, keepdim=True)
        cos_dist = 1 - F.cosine_similarity(embedding, other_mean).item()

        # Distancia euclidiana para o vizinho mais proximo (excluindo si mesmo)
        euc = torch.norm(embedding - all_embeddings, dim=1)
        valid = euc > 1e-8
        euc_min = euc[valid].min().item() if valid.any() else 0.0

        # Normaliza distancia euclidiana
        euc_mean = euc[valid].mean().item() if valid.any() else 1.0
        euc_norm = min(euc_min / (euc_mean + 1e-8), 1.0)

        return 0.6 * cos_dist + 0.4 * euc_norm

    # ------------------------------------------------------------------
    # Metodo principal
    # ------------------------------------------------------------------

    def analyze(self, client_models, global_model, metadata):
        """
        Analisa os modelos recebidos e retorna scores de anomalia.

        Args:
            client_models: list[nn.Module] — modelos locais enviados pelos clientes
            global_model: nn.Module — modelo global atual
            metadata: dict — informacoes do round (client_ids, round, etc.)

        Returns:
            list[float] — scores de 0 (benigno) a 1 (malicioso)
        """
        n = len(client_models)
        if n < 2:
            return [0.5] * n

        # Garantir que todos os modelos estao no device correto (GPU)
        client_models = [m.to(self.device) for m in client_models]
        global_model = global_model.to(self.device)

        # 1. Codificar todos os modelos em embeddings
        client_embs = []
        for model in client_models:
            emb = self._encode_model(model.state_dict())
            client_embs.append(emb)
        all_embs = torch.stack(client_embs)

        # 2. Codificar modelo global como referencia benigna
        global_emb = self._encode_model(global_model.state_dict())

        # 3. Treinar soft prompts se necessario (primeiro round apenas)
        if not self.prompt_trained:
            self._train_soft_prompts(all_embs, global_emb)

        # 4. Calcular scores
        scores = []
        with torch.no_grad():
            # Score do LLM (soft prompt classifier)
            llm_raw = self.classifier(all_embs)

            for i in range(n):
                # Componente 1: score do LLM (soft prompt)
                llm_score = llm_raw[i].item()

                # Componente 2: score baseado em distancia
                dist_score = self._anomaly_score_from_embeddings(
                    all_embs[i:i+1], all_embs
                )

                # Fusao ponderada
                combined = 0.5 * llm_score + 0.5 * dist_score
                scores.append(combined)

        return self.normalize_scores(scores)
