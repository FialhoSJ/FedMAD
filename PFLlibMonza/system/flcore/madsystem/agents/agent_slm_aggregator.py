"""
SLMAggregatorAgent - Agregador de MODELO baseado em SLM.

Funcao: decide os PESOS de agregacao (weighted FedAvg) dos modelos dos clientes
com base na analise semantica dos parametros + scores dos detectores.

IMPORTANTE (divisao de responsabilidades):
- DETECCAO/defesa: feita APENAS pelos agentes de defesa classica (sem SLM)
  (L2Norm, L3Norm, Cosine, Entropy) + fusao aritmetica.
  O SLM NAO decide quem e malicioso (evita risco de erro de deteccao).
- AGREGACAO: o SLM analisa os parametros recebidos e gera os pesos de
  agregacao dos modelos que passaram pela deteccao (substitui FedAvg puro).

Referencia: SLMFORGE (Sheikhi, IEEE BigData 2025) - SLMs em FL para cybersecurity.
Modelo recomendado: Phi-3-mini (3.8B) - bom equilibrio entre raciocinio e custo.

Fluxo:
1. Recebe modelos dos clientes + scores dos detectores (ja calculados sem SLM)
2. Extrai "tokens" estatisticos por camada (media, std, norma) de cada modelo
3. Converte para prompt estruturado com contexto do round + historico de pesos
4. SLM gera JSON: [{"client_id": 0, "weight": 0.12, "reason": "..."}]
5. Normaliza pesos para soma = 1 (agregacao convexa)
6. Fallback para FedAvg (pesos proporcionais ao dataset) se SLM falhar
"""

import json
import re
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class SLMAggregatorAgent:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.device = args.device  # auto-detect: cpu ou cuda

        # Configuracoes do SLM (definidas via args em main.py)
        self.model_name = getattr(args, 'slm_model', 'microsoft/Phi-3-mini-4k-instruct')
        self.max_tokens = getattr(args, 'slm_max_tokens', 1024)   # tokens maximos na resposta
        self.rounds_per_call = getattr(args, 'slm_every_n', 1)    # executa SLM a cada N rounds (cache nos restantes)
        self.last_round = -1
        self.cached_weights = None
        self.max_layers = getattr(args, 'slm_max_layers', 8)      # camadas analisadas por modelo

        # Carrega o SLM na inicializacao (pode demorar ~minutos no primeiro uso)
        if getattr(args, 'slm_enabled', True):
            self._load_model()

    def _load_model(self):
        """Carrega o SLM (Phi-3-mini, TinyLlama, etc.) via transformers com auto-detect CPU/GPU."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            print(f"[SLM-AGG] A carregar modelo: {self.model_name} em {self.device}...", flush=True)

            # Carrega config primeiro para corrigir possiveis incompatibilidades
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
                if 'type' not in config.rope_scaling:
                    print("[SLM-AGG] A corrigir rope_scaling (falta 'type')", flush=True)
                    config.rope_scaling['type'] = 'linear'

            gen_kwargs = dict(
                config=config,
                trust_remote_code=True,
                attn_implementation='eager',
            )

            if self.device == "cuda":
                # GPU: 4-bit quantizacao (bitsandbytes) para caber ~8GB VRAM
                import accelerate  # noqa: F401
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                    **gen_kwargs,
                )
            else:
                # CPU: carrega diretamente sem device_map (evita erros com accelerate)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    **gen_kwargs,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            # Garante pad_token para geracao (alguns modelos nao o definem)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("[SLM-AGG] Modelo carregado com sucesso!", flush=True)
        except Exception as e:
            print(f"[SLM-AGG] FALHA ao carregar modelo: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("[SLM-AGG] A usar fallback (FedAvg com pesos do dataset).", flush=True)
            self.model = None
            self.tokenizer = None

    # ------------------------------------------------------------------
    # Extracao de tokens estatisticos dos parametros
    # ------------------------------------------------------------------

    def _extract_layer_stat(self, param_tensor):
        """Resumo estatistico de uma camada: media, std, norma L2 (compacto p/ prompt)."""
        p = param_tensor.detach().flatten().float()
        if p.numel() < 2:
            return "0.0,0.0,0.0"
        return f"{p.mean().item():.4f},{p.std().item():.4f},{p.norm().item():.4f}"

    def _model_to_stats(self, state_dict):
        """Converte state_dict em lista de stats por camada (limitado a max_layers)."""
        stats = []
        for name, param in state_dict.items():
            if param.numel() < 2:
                continue
            stats.append(f"{name.split('.')[0]}={self._extract_layer_stat(param)}")
            if len(stats) >= self.max_layers:
                break
        return " | ".join(stats)

    # ------------------------------------------------------------------
    # Construcao do prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, client_models, client_scores, global_model, metadata):
        """
        Constroi o prompt em formato chat (system/user/assistant).

        O prompt inclui:
        - Sistema curto + exemplo few-shot do JSON esperado
        - Scores dos detectores (sem SLM) por cliente
        - Stats por camada de cada modelo cliente
        - Pesos do round anterior (padrao dataset) como referencia
        """
        round_n = metadata.get("round", 0)
        agent_names = metadata.get("agent_names", ["L2Norm", "L3Norm", "Cosine", "Entropy"])
        n_detectors = len(agent_names)
        default_weights = metadata.get("default_weights", {})
        client_ids = metadata.get("client_ids", list(client_scores.keys()))

        # Few-shot: mostra 1 exemplo para o modelo aprender o formato JSON
        example_json = f'''
Example output:
[{{"client_id": 0, "weight": 0.35, "reason": "low anomaly scores, params close to global model"}},
 {{"client_id": 1, "weight": 0.15, "reason": "moderate anomaly scores, params deviate"}}]
'''

        system_msg = (
            "You are the aggregation controller of a federated learning server.\n"
            "Assign an aggregation weight to each client model for weighted FedAvg.\n"
            f"Detector scores (0=benign, 1=malicious) come from {n_detectors} detectors: "
            f"{', '.join(agent_names)}.\n"
            "Higher weight = more influence in the global model. Weights must be >= 0 "
            "and roughly sum to 1.\n"
            "Output ONLY the JSON array. No preamble, no explanations, no markdown.\n"
            + example_json
        )

        abbrev = {name: name[:2].upper() for name in agent_names}
        user_msg = f"Round {round_n} | clients: {len(client_ids)}\n"
        for cid in client_ids:
            scores = client_scores.get(cid, [])
            parts = [f"{abbrev.get(agent_names[i], f'D{i}')}={s:.4f}" for i, s in enumerate(scores)]
            prev_w = default_weights.get(cid, 0.0)
            stats = metadata.get("client_stats", {}).get(cid, "")
            user_msg += f"Client {cid} | prev_weight={prev_w:.4f} | detectors: {', '.join(parts)} | layers: {stats}\n"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    # ------------------------------------------------------------------
    # Parsing da resposta
    # ------------------------------------------------------------------

    def _parse_response(self, text):
        """
        Extrai o JSON da resposta gerada pelo SLM.

        Estrategias de parsing (por ordem de precisao):
        1. Tenta extrair bloco JSON array [...] completo via regex
        2. Fallback: parse linha a linha para objetos JSON individuais {...}
        3. Se ambos falharem, dicionario vazio -> fallback para pesos do dataset
        """
        results = {}

        # Estrategia 1: bloco JSON array completo
        try:
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for item in parsed:
                    cid = item.get("client_id")
                    if cid is not None:
                        results[cid] = item
        except (json.JSONDecodeError, AttributeError):
            pass

        # Estrategia 2: parse linha a linha (SLM pode gerar JSON fragmentado)
        if not results:
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        item = json.loads(line)
                        cid = item.get("client_id")
                        if cid is not None:
                            results[cid] = item
                    except json.JSONDecodeError:
                        continue

        return results

    # ------------------------------------------------------------------
    # Metodo principal: gera pesos de agregacao
    # ------------------------------------------------------------------

    def aggregate_weights(self, client_models, client_scores, metadata=None):
        """
        Metodo principal do agregador SLM.

        Entrada:
            client_models: list[nn.Module] — modelos locais (JA filtrados pela deteccao)
            client_scores: dict {client_id: [score_emInspector, score_fedRE, ...]}
            metadata: dict com round, client_ids, default_weights, history, etc.

        Saida:
            dict {client_id: weight} (soma ~1, pesos >= 0)

        Fluxo:
        1. Se SLM nao carregou -> fallback pesos do dataset (FedAvg)
        2. Cache: se round % N != 0, reusa pesos do ultimo round SLM
        3. Extrai stats dos modelos -> prompt -> SLM infere -> parse JSON
        4. Clientes sem resposta do SLM usam o peso padrao (dataset)
        5. Normaliza pesos para soma = 1
        """
        if metadata is None:
            metadata = {}

        client_ids = metadata.get("client_ids", list(client_scores.keys()))

        # Fallback global: SLM nao carregou (erro no __init__)
        if self.model is None:
            print("[SLM-AGG] Modelo nao disponivel. Fallback: FedAvg (dataset weights).", flush=True)
            return self._fallback_weights(client_ids, metadata)

        round_n = metadata.get("round", 0)

        # Cache: SLM executa apenas a cada N rounds (poupa latencia/token)
        if round_n > 0 and round_n % self.rounds_per_call != 0 and self.cached_weights is not None:
            return self._reuse_cached_weights(client_ids)

        # Extrai stats dos modelos (tokens semanticos para o prompt)
        client_stats = {}
        for cid, model in zip(client_ids, client_models):
            client_stats[cid] = self._model_to_stats(model.state_dict())
        metadata = dict(metadata)
        metadata["client_stats"] = client_stats

        # Constroi o prompt com stats + scores + pesos anteriores
        prompt = self._build_prompt(client_models, client_scores, None, metadata)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            print(f"[SLM-AGG] A inferir pesos para round {round_n} ({len(client_ids)} clients)...", flush=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,  # greedy decoding (mais consistente)
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Descodifica apenas os tokens gerados (ignora o prompt original)
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"[SLM-AGG] Raw response (round {round_n}, {len(response)} chars):\n{response[:600]}", flush=True)
            parsed = self._parse_response(response)
            print(f"[SLM-AGG] Parsed {len(parsed)}/{len(client_ids)} clients", flush=True)

            # Constroi dicionario final: peso do SLM ou peso padrao (dataset)
            default_weights = metadata.get("default_weights", {})
            final_weights = {}
            for cid in client_ids:
                if cid in parsed:
                    w = parsed[cid].get("weight")
                    try:
                        w = float(w)
                    except (TypeError, ValueError):
                        w = default_weights.get(cid, 1.0)
                    final_weights[cid] = max(w, 0.0)  # pesos negativos -> 0
                    reason = parsed[cid].get("reason", "")
                    print(f"[SLM-AGG] Client {cid} | weight={final_weights[cid]:.4f} | razao: {reason}", flush=True)
                else:
                    final_weights[cid] = default_weights.get(cid, 1.0)

            # Normaliza para soma = 1 (se houver pelo menos um peso > 0)
            total = sum(final_weights.values())
            if total > 1e-8:
                final_weights = {cid: w / total for cid, w in final_weights.items()}
            else:
                final_weights = self._fallback_weights(client_ids, metadata)

            # Atualiza cache
            self.cached_weights = final_weights
            self.last_round = round_n
            return final_weights

        except Exception as e:
            print(f"[SLM-AGG] ERRO na inferencia: {e}", flush=True)
            print("[SLM-AGG] A usar fallback (pesos do dataset).", flush=True)
            return self._fallback_weights(client_ids, metadata)

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_weights(self, client_ids, metadata):
        """Fallback: pesos proporcionais ao tamanho do dataset (FedAvg original)."""
        default_weights = metadata.get("default_weights", {})
        if default_weights:
            total = sum(default_weights.get(cid, 0.0) for cid in client_ids)
            if total > 1e-8:
                return {cid: default_weights.get(cid, 0.0) / total for cid in client_ids}
        # Sem informacao: pesos uniformes
        n = len(client_ids)
        return {cid: 1.0 / n for cid in client_ids} if n > 0 else {}

    def _reuse_cached_weights(self, client_ids):
        """Reusa pesos cacheados para os clientes atuais (novos usam uniforme)."""
        if self.cached_weights is None:
            return self._fallback_weights(client_ids, {})
        n = len(client_ids)
        result = {}
        for cid in client_ids:
            if cid in self.cached_weights:
                result[cid] = self.cached_weights[cid]
            else:
                result[cid] = 1.0 / n if n > 0 else 0.0
        total = sum(result.values())
        if total > 1e-8:
            result = {cid: w / total for cid, w in result.items()}
        return result
