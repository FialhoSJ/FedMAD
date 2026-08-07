"""
SLMAggregatorAgent - Agregador de MODELO baseado em SLM.

Funcao: decide os PESOS de agregacao (weighted FedAvg) dos modelos dos clientes
com base na analise semantica dos parametros + scores dos detectores.

IMPORTANTE (divisao de responsabilidades):
- DETECCAO/defesa: feita APENAS pelos agentes de defesa classica (sem SLM)
  (Norm (L2+L3), Cosine, Entropy) + fusao aritmetica.
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
                from transformers import BitsAndBytesConfig
                import accelerate  # noqa: F401

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quant_config,
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
    # Construcao do prompt (system prompt estruturado + few-shot)
    # ------------------------------------------------------------------

    @staticmethod
    def _fewshot_examples():
        """Exemplos few-shot para o SLM aprender o contrato de saida."""
        return (
            "<example_1>\n"
            "user: Round 1 | clients: 3\n"
            'Client 0 | prev_weight=0.33 | detectors: NM=0.10, CO=0.12, EN=0.15 | layers: conv1=0.0,0.01,1.2 | conv2=0.0,0.02,3.1\n'
            'Client 1 | prev_weight=0.33 | detectors: NM=0.08, CO=0.10, EN=0.12 | layers: conv1=0.0,0.01,1.1 | conv2=0.0,0.02,2.9\n'
            'Client 2 | prev_weight=0.34 | detectors: NM=0.09, CO=0.11, EN=0.13 | layers: conv1=0.0,0.01,1.3 | conv2=0.0,0.02,3.0\n'
            "assistant: [{\"client_id\": 1, \"weight\": 0.35, \"reason\": \"lowest anomaly scores, params near global\"},\n"
            '           {"client_id": 2, "weight": 0.33, "reason": "low scores, similar to peers"},\n'
            '           {"client_id": 3, "weight": 0.32, "reason": "low scores, negligible deviation"}]\n'
            "</example_1>\n"
            "<example_2>\n"
            "Client: Round 5 | clients: 2\n"
            'Client 7 | prev_weight: 0.50 | detectors: NM=0.95, CO=0.88, EN=0.90 | layers: conv1=0.0,9.9,99.0 | conv2=0.0,8.8,88.0\n'
            'Client 8 | prev_weight: 0.50 | detectors: NM=0.10, CO=0.12, EN=0.11 | layers: conv1=0.0,0.01,1.1 | conv2=0.0,0.02,2.9\n'
            "{\"client_id\": 7, \"weight\": 0.30, \"reason\": \"high anomaly scores\"},\n"
            '         {"client_id": 8, "weight": 0.70, "reason": "low anomaly scores, consistent params"}]\n'
            "</example_2>\n"
        )

    def _build_system_msg(self, agent_names, n_detectors):
        """
        System prompt estruturado (role/task/rules/output_format).
        Instrucao opcional de chain-of-thought curto quando args.slm_cot.
        """
        cot_rule = (
            "Before the JSON, output a single line starting with 'Reasoning:' "
            "summarizing which clients look most/least reliable. Then the JSON array."
            if getattr(self.args, 'slm_cot', False)
            else ""
        )
        return (
            "<role>\n"
            "You are the Aggregation Weight Controller of a federated learning server\n"
            "that uses weighted FedAvg aggregation.\n"
            "Your ONLY job: assign aggregation weights to the client models residual after\n"
            f"a defense stage handled by {n_detectors} detector agents ({', '.join(agent_names)}).\n"
            "You do NOT decide who is malicious: the defense agents already did that.\n"
            "</role>\n\n"
            "<task>\n"
            "For each client listed in the user message, assign an aggregation weight in [0,1].\n"
            "The weight expresses how much each client model contributes to the global model.\n"
            "</task>\n\n"
            "<rules>\n"
            "- R1: weights must be >= 0 and must sum to approximately 1.0.\n"
            "- R2: a client with LOW anomaly scores should receive a HIGHER weight than one\n"
            "     with HIGH anomaly scores.\n"
            "- R3: if all scores are low and similar, keep weights close to uniform.\n"
            "     if one client stands out with higher scores, reduce its weight notably and\n"
            "     redistribute it to the others.\n"
            "- R4: only reference client_ids present in the request; never invent ids.\n"
            "- R5: your response must contain EXACTLY ONE JSON array and nothing else\n"
            "     except the optional Reasoning line (R5a).\n"
            "- R5a: " + cot_rule + "\n"
            "</rules>\n\n"
            "<output_format>\n"
            "Return one JSON array only. Element schema:\n"
            '{"client_id": <int>, "weight": <float 0..1>, "reason": "<one short phrase>"}\n'
            "Example:\n"
            '[{"client_id": 0, "weight": 0.35, "reason": "low anomaly scores, params near global"},\n'
            ' {"client_id": 1, "weight": 0.15, "reason": "moderate anomaly scores, params deviate"}]\n'
            "</output_format>\n\n"
            + self._fewshot_examples()
        )

    def _build_user_msg(self, client_ids, client_scores, metadata, default_weights):
        """Prompt do utilizador: bloco <client> por cliente (legibilidade p/ o SLM)."""
        agent_names = metadata.get("agent_names", ["Norm", "Cosine", "Entropy"])
        round_n = metadata.get("round", 0)
        abbrev = {name: name[:2].upper() for name in agent_names}
        client_stats = metadata.get("client_stats", {})

        user = f"<request>\nRound {round_n} | Survival clients: {len(client_ids)}\n"
        for cid in client_ids:
            scores = client_scores.get(cid, [])
            parts = [f"{abbrev.get(agent_names[i], 'D' + str(i))}={s:.4f}" for i, s in enumerate(scores)]
            stats = client_stats.get(cid, "")
            prev = default_weights.get(cid, 0.0)
            user += (
                f"<client id={cid}>\n"
                f"prev_weight={prev:.4f}\n"
                f"detectors: {', '.join(parts)}\n"
                f"layers: {stats}\n"
                "</client>\n"
            )
        user += "</request>"
        return user

    def _build_prompt(self, client_models, client_scores, global_model, metadata):
        """
        Constroi o prompt em formato chat (system/user/assistant).

        O prompt inclui:
        - System prompt estruturado: role/task/rules/output_format + few-shot
        - Scores dos detectores (sem SLM) por cliente (já tratados pela defesa)
        - Stats por camada de cada modelo cliente
        - Pesos de referencia (padrao dataset)
        - (Opcional) linha 'Reasoning:' (CoT) antes do JSON
        """
        if metadata is None:
            metadata = {}
        client_ids = metadata.get("client_ids", list(client_scores.keys()))
        default_weights = metadata.get("default_weights", {})
        agent_names = metadata.get("agent_names", ["Norm", "Cosine", "Entropy"])
        n_detectors = len(agent_names)

        system_msg = self._build_system_msg(agent_names, n_detectors)
        user_msg = self._build_user_msg(client_ids, client_scores, metadata, default_weights)

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
