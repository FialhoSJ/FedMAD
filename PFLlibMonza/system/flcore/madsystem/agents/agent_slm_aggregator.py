"""
SLMAggregatorAgent - Agregador baseado em SLM para deteccao de clientes maliciosos.

Substitui a media aritmetica (AggregatorAgent original) por raciocinio
com Small Language Model que analisa os scores dos 4 detectores.

Referencia: SLMFORGE (Sheikhi, IEEE BigData 2025) - SLMs em FL para cybersecurity.
Modelo recomendado: Phi-3-mini (3.8B) - bom equilibrio entre raciocinio e custo.

Fluxo:
1. Recebe scores de 4 agentes (EmInspector, FedREDefense, Behavior, FedLLMGuard)
2. Converte para prompt estruturado com contexto do round + historico
3. SLM gera JSON: client_id, score (0-1), verdict, justificacao textual
4. Fallback para media aritmetica se SLM falhar ou modelo nao carregado
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
        self.max_tokens = getattr(args, 'slm_max_tokens', 2048)   # tokens maximos na resposta
        self.rounds_per_call = getattr(args, 'slm_every_n', 1)    # executa SLM a cada N rounds (cache nos restantes)
        self.last_round = -1
        self.cached_scores = None

        # Carrega o SLM na inicializacao (pode demorar ~minutos no primeiro uso)
        if getattr(args, 'slm_enabled', True):
            self._load_model()

    def _load_model(self):
        """Carrega o SLM (Phi-3-mini, TinyLlama, etc.) via transformers com auto-detect CPU/GPU."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            print(f"[SLM] A carregar modelo: {self.model_name} em {self.device}...", flush=True)

            # Carrega config primeiro para corrigir possiveis incompatibilidades
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
                if 'type' not in config.rope_scaling:
                    print("[SLM] A corrigir rope_scaling (falta 'type')", flush=True)
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

            print("[SLM] Modelo carregado com sucesso!", flush=True)
        except Exception as e:
            print(f"[SLM] FALHA ao carregar modelo: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print("[SLM] A usar fallback (media aritmetica).", flush=True)
            self.model = None
            self.tokenizer = None

    def _build_prompt(self, client_scores, metadata):
        """
        Constroi o prompt em formato chat (system/user/assistant).

        Usa apply_chat_template do tokenizer para compatibilidade entre
        modelos (Phi-3, TinyLlama, Gemma, etc.).

        O prompt inclui:
        - Sistema curto + exemplo few-shot do JSON esperado
        - Scores dos detectores ativos para cada cliente
        """
        round_n = metadata.get("round", 0)
        n_clients = len(client_scores)
        quarantined = metadata.get("quarantined_ids", [])
        agent_names = metadata.get("agent_names", ["EmInspector", "FedREDefense", "Behavior", "FedLLMGuard"])
        n_detectors = len(agent_names)

        # Few-shot: mostra 1 exemplo para o modelo aprender o formato JSON
        example_json = f'''
Example output:
[{{"client_id": 0, "score": 0.1, "verdict": "benign", "reason": "low scores across all {n_detectors} detectors"}},
 {{"client_id": 1, "score": 0.9, "verdict": "malicious", "reason": "high scores on all {n_detectors} detectors"}}]
'''

        system_msg = (
            "Analyze anomaly detection scores for FL clients.\n"
            f"{n_detectors} detectors: {', '.join(agent_names)} (0=benign, 1=malicious).\n"
            "Output ONLY the JSON array. No preamble, no explanations, no markdown.\n"
            + example_json
        )

        abbrev = {name: name[:2].upper() for name in agent_names}
        user_msg = f"Round {round_n} | {n_clients} clients | {quarantined} quarantined\n"
        for cid, scores in client_scores.items():
            parts = [f"{abbrev.get(agent_names[i], f'D{i}')}={s:.4f}" for i, s in enumerate(scores)]
            user_msg += f"Client {cid}: {', '.join(parts)}\n"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _parse_response(self, text, client_ids):
        """
        Extrai o JSON da resposta gerada pelo SLM.

        Estrategias de parsing (por ordem de precisao):
        1. Tenta extrair bloco JSON array [...] completo via regex
        2. Fallback: parse linha a linha para objetos JSON individuais {...}
        3. Se ambos falharem, dicionario vazio -> fallback para media aritmetica
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

    def aggregate(self, client_scores, metadata=None):
        """
        Metodo principal do agregador SLM.

        Entrada:
            client_scores: dict {client_id: [score_emInspector, score_fedRE, score_behavior]}
            metadata: dict com round, client_ids, history, encoder, quarantined_ids

        Saida:
            dict {client_id: final_score} (float 0-1)

        Fluxo:
        1. Se SLM nao carregou -> fallback media aritmetica
        2. Cache: se round % N != 0, reusa scores do ultimo round SLM
        3. Constroi prompt -> SLM infere -> parse JSON -> extrai scores + justificacoes
        4. Clientes sem resposta do SLM usam media aritmetica individual
        """
        if metadata is None:
            metadata = {}

        # Fallback global: SLM nao carregou (erro no __init__)
        if self.model is None:
            print("[SLM] Modelo nao disponivel. Fallback: media aritmetica.", flush=True)
            return self._arithmetic_mean(client_scores)

        round_n = metadata.get("round", 0)

        # Cache: SLM executa apenas a cada N rounds (poupa latencia/token)
        if round_n > 0 and round_n % self.rounds_per_call != 0 and self.cached_scores is not None:
            return self.cached_scores

        # Constroi o prompt com scores e metadados
        prompt = self._build_prompt(client_scores, metadata)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            print(f"[SLM] A inferir para round {round_n} ({len(client_scores)} clients)...", flush=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,         # greedy decoding (mais consistente)
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Descodifica apenas os tokens gerados (ignora o prompt original)
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"[SLM] Raw response (round {round_n}, {len(response)} chars):\n{response[:600]}", flush=True)
            parsed = self._parse_response(response, list(client_scores.keys()))
            print(f"[SLM] Parsed {len(parsed)}/{len(client_scores)} clients", flush=True)

            # Constroi dicionario final: score do SLM ou media aritmetica
            final_scores = {}
            for cid in client_scores:
                if cid in parsed:
                    final_scores[cid] = parsed[cid].get("score", 0.5)
                    # Log com a justificacao textual do SLM
                    reason = parsed[cid].get("reason", "")
                    verdict = parsed[cid].get("verdict", "unknown")
                    print(f"[SLM] Client {cid} | score={final_scores[cid]:.4f} | verdict={verdict} | razao: {reason}", flush=True)
                else:
                    # Fallback individual: media dos 3 scores (cliente nao analisado pelo SLM)
                    s = client_scores[cid]
                    if len(s) > 0:
                        final_scores[cid] = sum(s) / len(s)
                    else:
                        final_scores[cid] = 0.5

            # Atualiza cache
            self.cached_scores = final_scores
            self.last_round = round_n
            return final_scores

        except Exception as e:
            print(f"[SLM] ERRO na inferencia: {e}", flush=True)
            print("[SLM] A usar fallback (media aritmetica).", flush=True)
            return self._arithmetic_mean(client_scores)

    def _arithmetic_mean(self, client_scores):
        """Fallback: media simples dos 3 scores (comportamento original do AggregatorAgent)."""
        result = {}
        for cid, scores in client_scores.items():
            result[cid] = sum(scores) / len(scores)
        return result
