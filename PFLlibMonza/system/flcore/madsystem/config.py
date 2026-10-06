"""Translate declarative V2 JSON into the existing PFLlib argument interface."""

import json
import math
from pathlib import Path


def normalized_weights(values, defaults):
    """Validate interpretable, finite, nonnegative weights and normalize them."""
    values = defaults if values is None else values
    if not isinstance(values, dict) or set(values) - set(defaults):
        raise ValueError(f"weights must use these keys: {', '.join(defaults)}")
    weights = {key: float(values.get(key, 0.0)) for key in defaults}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be finite, nonnegative and have a positive sum")
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


SECTIONS = {
    "sentinel": {"history_alpha": "mad_history_alpha", "weights": "mad_anomaly_weights",
                 "population_floor": "mad_population_floor", "cold_start_factor": "mad_cold_start_factor",
                 "history_min_observations": "mad_history_min_observations", "history_sensitivity": "mad_history_sensitivity"},
    "reputation": {"initial": "mad_reputation_initial", "penalty": "mad_reputation_penalty",
                   "recovery": "mad_reputation_recovery", "suspicious_profile_weight": "mad_suspicious_profile_weight"},
    "risk": {"weights": "mad_risk_weights", "round_weights": "mad_round_risk_weights",
             "watch_threshold": "mad_watch_threshold", "suspicious_threshold": "mad_suspicious_threshold",
             "defense_threshold": "mad_defense_threshold", "recovery_threshold": "mad_state_recovery_threshold",
             "window": "mad_persistence_window", "patience": "mad_high_patience",
             "anomaly_threshold": "mad_anomaly_threshold", "history_threshold": "mad_history_threshold",
             "low_threshold": "mad_low_threshold", "high_threshold": "mad_high_threshold"},
    "meta_defense": {"isolated_fraction": "mad_isolated_fraction", "widespread_fraction": "mad_widespread_fraction",
                     "magnitude_threshold": "mad_magnitude_threshold", "medium_defense": "mad_medium_defense",
                     "high_order": "mad_high_order", "max_attempts": "mad_max_defense_attempts",
                     "filter_medium_risk_updates": "mad_filter_medium_risk_updates", "quarantine_rounds": "mad_quarantine_rounds"},
    "validator": {"data": "mad_validation_data", "enabled": "mad_validator_enabled",
                  "batch_size": "mad_validation_batch_size", "loss_tolerance": "mad_validation_loss_tolerance",
                  "accuracy_tolerance": "mad_validation_accuracy_tolerance", "max_delta_norm": "mad_validation_max_delta",
                  "class_accuracy_tolerance": "mad_validation_class_tolerance", "class_min_examples": "mad_validation_class_min_examples"},
    "attack": {"type": "atack", "scale": "attack_scale", "noise_snr": "attack_noise_snr",
               "rate": "rate_client_fake", "source_label": "attack_source_label", "target_label": "attack_target_label"},
    "evaluation": {"batch_size": "mad_evaluation_batch_size", "every": "mad_evaluation_every"},
}


def load_experiment_config(path):
    """Accept flat legacy configs or strict hierarchical V2 JSON; CLI still wins."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    result = {}
    aliases = {"clients": "num_clients", "dataset": "dataset", "seed": "seed"}
    for key, value in raw.items():
        if key == "data_distribution":
            if not isinstance(value, dict) or set(value) - {"type", "alpha", "seed"}:
                raise ValueError("data_distribution accepts type, alpha and seed")
            result["mad_data_distribution"] = value
        elif key == "rounds":
            if not isinstance(value, int) or value < 2:
                raise ValueError("rounds must be an integer >= 2 (exact number of updates)")
            result["global_rounds"] = value - 1
        elif key == "malicious_fraction":
            if not 0 <= float(value) < 1:
                raise ValueError("malicious_fraction must be in [0,1)")
        elif key in SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            for name, setting in value.items():
                if key == "sentinel" and name == "enabled":
                    if setting is not True:
                        raise ValueError("V2 Sentinel must stay enabled in every round")
                elif key == "meta_defense" and name == "policy":
                    if setting != "rule_based":
                        raise ValueError("V2 supports only rule_based meta-defense")
                elif key == "attack" and name == "start_round":
                    if int(setting) < 0:
                        raise ValueError("attack.start_round must be non-negative")
                    result["round_init_atk"] = int(setting) - 1
                elif name not in SECTIONS[key]:
                    raise ValueError(f"unknown {key} option: {name}")
                else:
                    destination = SECTIONS[key][name]
                    if destination == "mad_validation_data" and setting:
                        setting = str((path.parent / setting).resolve())
                    result[destination] = setting
        else:
            result[aliases.get(key, key)] = value
    if "malicious_fraction" in raw:
        if "n_client_malicious" in raw:
            raise ValueError("specify malicious_fraction or n_client_malicious, not both")
        result["n_client_malicious"] = int(int(result.get("num_clients", 20)) * float(raw["malicious_fraction"]))
    return result
