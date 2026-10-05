"""Rule-based meta-agent for selecting one or more defenses per round."""


class RuleBasedMetaAgent:
    """Turn observed round state into an ordered list of defense strategies."""

    VALID_MEDIUM = ("median", "trimmed_mean", "clipping")
    VALID_HIGH = (
        "krum", "multi_krum", "bulyan", "foolsgold",
        "trimmed_mean", "median", "clipping",
    )

    def __init__(self, args):
        self.medium_preferred = getattr(args, "mad_medium_defense", "trimmed_mean")
        configured = getattr(
            args,
            "mad_high_order",
            "bulyan,multi_krum,krum,foolsgold,trimmed_mean,median",
        )
        self.high_order = self._parse_order(configured, self.VALID_HIGH)
        self.client_threshold = float(getattr(args, "mad_client_threshold", 0.35))
        self.anomaly_threshold = float(getattr(args, "mad_anomaly_threshold", 0.45))
        self.last_reason = ""

    @staticmethod
    def _parse_order(value, allowed):
        if isinstance(value, str):
            values = [item.strip().lower() for item in value.split(",")]
        else:
            values = list(value)
        ordered = []
        for item in values:
            if item in allowed and item not in ordered:
                ordered.append(item)
        return ordered

    def select_defenses(self, round_state):
        if isinstance(round_state, dict):
            level = str(round_state["round_level"]).upper()
            max_risk = float(round_state.get("max_risk", 0.0))
            fraction = float(round_state.get("fraction_high", 0.0))
            max_anomaly = float(round_state.get("max_anomaly", 0.0))
            anomalous_fraction = float(round_state.get("fraction_anomalous", 0.0))
        else:
            level = str(round_state).upper()
            max_risk = 0.0
            fraction = 0.0
            max_anomaly = 0.0
            anomalous_fraction = 0.0
        if (0 < fraction <= 0.25 and max_risk >= self.client_threshold) or (
            0 < anomalous_fraction <= 0.25 and max_anomaly >= self.anomaly_threshold
        ):
            self.last_reason = "isolated_high_risk_clients"
            return ["multi_krum", "krum", "trimmed_mean", "median", "clipping"]
        if fraction >= 0.5 or anomalous_fraction >= 0.5:
            self.last_reason = "widespread_high_risk_clients"
            return ["trimmed_mean", "median", "clipping", "multi_krum"]
        if level == "LOW" and max_risk < self.client_threshold:
            self.last_reason = "low_population_and_client_risk"
            return ["fedavg"]
        if level == "HIGH":
            self.last_reason = "high_round_risk"
            return list(self.high_order or self.VALID_HIGH)
        if level == "MEDIUM":
            self.last_reason = "moderate_round_risk"
            preferred = self.medium_preferred
            if preferred not in self.VALID_MEDIUM:
                preferred = "trimmed_mean"
            return [preferred] + [name for name in self.VALID_MEDIUM if name != preferred]
        self.last_reason = "low_round_risk_with_client_alert"
        return ["clipping", "trimmed_mean", "median"]

    def fallback_defenses(self, round_level, attempted):
        """Escalate if validation rejects every candidate at the current level."""
        level = str(round_level).upper()
        if level == "LOW":
            escalation = ["clipping", "trimmed_mean", "median"]
        elif level == "MEDIUM":
            escalation = list(self.high_order or self.VALID_HIGH)
        else:
            escalation = ["clipping", "trimmed_mean", "median", "fedavg"]
        return [name for name in escalation if name not in attempted]
