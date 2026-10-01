"""Rule-based meta-agent for selecting one or more defenses per round."""


class RuleBasedMetaAgent:
    """Turn round risk into an ordered list of independent defense agents."""

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

    def select_defenses(self, round_level):
        level = str(round_level).upper()
        if level == "HIGH":
            return list(self.high_order or self.VALID_HIGH)
        if level == "MEDIUM":
            preferred = self.medium_preferred
            if preferred not in self.VALID_MEDIUM:
                preferred = "trimmed_mean"
            return [preferred] + [name for name in self.VALID_MEDIUM if name != preferred]
        return ["fedavg"]

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
