
def classify_risk(text):
    text = str(text).lower()

    categories = {
        "credit_risk": [
            "bad loans", "loan defaults", "loan default",
            "credit quality", "rising loan defaults"
        ],
        "regulatory_risk": [
            "investigation", "regulatory", "compliance",
            "regulatory action", "legal action"
        ],
        "operational_risk": [
            "supply chain", "supply issues", "disruptions",
            "vehicle recall", "recalled vehicles", "recall",
            "safety issue", "operational failure"
        ],
        "market_risk": [
            "declining sales", "weak demand",
            "weaker consumer demand", "sales seem weak",
            "market volatility", "falling prices",
            "uncertainty", "weak"
        ],
        "positive_event": [
            "record revenue", "stronger profit",
            "successful capital raise", "capital raise",
            "renewable energy contract", "growth outlook",
            "positive development", "strong earnings",
            "promising", "new contract"
        ]
    }

    # Prioritize adverse events when multiple keywords match.
    priority = [
        "credit_risk",
        "regulatory_risk",
        "operational_risk",
        "market_risk",
        "positive_event"
    ]

    for category in priority:
        if any(keyword in text for keyword in categories[category]):
            return category

    return "general"