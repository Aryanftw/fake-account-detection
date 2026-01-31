import textstat

def compute_textstat_features(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "automated_readability_index": 0.0,
            "lexicon_count": 0
        }

    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "gunning_fog": round(textstat.gunning_fog(text), 2),
        "automated_readability_index": round(textstat.automated_readability_index(text), 2),
        "lexicon_count": textstat.lexicon_count(text, removepunct=True)
    }


def textstat_risk_score(features: dict) -> float:
    """
    Convert textstat features into a normalized risk score [0,1]
    """

    risk = 0.0

    # Very easy or very hard text is suspicious
    if features["flesch_reading_ease"] < 30 or features["flesch_reading_ease"] > 90:
        risk += 0.3

    # Very low or very high grade level
    if features["flesch_kincaid_grade"] < 3 or features["flesch_kincaid_grade"] > 16:
        risk += 0.2

    # Spam-like complexity
    if features["gunning_fog"] > 18:
        risk += 0.2

    # Structural abnormality
    if features["automated_readability_index"] > 20:
        risk += 0.2

    if features["lexicon_count"] < 3:
        return 0.0

    # Extremely short vocab
    if features["lexicon_count"] < 5:
        risk += 0.1

    


    return min(risk, 1.0)
