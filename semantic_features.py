import pandas as pd


SEMANTIC_LEXICON = {
    "action": {
        "chase", "chased", "finish", "finished", "jump", "jumps", "look", "looked",
        "read", "reading", "run", "running", "sat", "wandered",
    },
    "animal": {"cat", "cats", "dog", "dogs", "fox", "mice", "mouse"},
    "place": {"forest", "mat", "park", "world"},
    "positive_description": {
        "happy", "interesting", "playful", "popular", "quick", "quickly", "quiet",
    },
}


def semantic_category_features(tokens):
    """Count tokens that belong to each hand-written semantic category."""
    lower_tokens = [word.lower() for word in tokens]
    return {
        category: sum(word in words for word in lower_tokens)
        for category, words in SEMANTIC_LEXICON.items()
    }


def semantic_feature_matrix(documents):
    """Return semantic category counts with one row for each document."""
    return pd.DataFrame([semantic_category_features(tokens) for tokens in documents])