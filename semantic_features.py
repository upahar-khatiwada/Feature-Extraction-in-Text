import numpy as np
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


def cosine_similarity_matrix(feature_matrix):
    """Calculate pairwise cosine similarity for document feature vectors."""
    vectors = np.asarray(feature_matrix, dtype=float)
    dot_products = vectors @ vectors.T
    norms = np.linalg.norm(vectors, axis=1)
    denominators = np.outer(norms, norms)
    scores = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products),
        where=denominators != 0,
    )
    labels = [f"doc_{index}" for index in range(len(vectors))]
    return pd.DataFrame(scores, index=labels, columns=labels)