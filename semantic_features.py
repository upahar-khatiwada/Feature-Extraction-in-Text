import numpy as np
import pandas as pd


# Covers the vocabulary of both this project's corpora: the animal/nature
# sentences in demo.csv (action, animal, place, object, descriptive) and the
# tech news story titles/text in hackernews_dataset.csv (technology, ai,
# business, science). positive_description is shared by both.
SEMANTIC_LEXICON = {
    "action": {
        "chase", "chased", "finish", "finished", "jump", "jumps", "look", "looked",
        "read", "reading", "run", "running", "sat", "wandered",
    },
    "animal": {"cat", "cats", "dog", "dogs", "fox", "mice", "mouse"},
    # "mat" is an object, not a place, so it lives in "object" instead.
    "place": {"forest", "park", "world"},
    "object": {"mat", "book"},
    "technology": {
        "software", "code", "coding", "codebase", "app", "apps", "docker",
        "kernel", "linux", "macos", "windows", "github", "api", "browser",
        "editor", "terminal", "markdown", "compiler", "rust", "python",
        "hardware", "computer", "website", "cli", "sandbox", "dns", "ssh",
    },
    "ai": {
        "ai", "llm", "llms", "chatgpt", "claude", "gpt", "model", "models",
        "agent", "agents", "artificial", "intelligence",
    },
    "business": {
        "startup", "startups", "company", "companies", "market", "economy",
        "economics", "fund", "currency", "pioneer", "product", "customer",
    },
    "science": {
        "quantum", "physics", "protein", "proteins", "mathematician", "science",
        "scientists", "biology", "autism", "genome", "matter", "moons",
        "ocean", "mortality",
    },
    # Words with clearly positive connotation, from either corpus.
    "positive_description": {
        "happy", "interesting", "playful", "popular",
        "innovative", "impressive", "great", "amazing", "useful", "powerful", "open",
    },
    # Words with clearly negative connotation. "sandbox" (technology) vs.
    # "crimes" (negative_description) shows categories are assigned by
    # meaning, not by which story they happen to appear in together.
    "negative_description": {
        "broken", "slow", "insecure", "banned", "restricted", "dead",
        "terrifying", "crime", "crimes", "surveillance", "censorship",
    },
    # Other neutral descriptive adjectives that aren't inherently positive/negative.
    "descriptive": {
        "quick", "quickly", "quiet", "lazy", "brown", "full", "young", "curious",
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