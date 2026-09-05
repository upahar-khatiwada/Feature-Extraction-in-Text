import pandas as pd

VOWELS = set("aeiou")

# Small hand-written lexicon of closed-class words for POS tagging.
POS_LEXICON = {
    "DET": {"a", "an", "the", "this", "that", "these", "those", "no", "every"},
    "PRON": {
        "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
    },
    "PREP": {
        "of", "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "to", "from", "up",
        "down", "out", "off", "over", "under",
    },
    "CONJ": {"and", "or", "but", "if", "then", "so", "than", "nor", "as"},
    "AUX": {
        "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "doing",
        "have", "has", "had", "having",
    },
}

NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ism", "ist")
ADJ_SUFFIXES = ("ful", "ous", "ive", "able", "ible", "al", "ic")
VERB_SUFFIXES = ("ing", "ed")

# Common irregular forms that no suffix rule can recover.
IRREGULAR_LEMMAS = {
    "went": "go", "gone": "go", "going": "go", "goes": "go",
    "ran": "run", "running": "run", "runs": "run",
    "better": "good", "best": "good",
    "worse": "bad", "worst": "bad",
    "mice": "mouse", "geese": "goose", "men": "man", "women": "woman",
    "children": "child", "feet": "foot", "teeth": "tooth",
    "was": "be", "were": "be", "is": "be", "are": "be", "am": "be",
    "been": "be", "being": "be",
    "had": "have", "has": "have", "having": "have",
    "did": "do", "does": "do", "doing": "do",
}


def pos_tag_word(word):
    """Guess a part-of-speech tag for a single word using a lexicon of
    closed-class words plus suffix rules for open-class words."""
    lower = word.lower()

    if lower.isdigit():
        return "NUM"

    for tag, words in POS_LEXICON.items():
        if lower in words:
            return tag

    if lower.endswith("ly") and len(lower) > 4:
        return "ADV"
    if lower.endswith(VERB_SUFFIXES) and len(lower) > 4:
        return "VERB"
    if lower.endswith(ADJ_SUFFIXES):
        return "ADJ"
    if lower.endswith(NOUN_SUFFIXES):
        return "NOUN"

    return "NOUN"


def pos_tag(tokens):
    """Tag a list of tokens, returning a list of (word, tag) pairs."""
    return [(word, pos_tag_word(word)) for word in tokens]


def _degeminate(stem):
    # Undoes a doubled final consonant left over from suffix stripping,
    # e.g. "hopp" (from "hopping") -> "hop".
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in VOWELS:
        return stem[:-1]
    return stem


def lemmatize_word(word):
    """Reduce a word to its dictionary base form. Checks a small table of
    irregular forms first, then falls back to suffix rules. Like the
    stemmer in preprocessing.py, this is a heuristic, not a real
    morphological analyzer, but it aims to return valid words rather than
    truncated fragments."""
    lower = word.lower()

    if lower in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[lower]

    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("ves") and len(lower) > 4:
        return lower[:-3] + "fe"
    if lower.endswith(("sses", "shes", "ches", "xes")):
        return lower[:-2]
    if lower.endswith("es") and len(lower) > 4:
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        return lower[:-1]

    if lower.endswith("ing") and len(lower) > 5:
        return _degeminate(lower[:-3])
    if lower.endswith("ed") and len(lower) > 4:
        return _degeminate(lower[:-2])
    if lower.endswith("est") and len(lower) > 5:
        return _degeminate(lower[:-3])
    if lower.endswith("er") and len(lower) > 4:
        return _degeminate(lower[:-2])
    if lower.endswith("ly") and len(lower) > 4:
        return lower[:-2]

    return lower


def lemmatize_tokens(tokens):
    return [lemmatize_word(word) for word in tokens]


def morphological_features(word):
    """Extract hand-crafted morphological features for a single word:
    shape (prefix/suffix, capitalization, vowel/consonant counts) and
    simple inflection flags (plural, gerund, past tense, etc.)."""
    lower = word.lower()
    return {
        "word": word,
        "length": len(word),
        "num_vowels": sum(1 for ch in lower if ch in VOWELS),
        "num_consonants": sum(1 for ch in lower if ch.isalpha() and ch not in VOWELS),
        "prefix3": lower[:3],
        "suffix3": lower[-3:],
        "is_capitalized": word[:1].isupper(),
        "is_plural": lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3,
        "is_gerund": lower.endswith("ing") and len(lower) > 4,
        "is_past_tense": lower.endswith("ed") and len(lower) > 3,
        "is_comparative": lower.endswith("er") and len(lower) > 4,
        "is_superlative": lower.endswith("est") and len(lower) > 5,
    }


def morphological_analysis(tokens):
    """Build a DataFrame of morphological features, one row per token."""
    return pd.DataFrame([morphological_features(word) for word in tokens])
