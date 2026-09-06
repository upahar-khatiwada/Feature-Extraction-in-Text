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
        "down", "out", "off", "over", "under", "around",
    },
    "CONJ": {"and", "or", "but", "if", "then", "so", "than", "nor", "as"},
    "AUX": {
        "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "doing",
        "have", "has", "had", "having",
    },
    # Common descriptive adjectives that don't match any suffix rule below.
    "ADJ": {
        "quick", "brown", "lazy", "popular", "full", "happy", "quiet", "young",
    },
}

NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ism", "ist")
ADJ_SUFFIXES = ("ful", "ous", "ive", "able", "ible", "al", "ic")
VERB_SUFFIXES = ("ing", "ed")

# Irregular verb forms that no suffix rule can recover (unlike "looked" or
# "wandered", words like "sat" carry no -ed/-ing ending to key off of).
IRREGULAR_VERB_FORMS = {
    "sat": "sit", "ate": "eat", "saw": "see", "ran": "run", "went": "go",
    "came": "come", "gave": "give", "took": "take", "found": "find",
    "made": "make", "said": "say", "stood": "stand", "held": "hold",
    "left": "leave", "met": "meet", "sang": "sing", "swam": "swim",
    "won": "win", "knew": "know", "threw": "throw", "grew": "grow",
    "flew": "fly", "wore": "wear", "broke": "break", "chose": "choose",
    "drove": "drive", "rode": "ride", "wrote": "write", "spoke": "speak",
}

# Common verb base forms, used to recognize present-tense "-s"/"-es" verbs
# (e.g. "jumps") without misreading plural nouns like "dogs" as verbs.
COMMON_VERB_STEMS = {
    "jump", "look", "chase", "run", "sit", "walk", "talk", "read", "write",
    "find", "finish", "wander", "stand", "play", "bark", "laugh", "smile",
    "climb", "swim", "fly", "cook", "clean", "paint", "dance", "sing",
    "kick", "push", "pull", "open", "close", "move", "stop", "start",
    "help", "watch", "listen", "eat", "drink", "sleep", "work", "live",
    "love", "like", "want", "need", "know", "think", "see", "hear",
    "feel", "grow", "fall", "rise", "wait",
}

# Words ending in "-ing" that are actually nouns or adjectives, not verbs
# (a plain suffix rule can't tell "morning" from "running").
ING_EXCEPTIONS = {
    "morning": "NOUN", "evening": "NOUN", "building": "NOUN",
    "meeting": "NOUN", "feeling": "NOUN", "painting": "NOUN",
    "wedding": "NOUN", "ceiling": "NOUN",
    "interesting": "ADJ", "exciting": "ADJ", "boring": "ADJ",
    "amazing": "ADJ", "annoying": "ADJ", "fascinating": "ADJ",
    "surprising": "ADJ", "confusing": "ADJ", "tiring": "ADJ",
    "charming": "ADJ",
}
NOUN_ING_WORDS = {word for word, tag in ING_EXCEPTIONS.items() if tag == "NOUN"}

# Common "-er"/"-est" words that are not comparative/superlative adjectives.
NON_COMPARATIVE_ER_WORDS = {
    "over", "under", "water", "summer", "winter", "letter", "matter",
    "paper", "proper", "other", "after", "enter", "offer", "order",
    "power", "number", "weather", "master", "sister", "brother",
    "mother", "father", "teacher", "dinner", "corner", "answer",
    "member", "finger", "river",
}
NON_SUPERLATIVE_EST_WORDS = {
    "forest", "interest", "honest", "harvest", "modest", "protest", "request",
}


def _verb_stem_candidates(lower):
    """Return possible base-verb forms for a present-tense "-s" word."""
    candidates = []
    if lower.endswith("ies") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")
    if lower.endswith("es") and len(lower) > 4:
        candidates.append(lower[:-2])
    if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        candidates.append(lower[:-1])
    return candidates


def _is_verb_like_s_form(lower):
    return any(stem in COMMON_VERB_STEMS for stem in _verb_stem_candidates(lower))

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

    if lower in IRREGULAR_VERB_FORMS:
        return "VERB"
    if lower in ING_EXCEPTIONS:
        return ING_EXCEPTIONS[lower]

    if lower.endswith("ly") and len(lower) > 4:
        return "ADV"
    if lower.endswith(VERB_SUFFIXES) and len(lower) > 4:
        return "VERB"
    if _is_verb_like_s_form(lower):
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
    is_plural = (
        lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3
        and not _is_verb_like_s_form(lower)
    )
    is_gerund = (
        lower.endswith("ing") and len(lower) > 4 and lower not in NOUN_ING_WORDS
    )
    is_past_tense = (
        lower.endswith("ed") and len(lower) > 3
    ) or lower in IRREGULAR_VERB_FORMS
    is_comparative = (
        lower.endswith("er") and len(lower) > 4
        and lower not in NON_COMPARATIVE_ER_WORDS
    )
    is_superlative = (
        lower.endswith("est") and len(lower) > 5
        and lower not in NON_SUPERLATIVE_EST_WORDS
    )
    return {
        "word": word,
        "length": len(word),
        "num_vowels": sum(1 for ch in lower if ch in VOWELS),
        "num_consonants": sum(1 for ch in lower if ch.isalpha() and ch not in VOWELS),
        "prefix3": lower[:3],
        "suffix3": lower[-3:],
        "is_capitalized": word[:1].isupper(),
        "is_plural": is_plural,
        "is_gerund": is_gerund,
        "is_past_tense": is_past_tense,
        "is_comparative": is_comparative,
        "is_superlative": is_superlative,
    }


def morphological_analysis(tokens):
    """Build a DataFrame of morphological features, one row per token."""
    return pd.DataFrame([morphological_features(word) for word in tokens])
