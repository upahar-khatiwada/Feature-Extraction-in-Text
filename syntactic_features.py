import pandas as pd


NOMINAL_TAGS = {"NOUN", "PRON"}
VERBAL_TAGS = {"VERB", "AUX"}


def _first_tagged_word(tagged_tokens, allowed_tags, start=0):
    for word, tag in tagged_tokens[start:]:
        if tag in allowed_tags:
            return word
    return None


def subject_verb_object(tagged_tokens):
    """Find simple subject, verb, and object candidates from POS-tagged tokens.

    This is a positional heuristic rather than a full dependency parser: the
    first nominal before the first verb is the subject, and the first nominal
    after that verb is the object.
    """
    verb_index = next(
        (index for index, (_, tag) in enumerate(tagged_tokens) if tag in VERBAL_TAGS),
        None,
    )
    if verb_index is None:
        return {"subject": None, "verb": None, "object": None}

    subject = _first_tagged_word(tagged_tokens[:verb_index], NOMINAL_TAGS)
    verb = tagged_tokens[verb_index][0]
    object_word = _first_tagged_word(tagged_tokens, NOMINAL_TAGS, verb_index + 1)
    return {"subject": subject, "verb": verb, "object": object_word}


def syntactic_analysis(tagged_tokens):
    """Return sentence-level syntax features from a list of (word, POS) pairs."""
    structure = subject_verb_object(tagged_tokens)
    tags = [tag for _, tag in tagged_tokens]
    return pd.DataFrame([{
        **structure,
        "token_count": len(tagged_tokens),
        "noun_count": sum(tag == "NOUN" for tag in tags),
        "verb_count": sum(tag in VERBAL_TAGS for tag in tags),
        "has_subject_verb_object": all(structure.values()),
    }])