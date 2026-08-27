import re

STOPWORDS = [
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "than",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "to", "from", "up", "down",
    "out", "off", "over", "under", "again", "further",
    "as", "not", "no", "nor",
]


def lowercase(text):
    return text.lower()


def remove_punctuation_and_numbers(text):
    return re.sub(r"[^a-z\s]", " ", text)


def tokenize(text):
    return text.split()


def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOPWORDS]


def stem_word(word):
    suffixes = ["ing", "ed", "ly", "es", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def stem_tokens(tokens):
    return [stem_word(word) for word in tokens]


def preprocess(text):
    text = lowercase(text)
    text = remove_punctuation_and_numbers(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens
