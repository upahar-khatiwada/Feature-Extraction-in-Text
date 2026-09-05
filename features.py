import math
import numpy as np
import pandas as pd


def build_vocabulary(docs):
    vocab_set = set()
    for tokens in docs:
        for word in tokens:
            vocab_set.add(word)
    return sorted(vocab_set)


def bag_of_words(docs, vocab):
    counts = np.zeros((len(docs), len(vocab)), dtype=int)
    for doc_index, tokens in enumerate(docs):
        for word in tokens:
            if word in vocab:
                word_index = vocab.index(word)
                counts[doc_index, word_index] += 1
    return pd.DataFrame(counts, columns=vocab)


def binary_bow(docs, vocab):
    presence = np.zeros((len(docs), len(vocab)), dtype=int)
    for doc_index, tokens in enumerate(docs):
        unique_words = set(tokens)
        for word in unique_words:
            if word in vocab:
                word_index = vocab.index(word)
                presence[doc_index, word_index] = 1
    return pd.DataFrame(presence, columns=vocab)


def tf_idf(docs, vocab):
    # tf = raw count of word in doc, idf = log(N / df), score = tf * idf
    num_docs = len(docs)

    doc_freq = {}
    for word in vocab:
        count = 0
        for tokens in docs:
            if word in tokens:
                count += 1
        doc_freq[word] = count

    scores = np.zeros((num_docs, len(vocab)))
    for doc_index, tokens in enumerate(docs):
        for word_index, word in enumerate(vocab):
            tf = tokens.count(word)
            if tf > 0:
                idf = math.log(num_docs / doc_freq[word])
                scores[doc_index, word_index] = tf * idf

    return pd.DataFrame(scores, columns=vocab)


def one_hot_encode_tokens(tokens, vocab):
    # One-hot vector per token position (sequence-style encoding), as
    # opposed to binary_bow which gives one vector per whole document.
    vocab_index = {word: i for i, word in enumerate(vocab)}
    vectors = np.zeros((len(tokens), len(vocab)), dtype=int)
    for row, word in enumerate(tokens):
        if word in vocab_index:
            vectors[row, vocab_index[word]] = 1
    return pd.DataFrame(vectors, columns=vocab)


def one_hot_encode_categories(categories):
    # Generic one-hot encoder for any list of category labels, e.g. POS tags.
    labels = sorted(set(categories))
    label_index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(categories), len(labels)), dtype=int)
    for row, category in enumerate(categories):
        matrix[row, label_index[category]] = 1
    return pd.DataFrame(matrix, columns=labels)


def n_grams(tokens, n):
    grams = []
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        grams.append(gram)
    return grams
