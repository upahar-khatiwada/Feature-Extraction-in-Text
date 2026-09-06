import sys

import pandas as pd

from preprocessing import (
    preprocess,
    lowercase,
    remove_punctuation_and_numbers,
    tokenize,
    remove_stopwords,
    stem_word,
)
from features import (
    build_vocabulary,
    bag_of_words,
    binary_bow,
    tf_idf,
    n_grams,
    one_hot_encode_tokens,
    one_hot_encode_categories,
)
from linguistic_features import (
    pos_tag,
    morphological_analysis,
    lemmatize_word,
)
from syntactic_features import syntactic_analysis, syntactic_feature_matrix
from semantic_features import cosine_similarity_matrix, semantic_feature_matrix

pd.set_option("display.width", 120)

default_corpus = [
    "The cat sat on the mat and looked at the dog.",
    "Dogs are running quickly in the park every morning.",
    "The quick brown fox jumps over the lazy dog.",
    "Cats and dogs are popular pets around the world.",
    "She quickly finished reading the interesting book.",
    "The park was full of happy dogs and playful cats.",
]

if len(sys.argv) > 1:
    csv_path = sys.argv[1]
    data = pd.read_csv(csv_path)
    corpus = data["text"].tolist()
    print(f"Loaded {len(corpus)} documents from {csv_path}")
else:
    corpus = default_corpus

print("=== Raw corpus ===")
for i, doc in enumerate(corpus):
    print(f"Doc {i}: {doc}")

processed_docs = [preprocess(doc) for doc in corpus]

print("\n=== Tokens after preprocessing ===")
for i, tokens in enumerate(processed_docs):
    print(f"Doc {i}: {tokens}")

vocab = build_vocabulary(processed_docs)
print(f"\n=== Vocabulary ({len(vocab)} words) ===")
print(vocab)

bow_df = bag_of_words(processed_docs, vocab)
print("\n=== Bag of Words ===")
print(bow_df)

binary_df = binary_bow(processed_docs, vocab)
print("\n=== Binary Bag of Words ===")
print(binary_df)

tfidf_df = tf_idf(processed_docs, vocab)
print("\n=== TF-IDF ===")
print(tfidf_df.round(3))

print("\n=== Semantic Similarity (TF-IDF Cosine) ===")
print(cosine_similarity_matrix(tfidf_df).round(3))

print("\n=== N-gram example (Doc 0) ===")
print("Bigrams:", n_grams(processed_docs[0], 2))
print("Trigrams:", n_grams(processed_docs[0], 3))

# POS tagging, morphological analysis, and lemmatization work best on tokens
# that still have their original word forms and function words, so they run
# on lightly cleaned tokens rather than the stemmed/stopword-free ones above.
raw_tokens_doc0 = tokenize(remove_punctuation_and_numbers(lowercase(corpus[0])))

print("\n=== POS Tagging (Doc 0) ===")
tagged_doc0 = pos_tag(raw_tokens_doc0)
for word, tag in tagged_doc0:
    print(f"{word:12s} -> {tag}")

print("\n=== Syntactic Features (Doc 0) ===")
print(syntactic_analysis(tagged_doc0))

tagged_documents = [
    pos_tag(tokenize(remove_punctuation_and_numbers(lowercase(doc))))
    for doc in corpus
]
print("\n=== Syntactic Feature Matrix ===")
print(syntactic_feature_matrix(tagged_documents))

lightly_cleaned_documents = [
    tokenize(remove_punctuation_and_numbers(lowercase(doc)))
    for doc in corpus
]
print("\n=== Semantic Feature Matrix ===")
print(semantic_feature_matrix(lightly_cleaned_documents))

print("\n=== Morphological Analysis (Doc 0) ===")
print(morphological_analysis(raw_tokens_doc0))

print("\n=== Lemmatization vs. Stemming (Doc 0, stopwords removed) ===")
filtered_doc0 = remove_stopwords(raw_tokens_doc0)
print(f"{'word':12s} {'stem':12s} {'lemma':12s}")
for word in filtered_doc0:
    print(f"{word:12s} {stem_word(word):12s} {lemmatize_word(word):12s}")

print("\n=== One-hot Encoding (tokens, Doc 0) ===")
print(one_hot_encode_tokens(processed_docs[0], vocab))

print("\n=== One-hot Encoding (POS tags, Doc 0) ===")
pos_tags_doc0 = [tag for _, tag in tagged_doc0]
print(one_hot_encode_categories(pos_tags_doc0))
