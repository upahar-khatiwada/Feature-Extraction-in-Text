import pandas as pd

from preprocessing import preprocess
from features import build_vocabulary, bag_of_words, binary_bow, tf_idf, n_grams

pd.set_option("display.width", 120)

corpus = [
    "The cat sat on the mat and looked at the dog.",
    "Dogs are running quickly in the park every morning.",
    "The quick brown fox jumps over the lazy dog.",
    "Cats and dogs are popular pets around the world.",
    "She quickly finished reading the interesting book.",
    "The park was full of happy dogs and playful cats.",
]

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

print("\n=== N-gram example (Doc 0) ===")
print("Bigrams:", n_grams(processed_docs[0], 2))
print("Trigrams:", n_grams(processed_docs[0], 3))
