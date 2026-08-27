import math
from features import build_vocabulary, bag_of_words, tf_idf

# Tiny 2-document example, tokens given directly (no preprocessing needed).
docs = [
    ["cat", "sat", "mat"],
    ["cat", "dog", "dog"],
]

vocab = build_vocabulary(docs)
print("Vocabulary:", vocab)
# vocab = ["cat", "dog", "mat", "sat"]

bow_df = bag_of_words(docs, vocab)
print("\n=== Bag of Words ===")
print(bow_df)
# Expected counts:
# doc0: cat=1, dog=0, mat=1, sat=1
# doc1: cat=1, dog=2, mat=0, sat=0

tfidf_df = tf_idf(docs, vocab)
print("\n=== TF-IDF ===")
print(tfidf_df.round(4))

# Hand-computed expected values (N = 2 documents):
# df(cat) = 2 -> idf(cat) = log(2/2) = log(1)        = 0.0
# df(dog) = 1 -> idf(dog) = log(2/1) = log(2)        = 0.6931
# df(mat) = 1 -> idf(mat) = log(2/1) = log(2)        = 0.6931
# df(sat) = 1 -> idf(sat) = log(2/1) = log(2)        = 0.6931
#
# doc0: cat = tf(1) * idf(0.0)    = 0.0
#       dog = tf(0) * idf(0.6931) = 0.0
#       mat = tf(1) * idf(0.6931) = 0.6931
#       sat = tf(1) * idf(0.6931) = 0.6931
#
# doc1: cat = tf(1) * idf(0.0)    = 0.0
#       dog = tf(2) * idf(0.6931) = 1.3863
#       mat = tf(0) * idf(0.6931) = 0.0
#       sat = tf(0) * idf(0.6931) = 0.0

expected = {
    (0, "cat"): 0.0,
    (0, "dog"): 0.0,
    (0, "mat"): math.log(2),
    (0, "sat"): math.log(2),
    (1, "cat"): 0.0,
    (1, "dog"): 2 * math.log(2),
    (1, "mat"): 0.0,
    (1, "sat"): 0.0,
}

print("\n=== Checking TF-IDF against hand-computed values ===")
all_match = True
for (doc_index, word), expected_value in expected.items():
    actual_value = tfidf_df.loc[doc_index, word]
    match = math.isclose(actual_value, expected_value, abs_tol=1e-9)
    if not match:
        all_match = False
    print(f"doc{doc_index} '{word}': expected={expected_value:.4f}, actual={actual_value:.4f}, match={match}")

print("\nAll values match:" , all_match)
