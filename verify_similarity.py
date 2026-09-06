import math

import pandas as pd

from semantic_features import cosine_similarity_matrix


vectors = pd.DataFrame([
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])
similarities = cosine_similarity_matrix(vectors)

print(similarities.round(4))
assert math.isclose(similarities.loc["doc_0", "doc_0"], 1.0)
assert math.isclose(similarities.loc["doc_0", "doc_2"], 0.0)
assert math.isclose(similarities.loc["doc_0", "doc_1"], 1 / math.sqrt(2))
assert math.isclose(similarities.loc["doc_1", "doc_0"], similarities.loc["doc_0", "doc_1"])
print("\nCosine similarity checks passed.")