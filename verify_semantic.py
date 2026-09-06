from semantic_features import semantic_category_features, semantic_feature_matrix


tokens = ["Happy", "dogs", "run", "in", "the", "park"]
expected = {
    "action": 1,
    "animal": 1,
    "place": 1,
    "object": 0,
    "positive_description": 1,
    "descriptive": 0,
}
actual = semantic_category_features(tokens)

print("Expected categories:", expected)
print("Actual categories:  ", actual)
assert actual == expected

# "mat" is an object, not a place, and must not be counted under "place".
mat_tokens = ["the", "cat", "sat", "on", "the", "mat"]
mat_actual = semantic_category_features(mat_tokens)
print("\n'mat' categories:", mat_actual)
assert mat_actual["place"] == 0
assert mat_actual["object"] == 1

matrix = semantic_feature_matrix([tokens, ["quiet", "fox", "wandered", "forest"]])
print("\n=== Semantic Feature Matrix ===")
print(matrix)
assert matrix.shape == (2, 6)
assert list(matrix["animal"]) == [1, 1]
assert list(matrix["descriptive"]) == [0, 1]
print("\nSemantic feature checks passed.")