from semantic_features import semantic_category_features, semantic_feature_matrix


tokens = ["Happy", "dogs", "run", "in", "the", "park"]
expected = {
    "action": 1,
    "animal": 1,
    "place": 1,
    "positive_description": 1,
}
actual = semantic_category_features(tokens)

print("Expected categories:", expected)
print("Actual categories:  ", actual)
assert actual == expected

matrix = semantic_feature_matrix([tokens, ["quiet", "fox", "wandered", "forest"]])
print("\n=== Semantic Feature Matrix ===")
print(matrix)
assert matrix.shape == (2, 4)
assert list(matrix["animal"]) == [1, 1]
print("\nSemantic feature checks passed.")