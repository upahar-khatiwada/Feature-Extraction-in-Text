from semantic_features import semantic_category_features, semantic_feature_matrix


all_categories = {
    "action", "animal", "place", "object", "technology", "ai", "business",
    "science", "positive_description", "negative_description", "descriptive",
}

# demo.csv-style tokens (animal/nature sentences).
demo_tokens = ["Happy", "dogs", "run", "in", "the", "park"]
demo_expected = {category: 0 for category in all_categories}
demo_expected.update({"action": 1, "animal": 1, "place": 1, "positive_description": 1})
demo_actual = semantic_category_features(demo_tokens)

print("Expected (demo.csv-style):", demo_expected)
print("Actual (demo.csv-style):  ", demo_actual)
assert demo_actual == demo_expected

# "mat" is an object, not a place, and must not be counted under "place".
mat_tokens = ["the", "cat", "sat", "on", "the", "mat"]
mat_actual = semantic_category_features(mat_tokens)
print("\n'mat' categories:", mat_actual)
assert mat_actual["place"] == 0
assert mat_actual["object"] == 1

# hackernews_dataset.csv-style tokens (tech news titles/text).
hn_tokens = ["The", "innovative", "ai", "startup", "uses", "linux"]
hn_expected = {category: 0 for category in all_categories}
hn_expected.update({"technology": 1, "ai": 1, "business": 1, "positive_description": 1})
hn_actual = semantic_category_features(hn_tokens)

print("\nExpected (hackernews-style):", hn_expected)
print("Actual (hackernews-style):  ", hn_actual)
assert hn_actual == hn_expected

# "sandbox" is technology, not negative_description, even though it shows up
# in a headline about committing crimes.
sandbox_tokens = ["the", "software", "breaks", "out", "of", "the", "sandbox"]
sandbox_actual = semantic_category_features(sandbox_tokens)
print("\n'sandbox' categories:", sandbox_actual)
assert sandbox_actual["technology"] == 2
assert sandbox_actual["negative_description"] == 0

# "crimes" is negative_description, not technology.
crime_tokens = ["the", "company", "committed", "crimes"]
crime_actual = semantic_category_features(crime_tokens)
print("'crimes' categories:", crime_actual)
assert crime_actual["negative_description"] == 1
assert crime_actual["technology"] == 0

matrix = semantic_feature_matrix([
    demo_tokens,
    ["quiet", "fox", "wandered", "forest"],
    ["quantum", "physics", "protein", "scientists"],
])
print("\n=== Semantic Feature Matrix (mixed demo.csv + hackernews docs) ===")
print(matrix)
assert matrix.shape == (3, len(all_categories))
assert list(matrix["animal"]) == [1, 1, 0]
assert list(matrix["descriptive"]) == [0, 1, 0]
assert list(matrix["science"]) == [0, 0, 4]
print("\nSemantic feature checks passed.")