from syntactic_features import syntactic_feature_matrix


tagged_documents = [
    [("cats", "NOUN"), ("chase", "VERB"), ("mice", "NOUN")],
    [("quiet", "ADJ"), ("forest", "NOUN")],
]

matrix = syntactic_feature_matrix(tagged_documents)

print(matrix)
assert list(matrix["token_count"]) == [3, 2]
assert list(matrix["noun_count"]) == [2, 1]
assert list(matrix["verb_count"]) == [1, 0]
assert list(matrix["has_subject_verb_object"]) == [True, False]
print("\nSyntactic feature matrix checks passed.")