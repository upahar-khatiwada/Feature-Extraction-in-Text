from syntactic_features import syntactic_analysis, subject_verb_object


tagged_tokens = [
    ("The", "DET"),
    ("cat", "NOUN"),
    ("chased", "VERB"),
    ("the", "DET"),
    ("mouse", "NOUN"),
]

expected = {"subject": "cat", "verb": "chased", "object": "mouse"}
actual = subject_verb_object(tagged_tokens)

print("Expected SVO:", expected)
print("Actual SVO:  ", actual)
print("SVO matches:", actual == expected)
print("\n=== Syntactic Features ===")
print(syntactic_analysis(tagged_tokens))

if actual != expected:
    raise AssertionError("The SVO heuristic did not return the expected structure.")