# Feature Extraction in Text

A small hand-built pipeline that turns raw text documents into numerical
features. All text-processing and feature-extraction logic (tokenizing,
stopword removal, stemming, lemmatization, POS tagging, morphological
analysis, bag-of-words, one-hot encoding, TF-IDF, n-grams) is written from
scratch using only `numpy` and `pandas`. No scikit-learn, nltk, spaCy, or
similar libraries are used.

## Files

- `preprocessing.py` — cleans and tokenizes raw text
  - `lowercase(text)` — lowercases the text
  - `remove_punctuation_and_numbers(text)` — strips anything that isn't a
    letter or whitespace
  - `tokenize(text)` — splits text into tokens on whitespace
  - `remove_stopwords(tokens)` — drops common words found in the hand-written
    `STOPWORDS` list
  - `stem_word(word)` / `stem_tokens(tokens)` — a simple suffix-stripping
    stemmer that removes endings like `ing`, `ed`, `ly`, `es`, `s`
  - `preprocess(text)` — runs all of the above in order and returns the final
    list of tokens for a document

- `linguistic_features.py` — rule-based linguistic analysis on tokens
  - `pos_tag_word(word)` / `pos_tag(tokens)` — guesses a part-of-speech tag
    (`DET`, `PRON`, `PREP`, `CONJ`, `AUX`, `ADV`, `VERB`, `ADJ`, `NOUN`,
    `NUM`) using a small closed-class word lexicon plus suffix rules
  - `morphological_features(word)` / `morphological_analysis(tokens)` —
    hand-crafted per-word features: length, vowel/consonant counts,
    prefix/suffix, capitalization, and inflection flags (plural, gerund,
    past tense, comparative, superlative), returned as a DataFrame
  - `lemmatize_word(word)` / `lemmatize_tokens(tokens)` — reduces a word to
    its dictionary base form using a small irregular-form table plus
    suffix rules (e.g. `"looked"` -> `"look"`, `"mice"` -> `"mouse"`);
    unlike the stemmer, it aims to return valid words rather than
    truncated fragments

- `features.py` — builds numerical features from tokenized documents
  - `build_vocabulary(docs)` — sorted list of unique tokens across all
    documents
  - `bag_of_words(docs, vocab)` — DataFrame of word counts (rows = documents,
    columns = vocabulary words)
  - `binary_bow(docs, vocab)` — same as above but 1 if the word appears in the
    document, 0 otherwise
  - `tf_idf(docs, vocab)` — DataFrame of TF-IDF scores. Uses
    `tf * log(N / df)`, where `tf` is the raw count of the word in the
    document, `N` is the number of documents, and `df` is the number of
    documents containing the word
  - `n_grams(tokens, n)` — generates a list of n-grams (tuples of `n`
    consecutive tokens); works for any `n`, including bigrams and trigrams
  - `one_hot_encode_tokens(tokens, vocab)` — one-hot vector per token
    *position* in a sequence (as opposed to `binary_bow`, which gives one
    vector per whole document)
  - `one_hot_encode_categories(categories)` — generic one-hot encoder for
    any list of category labels, e.g. POS tags

- `main.py` — runs the full pipeline end to end and prints the tokenized
  documents, vocabulary, bag of words, TF-IDF table, a bigram/trigram
  example, POS tags, morphological features, a lemma-vs-stem comparison, and
  one-hot encodings (of tokens and of POS tags). Uses a small hardcoded
  corpus (5-6 sentences) by default, or reads documents from a CSV file if
  one is given on the command line.

## How to run

```
python main.py
```

To run on your own documents instead, pass a CSV file with a `text` column:

```
python main.py path/to/documents.csv
```

Requires only `numpy` and `pandas` to be installed.
