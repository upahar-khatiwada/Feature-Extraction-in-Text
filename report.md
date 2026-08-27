# Report: Feature Extraction in Text

This report explains the theory behind each step used in this project to turn
raw text into numerical features.

## 1. Tokenization

Tokenization means splitting a piece of text into individual words (tokens),
since machine learning models can't work with raw sentences directly. Before
splitting, the text is lowercased and all punctuation and numbers are removed,
so that "Dog" and "dog." are treated as the same word. Tokens are then formed
by simply splitting the cleaned text on whitespace.

## 2. Stopword Removal

Stopwords are very common words (like "the", "is", "and", "of") that appear in
almost every document and don't carry much meaning on their own. Keeping them
would add noise and make documents look more similar than they really are, so
they are removed using a fixed list of common English words.

## 3. Stemming

Stemming reduces words to a shorter base form by chopping off common suffixes,
so that related words like "running", "runs", and "run" are treated as the
same token. This project uses a simple rule-based stemmer that strips known
suffixes (`ing`, `ed`, `ly`, `es`, `s`) from the end of a word, as long as
enough of the word is left afterward. It's not linguistically perfect (it
doesn't know grammar), but it groups related word forms well enough for
counting purposes.

## 4. Bag of Words (BoW)

Bag of Words represents each document as a vector of word counts over a fixed
vocabulary, ignoring word order — only "how many times does each word appear"
matters. For a document `d` and a word `w`:

```
BoW(d, w) = number of times w appears in d
```

A binary version is also used, which only records whether a word appears at
all, not how many times:

```
BinaryBoW(d, w) = 1 if w appears in d, else 0
```

## 5. TF-IDF

The problem with plain word counts is that common words can dominate the
counts even if they aren't very informative for telling documents apart.
TF-IDF fixes this by weighing down words that appear in many documents and
weighing up words that are rare across the corpus but frequent in a specific
document.

- **TF (term frequency)** — how often a word appears in a document. Here it's
  just the raw count:

  ```
  tf(w, d) = number of times w appears in d
  ```

- **IDF (inverse document frequency)** — how rare a word is across all
  documents. If a word appears in every document, it's not very useful for
  distinguishing them, so its IDF is low (close to 0):

  ```
  idf(w) = log(N / df(w))
  ```

  where `N` is the total number of documents and `df(w)` is the number of
  documents that contain `w`.

- **TF-IDF score** — the two combined:

  ```
  tf_idf(w, d) = tf(w, d) * idf(w)
  ```

A word that appears in every document gets `idf = log(N/N) = log(1) = 0`, so
its TF-IDF score is 0 everywhere, no matter how often it appears. A word that
appears in only one document gets a high IDF, so it stands out more.

## 6. N-grams

N-grams are groups of `n` consecutive tokens, used to capture some word order
and short phrases that single words miss (for example "not good" vs. "good"
on their own lose the negation). A bigram is `n = 2` (pairs of words), a
trigram is `n = 3` (triples of words). They're built with a sliding window
over the token list: for each position `i`, take the tokens from `i` to
`i + n`.

## Summary

The overall pipeline is:

```
raw text -> lowercase -> remove punctuation/numbers -> tokenize
         -> remove stopwords -> stem -> feature extraction (BoW / TF-IDF / n-grams)
```

Each step reduces noise and turns unstructured text into a numeric form that
can be compared, counted, and eventually fed into a model.
