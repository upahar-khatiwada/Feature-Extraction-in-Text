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

## 4. POS Tagging

Part-of-speech (POS) tagging labels each word with its grammatical role —
noun, verb, adjective, determiner, and so on. Knowing that "dogs" is a noun
and "running" is a verb captures information that a plain bag of words
throws away. This project uses a rule-based tagger: a small hand-written
lexicon covers closed-class words that must be memorized (determiners like
"the", pronouns, prepositions, conjunctions, auxiliary verbs, and common
descriptive adjectives like "quick" or "happy"), and suffix rules cover
open-class words (`-ly` -> adverb, `-ing`/`-ed` -> verb, `-ful`/`-ous`/`-ive`
-> adjective, `-tion`/`-ness`/`-ity` -> noun), with noun as the default guess.
Small exception tables also cover cases the suffix rules get wrong on their
own: irregular verbs with no `-ed`/`-ing` ending (e.g. "sat" -> verb, not the
default noun), present-tense "-s" verbs like "jumps" (checked against a list
of common verb stems so plural nouns like "dogs" aren't misread as verbs),
and ambiguous `-ing` words that are actually nouns or adjectives ("morning" ->
noun, "interesting" -> adjective, instead of the generic "-ing" -> verb
rule). It still won't resolve every genuinely ambiguous word, but it's cheap
and needs no training data.

## 5. Morphological Analysis

Morphology is the internal structure of a word — its root plus whatever
prefixes, suffixes, and inflections are attached to it. Rather than trying
to fully parse that structure, this project extracts a set of shallow but
useful morphological features per word: length, vowel/consonant counts, the
first and last three characters (crude prefix/suffix), whether the word is
capitalized, and boolean flags for common inflections (`is_plural`,
`is_gerund`, `is_past_tense`, `is_comparative`, `is_superlative`) based on
suffix pattern matching. These features are cheap signals a model can use
even for words it has never seen before.

## 6. Lemmatization

Like stemming, lemmatization reduces a word to a base form so that related
forms are counted together — but where the stemmer in this project just
chops off suffixes and can produce fragments that aren't real words (e.g.
`"quickly"` -> `"quick"` is fine, but `"having"` -> `"hav"` is not), the
lemmatizer aims to return an actual dictionary word (a lemma). It does this
in two steps:

1. Look the word up in a small table of irregular forms that no suffix rule
   could ever derive (`"went"` -> `"go"`, `"mice"` -> `"mouse"`,
   `"better"` -> `"good"`).
2. If it's not irregular, apply suffix rules tuned to leave a real word
   behind (`"ies"` -> `"y"`, `"ves"` -> `"fe"`, undoing doubled consonants
   from suffixes like `-ing`/`-ed`, e.g. `"hopping"` -> `"hop"`).

It's still a heuristic, not a real dictionary-backed morphological analyzer,
but it produces cleaner base forms than the stemmer at the cost of a few
more rules.

## 7. One-Hot Encoding

One-hot encoding represents a category as a vector that is 0 everywhere
except a single 1 marking which category it is. This project uses it in two
ways:

- **Per-token position** (`one_hot_encode_tokens`) — each token in a
  sequence gets its own one-hot vector over the vocabulary, preserving word
  order and position (unlike bag-of-words, which collapses a whole document
  into one count vector).
- **Per-category label** (`one_hot_encode_categories`) — a generic encoder
  for any list of labels, used here to turn POS tags into one-hot vectors so
  they can be fed into a model alongside numeric features.

## 8. Bag of Words (BoW)

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

## 9. TF-IDF

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

## 10. N-grams

N-grams are groups of `n` consecutive tokens, used to capture some word order
and short phrases that single words miss (for example "not good" vs. "good"
on their own lose the negation). A bigram is `n = 2` (pairs of words), a
trigram is `n = 3` (triples of words). They're built with a sliding window
over the token list: for each position `i`, take the tokens from `i` to
`i + n`.

## 11. Syntactic Features

Syntax describes how words are arranged into a sentence. Rather than
guessing sentence structure from raw token order, this project follows a
small pipeline:

```
tokens -> POS tagging -> parsing -> parse tree -> extracted features
```

The "parsing" step is a hand-written recursive-descent chunker over a tiny
context-free grammar built from the POS tags:

```
S  -> NP? VP
NP -> PRON | DET? ADJ* NOUN+
VP -> AUX? ADV? (VERB|AUX) NP? PP*
PP -> PREP NP
```

It consumes POS-tagged tokens clause by clause (splitting on coordinating
conjunctions) and builds an actual tree of `S`/`NP`/`VP`/`PP` nodes, rather
than a flat list. From that tree, the project extracts:

- **Subject / verb / object** — the head noun or pronoun of the first
  clause's `NP` is the subject, the first verbal leaf in its `VP` is the
  verb, and the head of the first `NP` found inside that `VP` (a direct
  object, or a preposition's object if there's no direct object) is the
  object.
- **Noun / verb / adjective counts** — tag counts across the whole sentence.
- **Clause count** — the number of `S` nodes the parser found, which
  increases for coordinated sentences like "The cat sat... and looked...".
- **Dependency relations** — simple head-dependent pairs read directly off
  the tree: `det` (determiner -> noun), `amod` (adjective -> noun it
  modifies), `nsubj` (subject noun -> verb), `dobj` (verb -> direct object
  noun), and `pobj` (preposition -> its object noun). This approximates a
  dependency parse using the constituency tree instead of parsing
  dependencies directly.

For example, for `The cat chased the mouse`, the parser builds an `S` node
containing an `NP` (`the cat`) and a `VP` (`chased` + `NP` `the mouse`), from
which it reads `cat` as the subject, `chased` as the verb, `mouse` as the
object, and the dependency `dobj: chased -> mouse`. This is still a
hand-written chunking parser, not a trained statistical or dependency
parser, so its correctness depends on the rule-based POS tags and it cannot
resolve every sentence form — but building a real tree (instead of a flat
positional guess) makes clause counts and head-dependent relations possible
to extract directly.

## 12. Semantic Features

Semantic features represent meaning. A small hand-written lexicon assigns
known words to six categories: `action`, `animal`, `place`, `object`,
`positive_description` (clearly positive words like "happy" or "popular"),
and `descriptive` (other neutral descriptive words like "quick" or "lazy").
Words are assigned by meaning rather than by suffix, so, for example, "mat"
is counted as an `object` rather than a `place`. For each document, the
project counts how many tokens belong to each category. For example,
`Happy dogs run in the park` contributes one word to each of `action`,
`animal`, `place`, and `positive_description`: `run`, `dogs`, `park`, and
`happy`.

The project also compares the overall semantic relatedness of documents with
cosine similarity applied to their TF-IDF vectors:

```
similarity(A, B) = (A . B) / (||A|| * ||B||)
```

The result ranges from 0 for no weighted vocabulary overlap to 1 for identical
nonzero vectors. This is a lexical measure of similarity, rather than a
contextual embedding model, but it provides a useful semantic comparison
without external NLP libraries.

## Summary

The overall pipeline is:

```
raw text -> lowercase -> remove punctuation/numbers -> tokenize
         -> remove stopwords -> stem/lemmatize
         -> [POS tagging / morphology / syntax / semantic categories]
         -> feature extraction (BoW / one-hot / TF-IDF / n-grams)
         -> TF-IDF cosine similarity between documents
```

POS tagging, morphological analysis, syntactic analysis, and semantic category
analysis run on lightly cleaned tokens (before stopword removal and stemming),
since function words and full word forms carry the grammatical and lexical
signals those steps depend on. Each step reduces noise or adds structure,
turning unstructured text into a numeric form that can be compared, counted,
and eventually fed into a model.
