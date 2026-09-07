"""Syntactic analysis via a small hand-written constituency parser.

Pipeline: tokens -> POS tags (linguistic_features.pos_tag) -> parse tree
(this module) -> extracted subject/verb/object, POS counts, and
dependency-style relations.

The parser is a recursive-descent chunker over a tiny context-free grammar,
not a statistical or trained parser:

    S  -> NP? VP
    NP -> PRON | DET? ADJ* NOUN+
    VP -> AUX? ADV? (VERB|AUX) NP? PP*
    PP -> PREP NP

It's still a heuristic (it can't resolve every sentence), but it builds an
actual tree structure instead of guessing subject/object from raw token
positions, so features like clause counts and dependency relations can be
read off the tree.
"""

import pandas as pd


NOMINAL_TAGS = {"NOUN", "PRON"}
VERBAL_TAGS = {"VERB", "AUX"}


class Node:
    """A constituent in a parse tree: either a labeled phrase with child
    Nodes, or a leaf wrapping a single (word, tag) pair."""

    __slots__ = ("label", "children", "leaf")

    def __init__(self, label, children=None, leaf=None):
        self.label = label
        self.children = children or []
        self.leaf = leaf

    def is_leaf(self):
        return self.leaf is not None

    def leaves(self):
        """Yield every (word, tag) leaf under this node, left to right."""
        if self.is_leaf():
            yield self.leaf
        else:
            for child in self.children:
                yield from child.leaves()

    def find_all(self, label):
        """Yield every descendant node (including self) with this label."""
        if self.label == label:
            yield self
        for child in self.children:
            if not child.is_leaf():
                yield from child.find_all(label)

    def __repr__(self):
        if self.is_leaf():
            return f"{self.label}({self.leaf[0]})"
        inner = " ".join(repr(child) for child in self.children)
        return f"({self.label} {inner})"


def render_tree(node, indent=0):
    """Render a parse tree as a human-readable, indented multi-line string."""
    pad = "  " * indent
    if node.is_leaf():
        word, tag = node.leaf
        return f"{pad}{node.label}: {word!r}"
    lines = [f"{pad}{node.label}"]
    for child in node.children:
        lines.append(render_tree(child, indent + 1))
    return "\n".join(lines)


def _np_head(np_node):
    """Return the head word of a noun phrase: its rightmost noun, or its
    pronoun for a pronoun-only NP."""
    head = None
    for child in np_node.children:
        if child.label in ("NOUN", "PRON"):
            head = child.leaf[0]
    return head


def _parse_np(tokens, index):
    """Consume a noun phrase: PRON | DET? ADJ* NOUN+."""
    start = index
    children = []

    if index < len(tokens) and tokens[index][1] == "PRON":
        children.append(Node("PRON", leaf=tokens[index]))
        return Node("NP", children), index + 1

    if index < len(tokens) and tokens[index][1] == "DET":
        children.append(Node("DET", leaf=tokens[index]))
        index += 1

    while index < len(tokens) and tokens[index][1] == "ADJ":
        children.append(Node("ADJ", leaf=tokens[index]))
        index += 1

    noun_start = index
    while index < len(tokens) and tokens[index][1] == "NOUN":
        children.append(Node("NOUN", leaf=tokens[index]))
        index += 1

    if index == noun_start:
        # No head noun found (e.g. a bare determiner) -> not a valid NP.
        return None, start

    return Node("NP", children), index


def _parse_pp(tokens, index):
    """Consume a prepositional phrase: PREP NP."""
    if index >= len(tokens) or tokens[index][1] != "PREP":
        return None, index
    prep_node = Node("PREP", leaf=tokens[index])
    np_node, next_index = _parse_np(tokens, index + 1)
    if np_node is None:
        return None, index
    return Node("PP", [prep_node, np_node]), next_index


def _parse_vp(tokens, index):
    """Consume a verb phrase: AUX? ADV? (VERB|AUX) NP? PP*."""
    start = index
    children = []

    if index < len(tokens) and tokens[index][1] == "AUX":
        children.append(Node("AUX", leaf=tokens[index]))
        index += 1

    if index < len(tokens) and tokens[index][1] == "ADV":
        children.append(Node("ADV", leaf=tokens[index]))
        index += 1

    if index < len(tokens) and tokens[index][1] in VERBAL_TAGS:
        children.append(Node("VERB", leaf=tokens[index]))
        index += 1
    else:
        return None, start

    np_node, after_np = _parse_np(tokens, index)
    if np_node is not None:
        children.append(np_node)
        index = after_np

    while True:
        pp_node, after_pp = _parse_pp(tokens, index)
        if pp_node is None:
            break
        children.append(pp_node)
        index = after_pp

    return Node("VP", children), index


def _parse_clause(tokens, index):
    """Consume a single clause: NP? VP."""
    np_node, after_np = _parse_np(tokens, index)
    start_index = after_np if np_node is not None else index

    vp_node, after_vp = _parse_vp(tokens, start_index)
    if vp_node is None:
        return None, index

    children = [child for child in (np_node, vp_node) if child is not None]
    return Node("S", children), after_vp


def build_parse_tree(tagged_tokens):
    """Build a shallow constituency parse tree from POS-tagged tokens.

    Runs a recursive-descent chunker clause by clause, splitting on
    coordinating conjunctions. Tokens that don't fit the grammar (stray
    conjunctions, fragments with no verb) are kept as flat leaves under the
    root so no input is silently dropped.
    """
    tokens = list(tagged_tokens)
    top_level = []
    index = 0
    while index < len(tokens):
        clause_node, next_index = _parse_clause(tokens, index)
        if clause_node is not None and next_index > index:
            top_level.append(clause_node)
            index = next_index
            if index < len(tokens) and tokens[index][1] == "CONJ":
                top_level.append(Node("CONJ", leaf=tokens[index]))
                index += 1
        else:
            top_level.append(Node(tokens[index][1], leaf=tokens[index]))
            index += 1

    return Node("ROOT", top_level)


def subject_verb_object(tagged_tokens):
    """Extract subject, verb, and object candidates by parsing the tokens
    into a tree and reading them off its first clause: the head of the
    clause's NP is the subject, the first verbal leaf in its VP is the
    verb, and the head of the first NP found inside that VP (a direct
    object or a preposition's object) is the object.
    """
    tree = build_parse_tree(tagged_tokens)
    clause = next(tree.find_all("S"), None)
    if clause is None:
        return {"subject": None, "verb": None, "object": None}

    subject = None
    verb = None
    object_word = None

    for child in clause.children:
        if child.label == "NP" and subject is None:
            subject = _np_head(child)
        elif child.label == "VP":
            verb = next((w for w, tag in child.leaves() if tag in VERBAL_TAGS), None)
            inner_np = next(child.find_all("NP"), None)
            if inner_np is not None:
                object_word = _np_head(inner_np)

    return {"subject": subject, "verb": verb, "object": object_word}


def extract_dependencies(tree):
    """Read simple dependency relations off a parse tree: `det` and `amod`
    within noun phrases, `nsubj`/`dobj` between a clause's verb and its
    surrounding noun phrases, and `pobj` between a preposition and its
    object. This approximates a dependency parse using the constituency
    tree rather than parsing dependencies directly."""
    relations = []

    for np_node in tree.find_all("NP"):
        head = _np_head(np_node)
        if head is None:
            continue
        for child in np_node.children:
            if child.label == "DET":
                relations.append({"relation": "det", "head": head, "dependent": child.leaf[0]})
            elif child.label == "ADJ":
                relations.append({"relation": "amod", "head": head, "dependent": child.leaf[0]})

    for clause in tree.find_all("S"):
        subject_np = next((c for c in clause.children if c.label == "NP"), None)
        vp_node = next((c for c in clause.children if c.label == "VP"), None)
        if vp_node is None:
            continue

        verb = next((w for w, tag in vp_node.leaves() if tag in VERBAL_TAGS), None)
        if verb is None:
            continue

        if subject_np is not None:
            subject_head = _np_head(subject_np)
            if subject_head is not None:
                relations.append({"relation": "nsubj", "head": verb, "dependent": subject_head})

        direct_object_np = next((c for c in vp_node.children if c.label == "NP"), None)
        if direct_object_np is not None:
            object_head = _np_head(direct_object_np)
            if object_head is not None:
                relations.append({"relation": "dobj", "head": verb, "dependent": object_head})

        for pp_node in vp_node.find_all("PP"):
            prep_node = next((c for c in pp_node.children if c.label == "PREP"), None)
            pp_np = next((c for c in pp_node.children if c.label == "NP"), None)
            if prep_node is not None and pp_np is not None:
                pobj_head = _np_head(pp_np)
                if pobj_head is not None:
                    relations.append({
                        "relation": "pobj", "head": prep_node.leaf[0], "dependent": pobj_head,
                    })

    return relations


def syntactic_analysis(tagged_tokens):
    """Return sentence-level syntax features from a list of (word, POS) pairs."""
    tree = build_parse_tree(tagged_tokens)
    structure = subject_verb_object(tagged_tokens)
    tags = [tag for _, tag in tagged_tokens]
    dependencies = extract_dependencies(tree)
    clause_count = sum(1 for _ in tree.find_all("S"))

    return pd.DataFrame([{
        **structure,
        "token_count": len(tagged_tokens),
        "noun_count": sum(tag == "NOUN" for tag in tags),
        "verb_count": sum(tag in VERBAL_TAGS for tag in tags),
        "adj_count": sum(tag == "ADJ" for tag in tags),
        "clause_count": clause_count,
        "has_subject_verb_object": all(structure.values()),
        "dependency_count": len(dependencies),
        "dependencies": dependencies,
    }])


def syntactic_feature_matrix(tagged_documents):
    """Return numeric syntactic features with one row for each document."""
    feature_rows = []
    for tagged_tokens in tagged_documents:
        analysis = syntactic_analysis(tagged_tokens)
        feature_rows.append(analysis.drop(columns=["subject", "verb", "object", "dependencies"]))
    return pd.concat(feature_rows, ignore_index=True)