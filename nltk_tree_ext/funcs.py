from enum import IntEnum
import itertools
from typing import Literal, Sequence, Type, TypeVar, Callable, overload
from collections.abc import Iterable, Iterator

import rapidfuzz

from nltk.tree import Tree

NODE = TypeVar("NODE")
NODE_NEW = TypeVar("NODE_NEW")
LEAF = TypeVar("LEAF")
LEAF_NEW = TypeVar("LEAF_NEW")
D = TypeVar("D")
R = TypeVar("R")


@overload
def map(
    self: Tree[NODE, LEAF],
    func_node: None = None,
    func_leaf: None = None,
) -> Tree[NODE, LEAF]: ...
@overload
def map(
    self: Tree[NODE, LEAF],
    func_node: None = None,
    func_leaf: Callable[[LEAF], LEAF_NEW] = lambda x: x,
) -> Tree[NODE, LEAF_NEW]: ...
@overload
def map(
    self: Tree[NODE, LEAF],
    func_node: Callable[[NODE], NODE_NEW],
    func_leaf: None = None,
) -> Tree[NODE_NEW, LEAF]: ...
@overload
def map(
    self: Tree[NODE, LEAF],
    func_node: Callable[[NODE], NODE_NEW],
    func_leaf: Callable[[LEAF], LEAF_NEW],
) -> Tree[NODE_NEW, LEAF_NEW]: ...


def map(
    self,
    func_node=None,
    func_leaf=None,
):
    """
    Map the nodes and leaves of the tree.
    """
    return Tree(
        func_node(label := self.label()) if func_node else self.label(),
        [
            (
                map(child, func_node, func_leaf)
                if isinstance(child, Tree)
                else func_leaf(child) if func_leaf else child
            )
            for child in self
        ],
    )


def fold(
    self: Tree[NODE, LEAF],
    func: Callable[[NODE, Sequence[R]], R],
    init: Callable[[LEAF], R],
) -> R:
    """
    Fold the tree.
    """
    return func(
        self.label(),
        [
            (fold(child, func, init) if isinstance(child, Tree) else init(child))
            for child in self
        ],
    )


def fromlist_as_unary(
    cls: Type[Tree[NODE, LEAF]],
    nodes: Iterable[NODE],
    children: Iterable[Tree[NODE, LEAF] | LEAF],
) -> Tree[NODE, LEAF]:
    """
    Create a tree from `nodes`, a list of node labels, with each node as a unary child of the previous node.
    The children of the last node are given by `children`.

    `nodes` cannot be empty.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.fromlist_as_unary = fromlist_as_unary
    >>> tree = Tree.fromlist_as_unary("NP VP S".split(), ["cat"])
    >>> print(tree)
    ... (NP (VP (S cat)))
    """
    iter_nodes = iter(nodes)
    tree: Tree[NODE, LEAF] = cls(next(iter_nodes), [])
    tree_pointer = tree
    for node in iter_nodes:
        child: Tree[NODE, LEAF] = cls(node, [])
        tree_pointer.append(child)
        tree_pointer = child

    tree_pointer.extend(children)
    return tree


def inspect_unary(
    self: Tree[NODE, LEAF], default: D = None
) -> tuple[NODE, Tree[NODE, LEAF] | LEAF] | D:
    """
    Inspect and destructure a unary tree.
    `None` if the tree is not unary.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.inspect_unary = inspect_unary
    >>> tree1 = Tree.fromstring("(NP (DT the) (NN cat))")
    >>> tree1.inspect_unary(default=None)
    ... None
    >>> tree2 = Tree.fromstring("(NN (C cat))")
    >>> tree2.inspect_unary(default=None)
    ... ('NN', Tree('C', ['cat']))

    """
    if len(self) == 1:
        return self.label(), self[0]
    else:
        return default


def inspect_terminal(
    self: Tree[NODE, LEAF], default: D = None
) -> tuple[NODE, LEAF] | D:
    """
    Inspect and destructure a terminal tree.
    `default` is returned if the tree is not preterminal.

    Notes
    -----
    A terminal tree is a unary tree with a leaf (a lexical node) as its child.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.inspect_terminal = inspect_terminal
    >>> tree1 = Tree.fromstring("(NP (DT the) (NN cat))")
    >>> tree1.inspect_terminal(default=None)
    ... None
    >>> tree2 = Tree.fromstring("(NN cat)")
    >>> tree2.inspect_terminal(default=None)
    ... ('NN', 'cat')
    """
    if (res := inspect_unary(self, None)) and not isinstance(res[1], Tree):
        return res[0], res[1]
    else:
        return default


def inspect_unary_nonterminal(
    self: Tree[NODE, LEAF], default: D = None
) -> tuple[NODE, Tree[NODE, LEAF]] | D:
    """
    Inspect and destructure a nonterminal unary tree.
    `default` is returned if the tree is not preterminal.

    Notes
    -----
    A nonterminal unary tree is a unary tree with its child being another `Tree`.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.inspect_unary_nonterminal = inspect_unary_nonterminal
    >>> tree1 = Tree.fromstring("(NP (DT the) (NN cat))")
    >>> tree1.inspect_unary_nonterminal(default=None)
    ... None
    >>> tree2 = Tree.fromstring("(NN (C cat))")
    >>> tree2.inspect_unary_nonterminal(default=None)
    ... ('NN', Tree('C', ['cat']))
    """
    if (res := inspect_unary(self, None)) and isinstance(res[1], Tree):
        return res[0], res[1]
    else:
        return default


def iter_nodes_depth_first(self: Tree[NODE, LEAF]) -> Iterator[NODE]:
    """
    Iterate over the nodes of the tree in depth-first order.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.iter_nodes_depth_first = iter_nodes_depth_first
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> for node in tree.iter_nodes_depth_first():
    >>>     print(node)
    ... S
    ... NP
    ... DT
    ... NN
    ... VP
    ... VBZ
    ... ADJP
    ... JJ
    """
    pointer_stack: list[tuple[Tree[NODE, LEAF], int]] = [(self, 0)]
    while pointer_stack:
        current_node, child_pointer = pointer_stack.pop()

        if child_pointer < len(current_node):
            child = current_node[child_pointer]
            if isinstance(child, Tree):
                pointer_stack.append((current_node, child_pointer + 1))
                pointer_stack.append((child, 0))
            yield current_node.label()
        # else:
        # do nothing


def iter_nodes_leaves_depth_first(self: Tree[NODE, LEAF]) -> Iterator[NODE | LEAF]:
    """
    Iterate over the nodes of the tree in depth-first order.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.iter_nodes_depth_first = iter_nodes_depth_first
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> for node in tree.iter_nodes_depth_first():
    >>>     print(node)
    ... S
    ... NP
    ... DT
    ... the
    ... NN
    ... cat
    ... VP
    ... VBZ
    ... is
    ... ADJP
    ... JJ
    ... cute
    """
    pointer_stack: list[tuple[Tree[NODE, LEAF] | LEAF, int]] = [(self, 0)]
    while pointer_stack:
        current_node, child_pointer = pointer_stack.pop()

        if isinstance(current_node, Tree):
            if child_pointer < len(current_node):
                child = current_node[child_pointer]
                if isinstance(child, Tree):
                    pointer_stack.append((current_node, child_pointer + 1))
                    pointer_stack.append((child, 0))
                else:
                    yield child
                yield current_node.label()
        else:
            yield current_node
        # else:
        # do nothing


def iter_leaves_with_branches(
    self: Tree[NODE, LEAF],
) -> Iterator[tuple[tuple[NODE, ...], LEAF]]:
    """
    Iterate over the leaves of the tree with the branches leading to them.

    Yields
    ------
    tuple[tuple[NODE, ...], LEAF]
        The branch and the leaf.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.iter_leaves_with_branches = iter_leaves_with_branches
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> for branch, leaf in tree.iter_leaves_with_branches():
    >>>     print(branch, leaf)
    ... ('S', 'NP', 'DT') the
    ... ('S', 'NP', 'NN') cat
    ... ('S', 'VP', 'VBZ') is
    ... ('S', 'VP', 'ADJP', 'JJ') cute
    """
    pointer_stack: list[tuple[Tree[NODE, LEAF], int]] = [(self, 0)]
    while pointer_stack:
        current_node, child_pointer = pointer_stack.pop()

        if child_pointer < len(current_node):
            child = current_node[child_pointer]
            if isinstance(child, Tree):
                pointer_stack.append((current_node, child_pointer + 1))
                pointer_stack.append((child, 0))
            else:
                yield (
                    (
                        *(node.label() for node, _ in pointer_stack),
                        current_node.label(),
                    ),
                    child,
                )
        # else:
        # do nothing


def overwrite_leaves(
    self: Tree[NODE, LEAF],
    new_leaves: Iterator[LEAF_NEW],
) -> Tree[NODE, LEAF_NEW]:
    """
    Overwrite the leaves of the tree with `new_leaves`.

    Notes
    -----
    * Non-destructive.
    * If the number of `new_leaves` is less than the number of leaves in the tree, an exception will be raised.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.overwrite_leaves = overwrite_leaves
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> tree.overwrite_leaves("a b c d".split())
    ... (S (NP (DT a) (NN b)) (VP (VBZ c) (ADJP (JJ d))))
    """
    return Tree(
        self.label(),
        [
            (
                overwrite_leaves(child, new_leaves)
                if isinstance(child, Tree)
                else next(new_leaves)
            )
            for child in self
        ],
    )


def merge_nonterminal_unary_nodes(
    self: Tree[NODE, LEAF], concat: Callable[[NODE, NODE], NODE]
) -> Tree[NODE, LEAF]:
    """
    Merge unary nodes with their parent nodes.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.merge_unary_nodes = merge_unary_nodes
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> print(tree.merge_unary_nodes(concat=lambda x, y: f"{x}_{y}"))
    ... (S (NP (DT the) (NN cat)) (VBZ is) (ADJP_JJ cute))
    """
    if res := inspect_unary_nonterminal(self, None):
        parent, child = res

        tree_merged = Tree(
            concat(parent, child.label()),
            list(child),
        )

        return merge_nonterminal_unary_nodes(tree_merged, concat)
    else:
        return Tree(
            self.label(),
            [
                (
                    merge_nonterminal_unary_nodes(child, concat)
                    if isinstance(child, Tree)
                    else child
                )
                for child in self
            ],
        )


def unfold_nonterminal_unary_nodes(
    self: Tree[NODE, LEAF], splitter: Callable[[NODE], Sequence[NODE]] = lambda s: (s,)
) -> Tree[NODE, LEAF]:
    """
    Unfold unary nodes with multiple labels.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.unfold_nonterminal_unary_nodes = unfold_nonterminal_unary_nodes
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP_JJ cute)))")
    >>> print(tree.unfold_nonterminal_unary_nodes(splitter=lambda s: s.split("_")))
    ... (S (NP (DT the) (NN cat)) (VBZ is) (ADJP_JJ cute))
    """
    label = self.label()
    label_split = splitter(label)
    if len(label_split) <= 1:
        return Tree(
            label,
            [
                (
                    unfold_nonterminal_unary_nodes(child, splitter)
                    if isinstance(child, Tree)
                    else child
                )
                for child in self
            ],
        )
    else:
        return fromlist_as_unary(
            Tree,
            label_split,
            [
                (
                    unfold_nonterminal_unary_nodes(child, splitter)
                    if isinstance(child, Tree)
                    else child
                )
                for child in self
            ],
        )


class TokenType(IntEnum):
    OPEN = 0
    CLOSE = 1
    NODE = 2
    LEAF = 3


def to_tokens(
    self: Tree[NODE, LEAF]
) -> Iterator[
    Literal[TokenType.OPEN]
    | Literal[TokenType.CLOSE]
    | tuple[Literal[TokenType.NODE], NODE]
    | tuple[Literal[TokenType.LEAF], LEAF]
]:
    """
    Convert the tree back to a sequence of tokens used in parsing.

    Yields
    ------
    token
        Either `TokenType.OPEN`, `TokenType.CLOSE`, `TokenType.NODE` with a node label, or `TokenType.LEAF` with a leaf.
    """

    def _go(
        node: Tree[NODE, LEAF] | LEAF
    ) -> Iterator[
        Literal[TokenType.OPEN]
        | Literal[TokenType.CLOSE]
        | tuple[Literal[TokenType.NODE], NODE]
        | tuple[Literal[TokenType.LEAF], LEAF]
    ]:
        if isinstance(node, Tree):
            yield TokenType.OPEN
            yield TokenType.NODE, node.label()
            for child in node:
                yield from _go(child)
            yield TokenType.CLOSE
        else:
            yield TokenType.LEAF, node

    return _go(self)


ORD_START = ord(")") + 1  # 43


def encode_skeleton(self: Tree[NODE, LEAF]) -> str:
    """
    Extract the skeleton of the tree.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.encode_skeleton = encode_skeleton
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> tree.encode_skeleton()
    ... "((.)(.))((.)(.)(.))"
    """
    return "".join(
        (
            "("
            if token == TokenType.OPEN
            else (
                ")"
                if token == TokenType.CLOSE
                else "." if token[0] == TokenType.LEAF else ""
            )
        )
        for token in to_tokens(self)
    )


def encode_skeleton_nodes_leaves(
    self: Tree[NODE, LEAF], indices: dict[NODE | LEAF, str] | None = None
) -> tuple[str, dict[NODE | LEAF, str]]:
    """
    Encode the skeleton of the tree with the nodes and leaves replaced by placeholders.

    Arguments
    ---------
    indices
        A dictionary mapping nodes and leaves to their placeholders.
        Placehoders must be each a single character whose ordinal is greater than `ord(")")` (=42).
        Otherwise the result is undefined.
        If `None`, the dictionary is constructed on the fly.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.encode_skeleton_nodes_leaves = encode_skeleton_nodes_leaves
    >>> tree = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> enc, indices = tree.encode_skeleton_nodes_leaves()
    >>> print(enc)
    ... (5(.(,+)(40))(*(/2)(1(3-))))
    >>> print(indices)
    ... {'JJ': '*', 'DT': '+', 'cute': ',', 'cat': '-', 'the': '.', 'VBZ': '/', 'VP': '0', 'S': '1', 'NP': '2', 'is': '3', 'NN': '4', 'ADJP': '5'}
    """
    tokens = tuple(to_tokens(self))
    if indices is None:
        node_leaf_set = set(token[1] for token in tokens if isinstance(token, tuple))
        indices = {nl: chr(ORD_START + i) for i, nl in enumerate(node_leaf_set)}
    return (
        "".join(
            (
                "("
                if token == TokenType.OPEN
                else ")" if token == TokenType.CLOSE else indices[token[1]]
            )
            for token in to_tokens(self)
        ),
        indices,
    )


def levenshtein_ratio_skeleton(
    self: Tree[NODE, LEAF], other: Tree[NODE, LEAF]
) -> float:
    """
    Approximately measure the "similarity" between two trees by using the Levenshtein distance between the serializations of the skeletons of two trees.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.levenshtein_ratio_skeleton = levenshtein_ratio_skeleton
    >>> tree1 = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> tree2 = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> tree1.levenshtein_ratio_skeleton(tree2)
    ... 100
    """
    return rapidfuzz.fuzz.ratio(encode_skeleton(self), encode_skeleton(other))


def levenshtein_ratio_skeleton_nodes_leaves(
    self: Tree[NODE, LEAF], other: Tree[NODE, LEAF]
) -> float:
    """
    Approximately measure the "similarity" between two trees by using the Levenshtein distance between the serializations of the nodes, leaves and skeletons of two trees.

    Examples
    --------
    >>> from nltk.tree import Tree
    >>> Tree.levenshtein_ratio_skeleton = levenshtein_ratio_skeleton
    >>> tree1 = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> tree2 = Tree.fromstring("(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))")
    >>> tree1.levenshtein_ratio_skeleton(tree2)
    ... 100
    """
    indices_set = set(
        itertools.chain(
            iter_nodes_leaves_depth_first(self), iter_nodes_leaves_depth_first(other)
        )
    )
    indices = {nl: chr(ORD_START + i) for i, nl in enumerate(indices_set)}
    enc1, _ = encode_skeleton_nodes_leaves(self, indices)
    enc2, _ = encode_skeleton_nodes_leaves(other, indices)
    return rapidfuzz.fuzz.ratio(enc1, enc2)
