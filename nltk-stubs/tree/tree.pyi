from typing import (
    Any,
    Type,
    TextIO,
    Literal,
    Callable,
    Generic,
    TypeVar,
    Never,
    SupportsIndex,
    overload,
)
from collections.abc import Iterable, Iterator, Sequence

import nltk_tree_ext.funcs as funcs

__all__ = ["Tree", "NODE", "LEAF", "T"]

NODE = TypeVar("NODE")
NODE_NEW = TypeVar("NODE_NEW")
LEAF = TypeVar("LEAF")
LEAF_NEW = TypeVar("LEAF_NEW")
D = TypeVar("D")
R = TypeVar("R")
T = TypeVar("T", bound="Tree")

_INDEX = SupportsIndex | slice | Sequence[SupportsIndex]

class Tree(Generic[NODE, LEAF], list[Tree[NODE, LEAF] | LEAF]):
    # ------
    # Copied from nltk-tree-stubs
    # Start here:
    # ------
    def __init__(
        self, node: NODE, children: Iterable[Tree[NODE, LEAF] | LEAF] = ...
    ) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    def __lt__(self, other: Any) -> bool: ...
    def __mul__(self, v: Any) -> Never: ...
    def __rmul__(self, v: Any) -> Never: ...
    def __add__(self, v: Any) -> Never: ...
    def __radd__(self, v: Any) -> Never: ...
    def __getitem__(self, index: _INDEX) -> Tree[NODE, LEAF] | LEAF: ...
    def __setitem__(self, index: _INDEX, value: Tree[NODE, LEAF] | LEAF) -> None: ...
    def __delitem__(self, index: _INDEX) -> None: ...
    @property
    def node(self) -> NODE: ...
    def label(self) -> NODE: ...
    def set_label(self, label: NODE) -> None: ...
    def leaves(self) -> list[LEAF]: ...
    def flatten(self) -> Tree[NODE, LEAF]: ...
    def height(self) -> int: ...
    def treepositions(
        self,
        order: Literal["preorder", "postprder", "bothorder", "leaves"] = "preorder",
    ) -> list[tuple[int]]: ...
    def subtrees(
        self, filter: Callable[[Tree[NODE, LEAF]], bool] | None = None
    ) -> Iterator[Tree[NODE, LEAF]]: ...
    def productions(self) -> list["nltk.grammar.Production"]: ...
    def pos(self) -> list[tuple[NODE, LEAF]]: ...
    def leaf_treeposition(self, index: SupportsIndex) -> tuple[int, ...]: ...
    def treeposition_spanning_leaves(self, start: int, end: int) -> tuple[int, ...]: ...
    def chomsky_normal_form(
        self,
        factor: Literal["left", "right"] = "right",
        horzMarkov: int | None = None,
        vertMarkov: int | None = 0,
        childChar: str = "|",
        parentChar: str = "^",
    ) -> None: ...
    def un_chomsky_normal_form(
        self,
        expandUnary: bool = True,
        childChar: str = "|",
        parentChar: str = "^",
        unaryChar: str = "+",
    ) -> None: ...
    def collapse_unary(
        self, collapsePOS: bool = False, collapseRoot: bool = False, joinChar: str = "+"
    ) -> None: ...
    @classmethod
    def convert(cls: Type[T], tree: Any) -> T: ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def copy(self, deep: bool): ...
    def freeze(self, leaf_freezer: Callable[[NODE], Any] | None): ...
    @overload
    @classmethod
    def fromstring(
        cls,
        s: str,
        brackets: str = "()",
        read_node: None = None,
        read_leaf: None = None,
        node_pattern: str | None = None,
        leaf_pattern: str | None = None,
        remove_empty_top_bracketing: bool = False,
    ) -> Tree[str, str]: ...
    @overload
    @classmethod
    def fromstring(
        cls,
        s: str,
        brackets: str = "()",
        read_node: Callable[[str], NODE] = ...,
        read_leaf: None = None,
        node_pattern: str | None = None,
        leaf_pattern: str | None = None,
        remove_empty_top_bracketing: bool = False,
    ) -> Tree[NODE, str]: ...
    @overload
    @classmethod
    def fromstring(
        cls,
        s: str,
        brackets: str = "()",
        read_node: None = None,
        read_leaf: Callable[[str], LEAF] = ...,
        node_pattern: str | None = None,
        leaf_pattern: str | None = None,
        remove_empty_top_bracketing: bool = False,
    ) -> Tree[str, LEAF]: ...
    @overload
    @classmethod
    def fromstring(
        cls,
        s: str,
        brackets: str = "()",
        read_node: Callable[[str], NODE] = ...,
        read_leaf: Callable[[str], LEAF] = ...,
        node_pattern: str | None = None,
        leaf_pattern: str | None = None,
        remove_empty_top_bracketing: bool = False,
    ) -> Tree[NODE, LEAF]: ...
    @classmethod
    def fromlist(cls, l: Sequence[Any]) -> Tree[Any, Any]: ...
    def draw(self) -> None: ...
    def pretty_print(
        self,
        sentence: Sequence[str] | None = None,
        highlight: Sequence[Tree[NODE, LEAF]] = (),
        stream: TextIO | None = None,
        **kwargs
    ) -> None: ...
    def pprint(self, **kwargs) -> None: ...
    def pformat(
        self,
        margin: int = 70,
        indent: int = 0,
        nodesep: str = "",
        parens: str = "()",
        quotes: bool = False,
    ): ...
    def pformat_latex_qtree(self) -> str: ...
    # ------
    # End
    # ------

    # ------
    # Additional methods
    # ------

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
    def fold(
        self: Tree[NODE, LEAF],
        func: Callable[[NODE, Sequence[R]], R],
        init: Callable[[LEAF], R],
    ) -> R: ...
    @classmethod
    def fromlist_as_unary(
        cls: Type[Tree[NODE, LEAF]],
        nodes: Iterable[NODE],
        children: Iterable[Tree[NODE, LEAF] | LEAF],
    ) -> Tree[NODE, LEAF]: ...
    def inspect_unary(
        self, default: D = None
    ) -> tuple[NODE, Tree[NODE, LEAF] | LEAF] | D: ...
    def inspect_terminal(self, default: D = None) -> tuple[NODE, LEAF] | D: ...
    def inspect_unary_nonterminal(
        self, default: D = None
    ) -> tuple[NODE, Tree[NODE, LEAF]] | D: ...
    def iter_leaves_with_branches(
        self,
    ) -> Iterator[tuple[tuple[NODE, ...], LEAF]]: ...
    def overwrite_leaves(
        self,
        new_leaves: Iterator[LEAF_NEW],
    ) -> Tree[NODE, LEAF_NEW]: ...
    def merge_nonterminal_unary_nodes(
        self, concat: Callable[[NODE, NODE], NODE]
    ) -> Tree[NODE, LEAF]: ...
    def unfold_nonterminal_unary_nodes(
        self, splitter: Callable[[NODE], Sequence[NODE]] = lambda s: (s,)
    ): ...
    def to_tokens(
        self: Tree[NODE, LEAF]
    ) -> Iterator[
        Literal[funcs.TokenType.OPEN]
        | Literal[funcs.TokenType.CLOSE]
        | tuple[Literal[funcs.TokenType.NODE], NODE]
        | tuple[Literal[funcs.TokenType.LEAF], LEAF]
    ]: ...
    def encode_skeleton(self) -> str: ...
    def encode_skeleton_nodes_leaves(
        self, indices: dict[NODE | LEAF, str] | None = None
    ) -> tuple[str, dict[NODE | LEAF, str]]: ...
    def levenshtein_ratio_skeleton(self, other: Tree[NODE, LEAF]) -> float: ...
    def levenshtein_ratio_skeleton_nodes_leaves(
        self, other: Tree[NODE, LEAF]
    ) -> float: ...
