import io

from nltk.tree import Tree
import nltk_tree_ext.funcs as funcs
import pytest

RAW_TREES_WITH_BRANCHES = (
    (
        "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))))",
        (
            (("S", "NP", "PRP"), "My"),
            (("S", "NP", "NN"), "daughter"),
            (("S", "VP", "VBD"), "broke"),
            (("S", "VP", "NP", "NP", "DET"), "the"),
            (("S", "VP", "NP", "NP", "JJ"), "red"),
            (("S", "VP", "NP", "NP", "NN"), "toy"),
            (("S", "VP", "NP", "PP", "IN"), "with"),
            (("S", "VP", "NP", "PP", "NP", "DET"), "a"),
            (("S", "VP", "NP", "PP", "NP", "NN"), "hammer"),
        ),
    ),
)


@pytest.mark.parametrize("tree_raw, result", RAW_TREES_WITH_BRANCHES)
def test_iter_leaves_with_branches(tree_raw: str, result):
    tree_parsed = Tree.fromstring(tree_raw)

    assert tuple(funcs.iter_leaves_with_branches(tree_parsed)) == result


TREES_WITH_UNARIES = (
    (
        "(CONJ (NP-TMP (PP で)) (CONJ-PART-SUBWORD は))",
        "(CONJ (NP-TMP☆PP で) (CONJ-PART-SUBWORD は))",
    ),
    (
        "(CONJ (NP-TMP (PP (X で))) (CONJ-PART-SUBWORD は))",
        "(CONJ (NP-TMP☆PP☆X で) (CONJ-PART-SUBWORD は))",
    ),
)


@pytest.mark.parametrize("tree_unfold, tree_fold", TREES_WITH_UNARIES)
def test_merge_nonterminal_unary_nodes(tree_unfold, tree_fold):
    tree_unfold_parse = Tree.fromstring(tree_unfold)
    tree_fold_parse = Tree.fromstring(tree_fold)

    assert (
        funcs.merge_nonterminal_unary_nodes(
            tree_unfold_parse, concat=lambda x, y: f"{x}☆{y}"
        )
        == tree_fold_parse
    )


@pytest.mark.parametrize("tree_unfold, tree_fold", TREES_WITH_UNARIES)
def test_unfold_unary_nodes(tree_unfold, tree_fold):
    tree_unfold_parse = Tree.fromstring(tree_unfold)
    tree_fold_parse = Tree.fromstring(tree_fold)

    tree_fold_unfolded = funcs.unfold_nonterminal_unary_nodes(
        tree_fold_parse, splitter=lambda s: s.split("☆")
    )

    assert tree_unfold_parse == tree_fold_unfolded


TREES_DFUDS = (
    # (
    #     "(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))",
    #     None,
    #     "(5(.(,+)(40))(*(/2)(1(3-))))",
    # ),
    (
        "(S (NP (DT the) (NN cat)) (VP (VBZ is) (ADJP (JJ cute))))",
        {
            "JJ": "あ",
            "DT": "い",
            "cute": "う",
            "cat": "え",
            "the": "お",
            "VBZ": "か",
            "VP": "き",
            "S": "く",
            "NP": "け",
            "is": "こ",
            "NN": "さ",
            "ADJP": "し",
        },
        "(く(け(いお)(さえ))(き(かこ)(し(あう))))",
    ),
)


TREES_LEVENSHTEIN_RATIO = (
    (
        "(CONJ (NP-TMP (PP で)) (CONJ-PART-SUBWORD は))",
        "(CONJ (NP-TMP☆PP で) (CONJ-PART-SUBWORD は))",
        88.88888888888889,
        80.0,
    ),
    (
        "(CONJ (NP-TMP (PP (X で))) (CONJ-PART-SUBWORD は))",
        "(CONJ (NP-TMP☆PP☆X で) (CONJ-PART-SUBWORD は))",
        80.0,
        71.42857142857143,
    ),
)


@pytest.mark.parametrize(
    "this, other, expected_skeleton, expected_all", TREES_LEVENSHTEIN_RATIO
)
def test_levenshtein_ratio_skeletons(this, other, expected_skeleton, expected_all):
    this_parsed = Tree.fromstring(this)
    other_parsed = Tree.fromstring(other)

    assert (
        funcs.levenshtein_ratio_skeleton(this_parsed, other_parsed) == expected_skeleton
    )
    assert (
        funcs.levenshtein_ratio_skeleton_nodes_leaves(this_parsed, other_parsed)
        == expected_all
    )


TREES_TO_PRINT = ("(CONJ (NP-TMP (PP (X で))) (CONJ-PART-SUBWORD は))",)


@pytest.mark.parametrize("tree_raw", TREES_TO_PRINT)
def test_str_oneline(tree_raw: str):
    tree_parsed = Tree.fromstring(tree_raw)
    assert funcs.str_oneline(tree_parsed) == tree_raw
