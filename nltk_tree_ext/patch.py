from nltk.tree import Tree
import nltk_tree_ext.funcs as funcs
from nltk_tree_ext.funcs import TokenType

Tree.fromlist_as_unary = classmethod(funcs.fromlist_as_unary)
Tree.inspect_unary = funcs.inspect_unary
Tree.inspect_terminal = funcs.inspect_terminal
Tree.inspect_unary_nonterminal = funcs.inspect_unary_nonterminal
Tree.iter_leaves_with_branches = funcs.iter_leaves_with_branches
Tree.overwrite_leaves = funcs.overwrite_leaves
Tree.merge_nonterminal_unary_nodes = funcs.merge_nonterminal_unary_nodes
Tree.unfold_nonterminal_unary_nodes = funcs.unfold_nonterminal_unary_nodes
Tree.to_tokens = funcs.to_tokens
Tree.encode_skeleton = funcs.encode_skeleton
Tree.encode_skeleton_nodes_leaves = funcs.encode_skeleton_nodes_leaves
Tree.levenshtein_ratio_skeleton = funcs.levenshtein_ratio_skeleton
Tree.levenshtein_ratio_skeleton_nodes_leaves = funcs.levenshtein_ratio_skeleton_nodes_leaves