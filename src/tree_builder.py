from abc import ABCMeta, abstractmethod
import numpy as np
from random import randint

from tree import Tree
from ig import majority_value, choose_best_decision_attribute
from utils import create_attributes
from predictor import score_predictions


class AbstractTreeBuilder:
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_trees(self, examples, targets, optimise=False):
        return {}

    def _learn(self, examples, attributes, targets, size_examples,
               targets_range=(0, 1), attributes_range=(0, 1)):
        """ returns a decision tree for a given target label
        """
        val, count = majority_value(targets)
        error_margin = int(size_examples * 0.015)
        if count + error_margin >= targets.shape[0]:
            return Tree(val)
        elif not attributes.any():
            return Tree(val)
        else:
            best_attribute = choose_best_decision_attribute(examples, attributes, targets, targets_range)
            tree = Tree(best_attribute)
            for i in attributes_range:
                mask = examples[:, best_attribute] == i
                new_examples = examples[mask]
                new_binary_targets = targets[mask]

                if len(new_examples) == 0:
                    tree.add_child(i, Tree(val))
                else:
                    new_attributes = attributes.copy()
                    new_attributes[:, best_attribute] = False
                    subtree = self._learn(new_examples, new_attributes, new_binary_targets, size_examples,
                                          targets_range, attributes_range)
                    tree.add_child(i, subtree)
            return tree


class BasicTreeBuilder(AbstractTreeBuilder):
    def build_trees(self, examples, targets, optimise=False):
        trees = {}
        values = np.unique(targets)
        for value in values.flat:
            binary_targets = np.vectorize(lambda x: np.int8(x == value))(targets)
            trees[value] = \
                self._learn(examples, create_attributes(examples.shape), binary_targets,
                            examples.shape[0] if optimise else 0)
        trees["ag"] = self._learn(examples, create_attributes(examples.shape), targets,
                                  examples.shape[0] if optimise else 0, values)
        return trees


class PrunedTreeBuilder(AbstractTreeBuilder):
    def build_trees(self, examples, targets, optimise=False):
        size, _ = examples.shape
        validation_size = int(size / 10)
        start = randint(0, (size - validation_size))
        mask = np.full((size,), False, dtype=bool)
        mask[start:start + validation_size] = True

        trees = {}
        tr_ex = examples[~mask]
        va_ex = examples[mask]
        tr_ta = targets[~mask]
        va_ta = targets[mask]

        values = np.unique(targets)
        for value in values.flat:
            bi_ta = np.vectorize(lambda x: np.int8(x == value))(tr_ta)
            tree = self._learn(tr_ex, create_attributes(tr_ex.shape), bi_ta, size if optimise else 0)
            bi_ta = np.vectorize(lambda x: np.int8(x == value))(va_ta)
            trees[value] = self.prune_tree(tree, va_ex, bi_ta)
        tree = self._learn(tr_ex, create_attributes(tr_ex.shape), tr_ta, size if optimise else 0, values)
        trees["ag"] = self.prune_tree(tree, va_ex, va_ta)

        return trees

    def prune_tree(self, tree, examples, targets, targets_range=(0, 1)):
        max_index = -1
        max_score = score_predictions(tree, examples, targets)

        index = 0
        for t in tree:
            if t.is_leaf():
                continue
            for i in targets_range:
                t.pruned = i
                score = score_predictions(tree, examples, targets)
                if score > max_score:
                    max_index = index
                    max_score = score
                index += 1
            del t.pruned

        if max_index == -1:
            return tree

        index = max_index / 2
        for i, t in enumerate(tree):
            if not t.is_leaf() and i == index:
                t.data = max_index % 2
                t.children = {}
                return self.prune_tree(t, examples, targets, targets_range)
