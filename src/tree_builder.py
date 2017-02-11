from abc import ABCMeta, abstractmethod
import numpy as np

from tree import Tree
from ig import majority_value, choose_best_decision_attribute
from utils import create_attributes

class AbstractTreeBuilder:
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_trees(self, examples, targets):
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
                    subtree = self._learn(new_examples, new_attributes, new_binary_targets, size_examples, targets_range, attributes_range)
                    tree.add_child(i, subtree)
            return tree


class BasicTreeBuilder(AbstractTreeBuilder):
    def build_trees(self, examples, targets):
        trees = {}
        values = np.unique(targets)
        for value in values.flat:
            binary_targets = np.vectorize(lambda x: np.int8(x == value))(targets)
            trees[value] = \
                self._learn(examples, create_attributes(examples.shape), binary_targets, examples.shape[0])
        self.trees["ag"] = self._learn(examples, create_attributes(examples.shape), targets, examples.shape[0], values)
        return trees
