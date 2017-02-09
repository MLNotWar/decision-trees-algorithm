import numpy as np
import pandas as pd

from tree import Tree
from ig import majority_value, choose_best_decision_attribute
from utils import create_attributes


class DecisionTreeLearning:
    def __init__(self):
        super().__init__()

        self.trees = {}

    def fit(self, examples, targets):
        values = np.unique(targets)
        for value in values.flat:
            binary_targets = np.vectorize(lambda x: np.int8(x == value))(targets)

            self.trees[value] = \
                self._learn(examples,
                            create_attributes(examples.shape),
                            binary_targets)

    def predict(self, examples):
        n_rows, _ = examples.shape
        results = pd.DataFrame(index=range(n_rows), columns=self.trees.keys())

        for i in range(n_rows):
            for k, tree in self.trees.items():
                results[i, k] = self._predict_one(examples[i], tree)

        return results

    def _predict_one(self, example, tree):
        node = tree
        while not node.is_leaf():
            node = node.go(example[node.data])

        return node.data

    def _learn(self, examples, attributes, targets, targets_range=(0, 1)):
        """ returns a decision tree for a given target label
        """
        val, count = majority_value(targets)
        if count == targets.shape[0]:
            return Tree(val)
        elif not attributes.any():
            return Tree(val)
        else:
            best_attribute = choose_best_decision_attribute(examples, attributes, targets, targets_range)
            tree = Tree(best_attribute)
            for i in targets_range:
                mask = examples[:, best_attribute] == i
                new_examples = examples[mask]
                new_binary_targets = targets[mask]

                if len(new_examples) == 0:
                    tree.add_child(i, Tree(val))
                else:
                    new_attributes = attributes.copy()
                    new_attributes[:, best_attribute] = False
                    subtree = self._learn(new_examples, new_attributes, new_binary_targets)
                    tree.add_child(i, subtree)
            return tree
