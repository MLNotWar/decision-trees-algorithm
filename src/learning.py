import numpy as np

from tree import Tree
from ig import majority_value, choose_best_decision_attribute
from utils import create_attributes


class DecisionTreeLearning:
    def __init__(self):
        super().__init__()

        self.tree = None

    def fit(self, examples, binary_targets):
        self.tree = self._learn(examples,
                                create_attributes(examples.shape),
                                binary_targets)
        self.optimise()

    def predict(self, examples):
        pass # TODO

    def optimise(self):
        self._prune()

    def _prune(self):
        pass

    def _learn(self, examples, attributes, binary_targets):
        """ returns a decision tree for a given target label
        """
        if np.all(binary_targets == binary_targets[0][0]):
            return Tree(binary_targets[0][0])
        elif not attributes.any():
            return Tree(majority_value(binary_targets)[0])
        else:
            best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
            tree = Tree(best_attribute)
            for i in (0, 1):
                mask = examples[:,best_attribute] == i
                new_examples = examples[mask]
                new_binary_targets = binary_targets[mask]

                if len(new_examples) == 0:
                    tree.add_child(i, Tree(majority_value(binary_targets)[0]))
                else:
                    new_attributes = attributes.copy()
                    new_attributes[:,best_attribute] = False
                    subtree = self._learn(new_examples, new_attributes, new_binary_targets)
                    tree.add_child(i, subtree)
            return tree
