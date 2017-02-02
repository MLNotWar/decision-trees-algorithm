import numpy as np
from collections import Counter

import Tree

POSSIBLE_VALUES = 2

# returns a decision tree for a given target label
def decision_tree_learning(examples, attributes, binary_targets):
    if(all(binary_targets == binary_targets[0])):
        return Tree(binary_targets[0])
    elif(attributes.size() == 0):
        return Tree(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        tree = Tree(best_attribute)
        for i in range(POSSIBLE_VALUES)
            subtree = Tree()
            tree.add_child(subtree)
            new_examples, new_binary_targets = get_revelant_examples(examples, attributes, binary_targets, i)
            if(new_examples.size() == 0):
                return Tree(majority_value(binary_targets))
            else:
                new_attributes = attributes.remove(best_attribute) #TODO
                subtree = decision_tree_learning(new_examples, new_attributes, new_binary_targets)
    return tree

def majority_value(binary_values):
    data = Counter(binary_targets)
    return data.most_common(1)

def choose_best_decision_attribute(examples, attributes, binary_targets):
    #TODO

def get_revelant_examples(examples, attributes, binary_targets, i):
    #TODO
