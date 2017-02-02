import numpy as np

POSSIBLE_VALUES = 2

# returns a decision tree for a given target label
def decision_tree_learning(examples, attributes, binary_targets):
    if(all_equal(binary_targets)):
        # return leaf
    elif(attributes.size() == 0):
        return new leaf(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attibutes, binary_targets)
        tree = new tree
        for i in range(POSSIBLE_VALUES)
            tree.add(i)
            new_examples, new_binary_targets = get_revelant_examples(examples, attributes, binary_targets, i)
            if(new_examples.size() == 0):
                return new leaf(majority_value(binary_targets))
            else:
                new_attributes = attributes.remove(best_attribute)
                tree.subtree(i) = decision_tree_learning(new_examples, new_attributes, new_binary_targets)
    return tree
