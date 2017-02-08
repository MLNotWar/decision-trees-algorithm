from tree import Tree
from ig import majority_value, choose_best_decision_attribute


def decision_tree_learning(examples, attributes, binary_targets):
    """ returns a decision tree for a given target label
    """
    if all(binary_targets == binary_targets[0]):
        return Tree(binary_targets[0])
    elif attributes.size() == 0:
        return Tree(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        tree = Tree(best_attribute)
        for i in (0, 1):
            mask = examples[:,best_attribute] == i
            new_examples = examples[mask]
            new_binary_targets = binary_targets[mask]

            if new_examples.size() == 0:
                tree.add_child(i, Tree(majority_value(binary_targets)))
            else:
                new_attributes = attributes.copy()
                new_attributes[best_attribute] = False
                subtree = decision_tree_learning(new_examples, new_attributes, new_binary_targets)
                tree.add_child(i, subtree)
    return tree

