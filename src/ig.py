from scipy import stats
from math import log2
import numpy as np


def majority_value(binary_targets):
    mode = stats.mode(binary_targets)
    return tuple(map(lambda x: x[0], mode))


def entropy(binary_targets):
    size, _ = binary_targets.shape
    _, mode_size = majority_value(binary_targets)

    majority_proportion = float(mode_size) / float(size)
    minority_proportion = 1. - majority_proportion

    return - majority_proportion * log2(majority_proportion) \
           - minority_proportion * log2(minority_proportion)


def information_gain(examples, binary_targets, attribute):
    size_targets, _ = binary_targets.shape

    ed = entropy(binary_targets)

    column = examples[:,attribute]
    mask1 = column == 1

    positive_proportion = float(np.count_nonzero(mask1)) / size_targets
    negative_proportion = 1. - positive_proportion

    return ed - positive_proportion * entropy(binary_targets[mask1]) \
              - negative_proportion * entropy(binary_targets[~mask1])


def choose_best_decision_attribute(examples, attributes, binary_targets):
    best_attribute = None
    best_gain = float("-inf")

    for attribute in np.ndindex(attributes.shape):
        if attributes[attribute] == 0:
            continue

        ig = information_gain(examples, binary_targets, attribute)
        if ig > best_gain:
            best_attribute = attribute
            best_gain = ig

    return best_attribute
