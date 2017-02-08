from scipy import stats
import numpy as np


def short_mul(l_, r):
    if l_ == 0.:
        return 0.
    else:
        return l_ * r()


def majority_value(binary_targets):
    mode = stats.mode(binary_targets)
    return mode.mode[0][0], mode.count[0][0]


def entropy(binary_targets):
    size, _ = binary_targets.shape
    _, mode_size = majority_value(binary_targets)

    majority_proportion = float(mode_size) / float(size)
    minority_proportion = 1. - majority_proportion

    return - short_mul(majority_proportion, lambda: np.log2(majority_proportion)) \
           - short_mul(minority_proportion, lambda: np.log2(minority_proportion))


def information_gain(examples, binary_targets, attribute):
    size_targets, _ = binary_targets.shape

    ed = entropy(binary_targets)

    column = examples[:,attribute]
    mask1 = column == 1

    positive_proportion = float(np.count_nonzero(mask1)) / size_targets
    negative_proportion = 1. - positive_proportion

    return ed - short_mul(positive_proportion, lambda: entropy(binary_targets[mask1])) \
              - short_mul(negative_proportion, lambda: entropy(binary_targets[~mask1]))


def choose_best_decision_attribute(examples, attributes, binary_targets):
    best_attribute = None
    best_gain = float("-inf")

    for attribute in np.ndindex(attributes.shape):
        if attributes[attribute] == 0:
            continue

        ig = information_gain(examples, binary_targets, attribute[1])
        if ig > best_gain:
            best_attribute = attribute[1]
            best_gain = ig

    return best_attribute
