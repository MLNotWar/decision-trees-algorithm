import numpy as np

from test.test_base import Test
from predictor import predict
from test.confusion_matrix import ConfusionMatrix


class KFoldTest(Test):
    def __init__(self, examples, targets, n_folds=10):
        super().__init__(examples, targets)

        self.n_folds = n_folds

    def split_examples(self):
        size, _ = self.examples.shape
        n_examples_per_fold = int(size / self.n_folds)

        for i in range(self.n_folds):
            mask = np.full((size,), False, dtype=bool)
            start = i * n_examples_per_fold

            if i == self.n_folds - 1:
                mask[start:-1] = True
            else:
                mask[start:start + n_examples_per_fold] = True

            yield mask

    def evaluate(self, algorithm, optimise=False):
        confusion_matrix = ConfusionMatrix()
        for testing_mask in self.split_examples():
            trees = algorithm.build_trees(self.examples[~testing_mask], self.targets[~testing_mask], optimise=optimise)
            predictions = predict(trees, self.examples[testing_mask])
            expectations = self.targets[testing_mask]

            confusion_matrix.update_data(predictions, expectations)
        return confusion_matrix
