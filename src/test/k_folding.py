import numpy as np

from test.test_base import Test
from test.confusion_matrix import ConfusionMatrix


class KFoldTest(Test):
    def __init__(self, examples, targets, n_folds=10):
        super().__init__(examples, targets)

        self.n_folds = n_folds
        self.testing_masks = self.split_examples()

    def split_examples(self):
        testing_masks = []
        size, _ = self.examples.shape
        n_examples_per_fold = int(size / self.n_folds)

        for i in range(self.n_folds):
            mask = np.full((size,), False, dtype=bool)
            start = i * n_examples_per_fold

            if i == self.n_folds - 1:
                mask[start:-1] = True
            else:
                mask[start:start + n_examples_per_fold] = True

            testing_masks.append(mask)
        return testing_masks

    def evaluate(self, algorithm):
        confusion_matrix = ConfusionMatrix()
        for i in range(self.n_folds):
            testing_mask = self.testing_masks[i]
            algorithm.fit(self.examples[~testing_mask], self.targets[~testing_mask])

            predictions = algorithm.predict(self.examples[testing_mask])
            expectations = self.targets[testing_mask]
            confusion_matrix.update_data(predictions, expectations)
        confusion_matrix
