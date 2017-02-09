import numpy as np

from test.test_base import Test


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
        for i in range(self.n_folds):
            testing_mask = self.testing_masks[i]
            algorithm.fit(self.examples[~testing_mask], self.targets[~testing_mask])
            result = algorithm.predict(self.examples[testing_mask])

            print(result)

            exit(-1)

            # self._update_ftp(result, binary_targets[~testing_mask])

    def _update_ftp(self, prediction, actual):
        # TODO check validity of for loop
        for i in range(prediction[0]):
            if prediction[0][i] == 1 and prediction[0][i] == 1:
                self.n_true_positives += 1
            elif prediction[0][i] == 1 and prediction[0][i] == 0:
                self.n_false_positives += 1
            elif prediction[0][i] == 0 and prediction[0][i] == 0:
                self.n_true_negatives += 1
            elif prediction[0][i] == 0 and prediction[0][i] == 1:
                self.n_false_negatives += 1
