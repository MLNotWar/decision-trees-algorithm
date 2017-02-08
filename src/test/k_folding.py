from numpy import np

class KFoldTest(Test):
    def __init__(self, examples, binary_targets, n_folds=10):
        super.__init__(examples, binary_targets)

        self.n_folds = n_folds
        self.training_masks = self.split_examples()

    def split_examples(self):
        training_masks = []
        size, _ = self.examples.shape
        n_examples_per_fold = size / self.n_folds

        for i in range(self.n_folds):
            mask = np.full((size,), False, dtype=bool)
            start = i * n_examples_per_fold

            if i == self.n_folds - 1:
                mask[start, -1] = True
            else:
                mask[start, start + n_examples_per_fold] = True

            training_masks.append(mask)
        return training_masks

    def evaluate(self, algorithm):
