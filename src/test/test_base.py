from utils import create_attributes

METRIC_F1 = 1
METRIC_RECALL = 1 << 1
METRIC_PRECISION = 1 << 2
METRIC_ERROR_MEAN_SQUARED = 1 << 3


class Test:
    def __init__(self, examples, targets):
        self.attributes = create_attributes(examples.shape)

        self.examples = examples
        self.targets = targets

        self.n_true_positives = 0
        self.n_false_positives = 0
        self.n_true_negatives = 0
        self.n_false_negatives = 0

    def evaluate(self, *args, **options):
        raise RuntimeError("Not implemented")

    def stats(self, metrics):
        pass


"""
  > test = ...Test()
  > test.evaluate(algo)
  > test.stats(METRIC_PRECISION | METRIC_ERROR_MEAN_SQUARED)

  > test.evaluate(algo2)
  > test.stats(...)
"""
