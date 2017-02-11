import numpy as np
import pandas as pd
from collections import defaultdict

from tree import Tree
from tree_builder import BasicTreeBuilder


class DecisionTreeLearning:
    def __init__(self, builder=BasicTreeBuilder()):
        super().__init__()
        self.trees = {}
        self.builder = builder
        self.last_results = None

    def fit(self, examples, targets):
        self.trees = self.builder.build_trees(examples, targets)

    def predict(self, examples):
        n_rows, _ = examples.shape
        results = pd.DataFrame(index=range(n_rows), columns=self.trees.keys())
        results["prediction"] = pd.Series(index=range(n_rows))

        for column in results.columns:
            results[column] = np.uint8(255)
            # results[column] = results[column].astype(np.uint8)

        non_ones_columns = results.columns[-2:]
        for i in range(n_rows):
            for k, tree in self.trees.items():
                results.set_value(i, k, self._predict_one(examples[i], tree))

            result = results.ix[i].drop(non_ones_columns)

            results.set_value(
                i, "prediction",
                result[result == 1].index[0] if result.sum() == 1 else results.ix[i]["ag"]
            )

        self.last_results = results

        return results.as_matrix(columns=["prediction"])

    def _predict_one(self, example, tree):
        node = tree
        while not node.is_leaf():
            node = node.go(example[node.data])

        return node.data
