import numpy as np
import pandas as pd
from collections import defaultdict

def predict(trees, examples):
    n_rows, _ = examples.shape
    results = pd.DataFrame(index=range(n_rows), columns=trees.keys())
    results["prediction"] = pd.Series(index=range(n_rows))

    for column in results.columns:
        results[column] = np.uint8(255)

    non_ones_columns = results.columns[-2:]
    for i in range(n_rows):
        for k, tree in trees.items():
            results.set_value(i, k, tree.predict(examples[i]))

        result = results.ix[i].drop(non_ones_columns)
        val = result[result == 1].index[0] if result.sum() == 1 else results.ix[i]["ag"]
        results.set_value(i, "prediction", val)

    return results.as_matrix(columns=["prediction"])

def score_predictions(tree, examples, targets):
    size, _ = examples.shape
    score = 0
    for i in range(size):
        if tree.predict(examples[i]) == targets[i]:
            score += 1
    return score
