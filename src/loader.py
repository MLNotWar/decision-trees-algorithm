from scipy.io import loadmat
import numpy as np
from learning import DecisionTreeLearning
import pickle

from visualisation.server import main as visualise


def load_data(data_file):
    data = loadmat(data_file)
    return data['x'], data['y']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Decision Tree Learning.")
    parser.add_argument("data", type=str, help="the data file")
    parser.add_argument("-v", dest="visualisation", help="visualisation", action="store_true")

    args = parser.parse_args()

    examples, binary_targets = load_data(args.data)

    algorithm = DecisionTreeLearning()
    algorithm.fit(examples, binary_targets)

    trees = {k: v.to_data() for k, v in algorithm.trees.items()}
    if args.visualisation:
        visualise(trees)
