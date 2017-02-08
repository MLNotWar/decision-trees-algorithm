from scipy.io import loadmat
import numpy as np
from learning import decision_tree_learning
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
    parser.add_argument("-o", dest="output", type=str, default="out", help="output file")

    args = parser.parse_args()

    examples, binary_targets = load_data(args.data)
    _, n_attributes = examples.shape
    attributes = np.full((1, n_attributes), True, dtype=np.bool)

    tree = decision_tree_learning(examples, attributes, binary_targets)

    data = tree.to_data()
    with open(args.output, mode="wb") as f:
        f.write(pickle.dumps(data))

    if args.visualisation:
        visualise(data)
