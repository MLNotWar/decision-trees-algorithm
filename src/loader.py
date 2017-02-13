from scipy.io import loadmat, savemat
from tree_builder import BasicTreeBuilder, PrunedTreeBuilder
import pprint
import os

from visualisation.server import main as visualise
from test.k_folding import KFoldTest


def load_data(data_file):
    data = loadmat(data_file)
    return data['x'], data['y']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Decision Tree Learning.")
    parser.add_argument("data", type=str, help="the data file")
    parser.add_argument("-v", dest="visualisation", help="visualisation", action="store_true")
    parser.add_argument("-t", dest="test", action="store_true")
    parser.add_argument("-p", dest="prune", action="store_true")
    parser.add_argument("-o", dest="optimise", action="store_true")
    parser.add_argument("-s", action="store_true", dest="save", help="save output as Matlab file (in out/)")

    args = parser.parse_args()

    examples, binary_targets = load_data(args.data)
    builder = BasicTreeBuilder() if not args.prune else PrunedTreeBuilder()

    if args.test:
        test = KFoldTest(examples, binary_targets)
        confusion_matrix = test.evaluate(builder)
        pprint.pprint(confusion_matrix.generate_report())

        exit(0)

    trees = builder.build_trees(examples, binary_targets, optimise=args.optimise)

    if args.save:
        if not os.path.exists("out/"):
            os.mkdir("out")

        for k, v in trees.items():
            if k == "ag":
                continue
            savemat("out/%s.mat" % k, {"tree": v.to_matlab()})

    if args.visualisation:
        trees = {k: v.to_data() for k, v in trees.items()}
        if args.visualisation:
            visualise(trees)

