import numpy as np

class Tree:
    def __init__(self, data):
        self.children = {}
        self.data = data

    def __iter__(self):
        self.stack = [self]
        return self

    def __next__(self):
        if (len(self.stack) == 0):
            raise StopIteration
        next_node = self.stack.pop(0)
        for k, child in next_node.children.items():
            if not child.is_leaf:
                self.stack.insert(0, child)
        return next_node

    def add_child(self, rule, child):
        self.children[rule] = child

    def go(self, key):
        return self.children[key]

    def is_leaf(self):
        return len(self.children) == 0

    def predict(self, example):
        if hasattr(self, 'pruned'):
            return self.pruned
        if self.is_leaf():
            return self.data
        return self.go(example[self.data]).predict(example)

    def to_data(self, rule="null"):
        node = {
            "name": str(self.data),
            "rule": rule
        }

        if len(self.children) > 0:
            children = []
            node["children"] = children
            for r, t in self.children.items():
                children.append(t.to_data(r))

        return node

    def to_matlab(self):
        if len(self.children) == 0:
            return {
                # "op": None,
                "kids": np.zeros((0, 0), dtype=np.object),
                "class": self.data
            }
        else:
            kids = np.zeros((1, 2), dtype=np.object)
            kids[0][0] = self.children[0].to_matlab()
            kids[0][1] = self.children[1].to_matlab()

            return {
                "op": self.data,
                "kids": kids
                # "class": None
            }
