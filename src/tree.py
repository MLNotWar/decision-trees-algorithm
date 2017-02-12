class Tree:
    def __init__(self, data):
        self.children = {}
        self.data = data

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
