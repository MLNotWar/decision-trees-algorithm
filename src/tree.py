class Tree:
    def __init__(self, data):
        self.children = {}
        self.data = data

    def add_child(self, rule, child):
        self.children[rule] = child
