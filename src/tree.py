class Tree:
    def __init__(self, data=None):
        self.children = []
        self.data = data

    def add_child(self, child):
        self.children.append(child)
