import numpy as np

def create_attributes(shape):
    _, n_attributes = shape
    attributes = np.full((1, n_attributes), True, dtype=np.bool)
    return attributes
