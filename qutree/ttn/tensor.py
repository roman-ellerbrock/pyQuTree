import numpy as np
from .network import *

class quTensor(np.ndarray):
    """
    Decorated tensor that keeps track of corresponding edges
    edges: those correspond to the tensor legs
    flattened_to: None or edge that the current tensor is flattened to.
    expanded_shape: shape if not permuted and flattened

    Note: edges & expanded_shape is not permuted. Only 
    """
    def __new__(cls, array, edges, flattened_to = None):
        if (len(edges) != len(array.shape)):
            raise ValueError("Number of edges does not match the shape of the tensor.")
        obj = np.asarray(array).view(cls)
        obj.edges = [tuple(sorted(edge)) for edge in edges] # edge = (small, large)
        return obj

    def flatten(self, edge):
        edge = tuple(sorted(edge))
        p = back_permutation(self.edges, edge)
        A = self.transpose(p)
        s = [self.shape[i] for i in p]
        edges_p = [self.edges[i] for i in p]
        edges = [edges_p[:-1], edge]
        return quTensor(A.reshape((-1, s[-1])), edges)
    
    def transpose(self, axes = None):
        if axes is None:
            axes = list(range(len(self.shape)))
            axes = axes[::-1]
        edges = [self.edges[i] for i in axes]
        A = super().transpose(axes)
        return quTensor(A, edges)

def tensordot(A, B, edge):
    e = tuple(sorted(edge))
    iA = A.edges.index(e)
    iB = B.edges.index(e)
    edges_a = A.edges.copy()
    edges_a.remove(e)
    edges_b = B.edges.copy()
    edges_b.remove(e)
#    print(edges_a, edges_b)
    edges_c = edges_a + edges_b
    for i, ex in enumerate(edges_c):
        if (ex[1] == e[0]):
#            print('swapping', ex, ', new edge: ', (ex[0], e[1]))
            edges_c[i] = (ex[0], e[1])
    edges_c = [tuple(sorted(edge)) for edge in edges_c]
#    print(iA, iB)

    return quTensor(np.tensordot(A, B, axes = (iA, iB)), edges_c)
