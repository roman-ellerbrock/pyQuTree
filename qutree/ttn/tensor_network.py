import numpy as np
import qutree.ttn.grid as grid
from qutree.ttn.network import *
from qutree.ttn.grid import *
from qutree.plot import *

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

def tensordot(A, B, edge):
    e = tuple(sorted(edge))
    iA = A.edges.index(e)
    iB = B.edges.index(e)
    edges_c = A.edges + B.edges
    edges_c.remove(e)
    edges_c.remove(e)
    # update A.edges to go into B
    # means: contract A into B
    return quTensor(np.tensordot(A, B, axes = (iA, iB)), edges_c)

def ttnopt_step(G, O, dfs = []):
    G = G.copy()
    for edge in depth_first(G):
        if (is_leaf(edge)):
            continue

        edges = pre_edges(G, edge)
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        A = quTensor(grid.evaluate(O).reshape(ranks), edges)

        next, cross_inv = maxvol_grids(A, G, edge)

        # save results
        G.nodes[edge[0]]['grid'] = grid
        G[edge[0]][edge[1]]['grid'] = next
        G[edge[0]][edge[1]]['A'] = cross_inv
        G.nodes[edge[0]]['A'] = A

        # update dfs
        dfs.append(tngrid_to_df(G, O))
    return G, dfs

def ttnopt(G, O, nsweep = 6):
    dfs = []
    for sw in range(nsweep):
        G, dfs = ttnopt_step(G, O, dfs)
    df = concat_pandas(dfs)
    return G, df
