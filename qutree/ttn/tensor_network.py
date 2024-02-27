import numpy as np
from qutree.ttn.network import *
from qutree.ttn.grid import *
from qutree.plot import *
from qutree.ttn.tensor import *

class TensorNetwork(nx.DiGraph):
    pass

def ttnopt_step(G, O, dfs = []):
    G = G.copy()
    for edge in star_sweep(G):
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

def create_tensors(G):
    for node in G.nodes():
        if (node < 0):
            continue
        edges = G.incoming_edges(node)
        ranks = collect(G, edges, 'r')
        A = quTensor(np.zeros(ranks), edges)
        G.nodes[node]['A'] = A
    
    for edge in G.nodes():
        if (is_leaf(edge)):
            continue
        edges = [edge, flip(edge)]
        ranks = collect(G, edges, 'r')
        A = quTensor(np.eye(ranks[0]), edges)
        G.nodes[edge]['A'] = A
    return G
