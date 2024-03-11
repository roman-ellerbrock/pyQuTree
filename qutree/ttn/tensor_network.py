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
        if is_leaf(edge, G):
            continue

        edges = pre_edges(G, edge)
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        A = quTensor(grid.evaluate(O).reshape(ranks), edges)

        next, cross_inv = maxvol_grids(A, G, edge)

        # save results
        G.edges[edge]['grid'] = next
        G.edges[edge]['A'] = cross_inv
        G.nodes[edge[0]]['grid'] = grid
        G.nodes[edge[0]]['A'] = A

        # update dfs
        dfs.append(tngrid_to_df(G, O))
    return G, dfs

def ttnopt(G, O, nsweep = 6):
    dfs = []
    for sw in range(nsweep):
        G, dfs = ttnopt_step(G, O, dfs)
    df = concat_pandas(dfs)

    # fix the tensors at the nodes
    for node in G.nodes():
        if node < 0:
            continue
        edges = G.in_edges(node)
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        A = quTensor(grid.evaluate(O).reshape(ranks), edges)
        G.nodes[node]['A'] = A

    for edge in sweep(G, include_leaves=False):
        edges = pre_edges(G, edge)
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        A = quTensor(grid.evaluate(O).reshape(ranks), edges)
        G[edge[0]][edge[1]]['A'] = A

    for edge in sweep(G, include_leaves=False):
        edges = [edge, flip(edge)]
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        cross = grid.evaluate(O).reshape(ranks)
        Ainv = quTensor(regularized_inverse(cross, 1e-12), edges)
        G[edge[0]][edge[1]]['A'] = Ainv

    return G, df

def create_tensors(G, generator = np.zeros, key = 'A'):
    for node in G.nodes():
        if (node < 0):
            continue
        #edges = G.incoming_edges(node)
        edges = G.in_edges(node)
        ranks = collect(G, edges, 'r')
        R = np.prod(ranks)
        A = quTensor(generator(R).reshape(ranks), edges)
        G.nodes[node][key] = A
    
    for edge in sweep(G):
        if (is_leaf(edge, G)):
            continue
        edges = [edge, flip(edge)]
        ranks = collect(G, edges, 'r')
        A = quTensor(np.eye(ranks[0]), edges)
        G[edge[0]][edge[1]][key] = A
    return G

def contract(G, contraction_path = None):
    # construct a default contraction path for a bottom-up sweep
    if contraction_path is None:
        contraction_path = sweep(G, include_leaves=False)
        contraction_path = [edge for edge in contraction_path if up_edge(edge, G)]

    for edge in contraction_path:
        a, b = edge
        A = G.nodes[a]['A']
        AB = G[a][b]['A']
        B = G.nodes[b]['A']
        A = tensordot(A, AB, edge)
        G.nodes[b]['A'] = tensordot(A, B, edge)
        G = remove_edge(G, (a, b))
    return G

def extract_root_tensor(G):
    """
    Small helper function that allows to extract a single-tensor from a tn
    after contraction
    """
    return G.nodes[root(G)]['A'].reshape(-1)

def tn_to_tensor(G):
    return extract_root_tensor(contract(G))

#def tn_to_tensor(G):
#    """
#    Small helper function that allows to extract a single-tensor from a tn
#    after contraction
#    """
#    F = contract(G)
#    leaves = G.nodes
#    leaves = [node for node in leaves if node < 0]
#    n_leaves = len(leaves)
#    node_id = max(G.nodes)
#    F = F.nodes[node_id]['A']
#    p = list(range(n_leaves - 1, -1, -1))
#    F = np.transpose(F, p).reshape(-1)
#    return F
#