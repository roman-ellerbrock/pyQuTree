import numpy as np
from qutree.ttn.network import *
from qutree.ttn.grid import *
from qutree.plot import *
from qutree.ttn.tensor import *

class TensorNetwork(nx.DiGraph):
    pass

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