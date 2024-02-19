import networkx as nx
import numpy as np

"""
Network Utilities
"""

def pre_edges(G, edge, remove_flipped = False):
    pre = list(G.in_edges(edge[0]))
    if remove_flipped:
        pre.remove(flip(edge))
    return pre

def is_leaf(edge):
    return edge[0] < 0

def up_edge(edge):
    return edge[0] < edge[1]

def flip(edge):
    return (edge[1], edge[0])

def back_permutation(edges, edge):
    el = edges.index(edge)
    p = list(range(len(edges)))
    p.remove(el)
    p.append(el)
    return p

def flatten_back(A, shape):
    return A.reshape((-1, shape[-1]))

def collect(G, edges, key):
    """
    Graph objects from edges that correspond to 'key' as a list
    """
    items = []
    for e in edges:
        items.append(G.edges[e][key])
    return items

def sweep(G):
    def custom_sort_key(edge):
        return G[edge[0]][edge[1]]['sweep_id']

    # Sort edges based on the custom key
    return sorted(G.edges, key=custom_sort_key)

def rsweep(G):
    return reversed(sweep(G))

def add_leaves(G, f):
    for i in range(f):
        G.add_edge(-i - 1, i)

def leaf_idx(coord):
    return -coord - 1

def get_coordinate(leaf):
    return -leaf[0] - 1

def fill_qutree_network(G):
    """
    Fill a qutree network with missing inverse edges.
    """

def root(G):
    """
    Return the root of a tree
    """
    return max(G.nodes)

def leaves(G):
    """
    Return the leaves of a tree
    """
    return [node for node in G.nodes if node < 0]

def children(G, node):
    in_edges = G.in_edges(node)
    return [e[0] for e in in_edges if e[0] < node]

"""
Tensor Networks
"""
def tt_graph(f, r = 2, N = 8):
    """
    Generate a tensor train network
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(f))

    uid = 0
    # leaf edges
    add_leaves(G, f)
    for i in range(f):
        G.add_edge(-i - 1, i)
        G.edges[(-i - 1, i)]['sweep_id'] = uid
        uid += 1

    # normal edges
    for i in range(f - 1):
        G.add_edge(i, i + 1)
        G.edges[(i, i + 1)]['sweep_id'] = uid
        uid += 1

    # reverse edges
    for i in range(f - 1, 0, -1):
        G.add_edge(i, i - 1)
        G.edges[(i, i - 1)]['sweep_id'] = uid
        uid += 1

    # add ranks
    for edge in G.edges():
        if not is_leaf(edge):
            G.edges[edge]['r'] = r
        else:
            G.edges[edge]['r'] = N

    # add random edge entries
    for edge in G.edges():
        if is_leaf(edge):
            G[edge[0]][edge[1]]['coordinate'] = get_coordinate(edge)
    return G

def permute_to_back(edges, edge):
    if (edges[-1] == edge):
        return edges
    edges.remove(edge)
    edges.append(edge)
    return edges

def add_layer_index(G, root = None):
    if root is None:
        root = max(G.nodes)
    
    for node in G.nodes:
        layer = nx.shortest_path_length(G, node, root)
        G.nodes[node]['layer'] = layer
    
    return G


def _combine_nodes(nodes, id, edges):
    """
    combine nodes from a vector into new nodes
    nodes: list of nodes
    id: new node idx
    nodes: {1, 2, 3, 4} -> {5, 6}
    n = 5 -> 7
    and add edges {(1, 5), (2, 5), (3, 6), (4, 6)} to G
    """
    for i in range(len(nodes) - 2, -1, -2):
        l = nodes[i]
        r = nodes[i+1]
        nodes.pop(i)
        nodes.pop(i)
        nodes.append(id)
        edges.append((r, id))
        edges.append((l, id))
        id += 1
    return nodes, id, edges

def build_tree(f):
    """
    create edges for a (close-to) balanced tree with f leaves
    """
    nodes = list(range(f))
    id = f
    edges = []
    while len(nodes) > 1:
        nodes, id, edges = _combine_nodes(nodes, id, edges)
    return edges

def balanced_tree(f, r = 2, N = 8):
    """
    Generate a close-to balanced tree tensor network
    """
    G = nx.DiGraph()
    id = 0
    for i in range(f):
        edge = (-i - 1, i)
        G.add_edge(edge[0], edge[1])
        G.edges[edge]['sweep_id'] = id

    edges = build_tree(f)
    for edge in edges:
        G.add_edge(edge[0], edge[1])
        G.edges[edge]['sweep_id'] = id
        id += 1

    for edge in reversed(edges):
        G.add_edge(edge[1], edge[0])
        G.edges[flip(edge)]['sweep_id'] = id
        id += 1
    
    # add ranks
    for edge in G.edges():
        if not is_leaf(edge):
            G.edges[edge]['r'] = r
        else:
            G.edges[edge]['r'] = N

    # add coordinate info
    for edge in G.edges():
        if is_leaf(edge):
            G[edge[0]][edge[1]]['coordinate'] = get_coordinate(edge)
    return G 
