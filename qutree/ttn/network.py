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

def is_leaf(edge, G):
    return G.in_degree(edge[0]) <= 1 or G.in_degree(edge[1]) <= 1
#    return G.in_degree(edge[0]) == 0 or G.in_degree(edge[1]) == 0 # works only if leaves are added as (-i - 1, i) but not reverse

def is_leaf_node(node, G):
    return G.in_degree(node) <= 1 or G.in_degree(node) <= 1

def up_edge(edge, G):
    d0 = nx.shortest_path_length(G, source=edge[0], target=root(G))
    d1 = nx.shortest_path_length(G, source=edge[1], target=root(G))
    return d0 > d1

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

def up_edges_by_distance_to_root(G, root):
    distance = nx.shortest_path_length(G, source=root)

    up_edges = [edge for edge in G.edges() if distance[edge[0]] > distance[edge[1]]]
    down_edges = [edge for edge in G.edges() if distance[edge[0]] < distance[edge[1]]]

    sorted_up_edges = sorted(up_edges, key=lambda edge: -distance[edge[1]])
    sorted_down_edges = sorted(down_edges, key=lambda edge: distance[edge[1]])
    return sorted_up_edges, sorted_down_edges

def sweep(G, include_leaves = True):
    up, down = up_edges_by_distance_to_root(G, root(G))
    sw = up + down
    if not include_leaves:
        sw = [edge for edge in sw if not is_leaf(edge, G)]
    return sw

def up_sweep(G, include_leaves = True):
    # @todo: add unit test
    up, _ = up_edges_by_distance_to_root(G, root(G))
    sw = up
    if not include_leaves:
        sw = [edge for edge in sw if not is_leaf(edge, G)]
    return sw

#def sweep(G, include_leaves = True):
#    up = sorted(G.edges, key = lambda x: x[0])
#    up = [edge for edge in up if up_edge(edge)]
#    down = reversed(up)
#    down = [edge for edge in down if not is_leaf(edge, G)]
#    down = [flip(edge) for edge in down]
#    res = up + down
#    if not include_leaves:
#        res = [edge for edge in res if not is_leaf(edge, G)]
#    return res

def rsweep(G):
    return reversed(sweep(G))

def add_leaves(G, f):
    for i in range(f):
        edge = (i, -i - 1)
        G.add_edge(i, -i - 1)
        G.add_edge(-i - 1, i)
        G.edges[edge]['coordinate'] = i
        G.edges[flip(edge)]['coordinate'] = i

def root(G):
    """
    Return the root of a tree
    """
    # @todo: remove dependency of node indexing. Maybe at least use '0' as root
    return max(G.nodes)

def up_leaves(G):
    """
    Return the leaves of a tree
    """
    return [edge for edge in G.edges if is_leaf(edge, G) and up_edge(edge, G)]

def children(G, node):
    in_edges = G.in_edges(node)
    return [e[0] for e in in_edges if e[0] < node]

def _star_sweep(G, node, parent = None, sweep = []):
    childs = children(G, node)
    for child in childs:
        if child < 0:
            continue
        sweep.append((node, child))
        _star_sweep(G, child, node, sweep)
    if parent is not None:
        sweep.append((node, parent))

def star_sweep(G, exclude_leafs = False):
    rt = root(G)
    sweep = []
    _star_sweep(G, rt, None, sweep)
    if exclude_leafs:
        sweep = [edge for edge in sweep if not is_leaf(edge, G)]
    return sweep

def remove_edge(G, edge):
    pre = pre_edges(G, edge, remove_flipped=True)
    v = edge[0]
    w = edge[1]
    # redirect edges
    for e in pre:
        attr = G.get_edge_data(*e)
        G.add_edge(e[0], w, **attr)
        G.remove_edge(*e)
    G.remove_edge(*edge)
    G.remove_edge(*flip(edge))
    G.remove_node(v)
    return G

"""
Tensor Networks
"""

def add_layer_index(G, root = None):
    if root is None:
        root = max(G.nodes)
    for node in G.nodes:
        layer = nx.shortest_path_length(G, node, root)
        G.nodes[node]['layer'] = layer
    return G

def tensor_train_graph(f, r = 2, primitive_grid: int | list[int] = 8):
    """
    Generate a tensor train network
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(f))

    # leaf edges
    add_leaves(G, f)

    # normal edges
    for i in range(f - 1):
        G.add_edge(i, i + 1)

    # reverse edges
    for i in range(f - 1, 0, -1):
        G.add_edge(i, i - 1)

    if isinstance(primitive_grid, int):
        Ns = [primitive_grid] * f
    else:
        # Ns = N
        Ns = []
        for grid in primitive_grid:
            Ns.append(len(grid))
            

    # add ranks
    for edge in sweep(G):
        if not is_leaf(edge, G):
            pre = pre_edges(G, edge, True)
            rmax = np.prod(collect(G, pre, 'r'))
            G.edges[edge]['r'] = min(r, rmax)
        else:
            coord = G.edges[edge]['coordinate']
            G.edges[edge]['r'] = Ns[coord]

    for edge in sweep(G):
        if not is_leaf(edge, G):
            r = G.edges[edge]['r']
            other = G.edges[flip(edge)]['r']
            G.edges[edge]['r'] = min(r, other)

    return G

def tensor_train_operator_graph(f, r = 2, N = 8):
    """
    Generate a tensor train network
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(f))

    # leaf edges
    for i in range(0, f):
        node = i
        x = -2*i - 1
        y = -2*i - 2
        G.add_edge(node, x)
        G.add_edge(x, node)
        G.add_edge(node, y)
        G.add_edge(y, node)


    # normal edges
    for i in range(f - 1):
        G.add_edge(i, i + 1)

    # reverse edges
    for i in range(f - 1, 0, -1):
        G.add_edge(i, i - 1)

    # add ranks
    for edge in G.edges():
        if not is_leaf(edge, G):
            G.edges[edge]['r'] = r
        else:
            G.edges[edge]['r'] = N

    # add random edge entries
    coord = 0
    for edge in G.edges():
        if is_leaf(edge, G) and up_edge(edge, G):
            G.edges[edge]['coordinate'] = coord
            coord += 1
    return G

def _combine_nodes(nodes, id, edges):
    """
    Helper function for balanced_tree
    Combines nodes from a vector into new nodes.
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
    # special case for odd number of leaves
    if f % 2 == 1:
        nodes[0] = -1 
    id = f
    edges = []
    while len(nodes) > 1:
        nodes, id, edges = _combine_nodes(nodes, id, edges)
    return edges

def balanced_tree(f, r = 2, N = 8):
    """
    Generate a close-to balanced tree tensor network
    Node: odd f is implemented manually:
          avoids adding unnecessary node by added code.
          See # odd-leaf tag
    """
    G = nx.DiGraph()

    start = f % 2 # special case for odd number of leaves
    for i in range(start, f):
        edge = (-i - 1, i)
        G.add_edge(edge[0], edge[1])
        G.add_edge(edge[1], edge[0])

    edges = build_tree(f)
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    for edge in reversed(edges):
#        if edge[0] < 0:
#            continue # special case for odd number of leaves
        G.add_edge(edge[1], edge[0])
    
    # add ranks
    for edge in sweep(G):
        if not is_leaf(edge, G):
            pre = pre_edges(G, edge, True)
            rmax = np.prod(collect(G, pre, 'r'))
            G.edges[edge]['r'] = min(r, rmax)
        else:
            G.edges[edge]['r'] = N

    for edge in sweep(G):
        if not is_leaf(edge, G):
            r = G.edges[edge]['r']
            other = G.edges[flip(edge)]['r']
            G.edges[edge]['r'] = min(r, other)

    return G 
