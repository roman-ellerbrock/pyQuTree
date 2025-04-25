from qutree.ttn.tensor_network import *
from qutree.ttn.grid import *

class Logger:
    def __init__(self):
        self.df = pd.DataFrame()
    
    def __call__(self, dic):
        self.df = pd.concat([self.df, pd.DataFrame(dic, index=[0])], ignore_index=True)

    def __str__(self):
        return f"Optimal value:\n{self.df.loc[self.df['f'].idxmin()]}"


def nparray_to_tuple(arr, precision = 8):
    return tuple(np.round(arr, precision))
    
class Objective:
    def __init__(self, Err, transformer = lambda x: x):
        self.Err = Err
        self.logger = Logger()
        self.cache = {}
        self.cache_hits = 0
        self.function_calls = 0
        self.transformer = transformer

    def __call__(self, x, **kwargs):
        # cache check
        key = nparray_to_tuple(x)
        if key in self.cache:
            self.cache_hits += 1
            return self.transformer(self.cache[key])

        f = self.Err(x)
        feff = self.transformer(f)

        # add to cache
        self.cache[key] = f
        self.function_calls += 1

        # logging
        if kwargs:
            xs = {f'x{i+1}': key[i] for i in range(len(x))}
            self.logger({**xs, 'f': f, **kwargs})
        return feff

    def __str__(self):
        nfunction = self.function_calls
        ncache = self.cache_hits
        output = str(self.logger) + f"""\n
Number of objective function calls: {nfunction}
Number of cached function accesses: {ncache}
Total number calls: {nfunction+ncache}"""
        return output

def ttnopt_step(G, O, sweep_id):
    G = G.copy()
    for edge in star_sweep(G, exclude_leafs=True):

        edges = pre_edges(G, edge)
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        kwargs = {'sweep': sweep_id, 'node': edge[0]}
        A = quTensor(grid.evaluate(O, **kwargs).reshape(ranks), edges)

        next, cross_inv = maxvol_grids(A, G, edge)

        # save results
        G.edges[edge]['grid'] = next
        G.edges[edge]['A'] = cross_inv
        G.nodes[edge[0]]['grid'] = grid
        G.nodes[edge[0]]['A'] = A
    return G


def ttnopt(G: nx.DiGraph, O: Objective, nsweep: int = 6,
           primitive_grid: list[np.array] = None,
           start_grid = None):
    """
    Run tensor network optimization on objective function using provided graph and grids.
    
    Arguments:
    G (nx.DiGraph): tensor network (graph)
    O (Objective): objective function
    primitive grids (list[np.array[int]]): list of numpy arrays of 
    """
    if primitive_grid:
        G = tn_grid(G, primitive_grid, start_grid=start_grid)

    for sw in range(nsweep):
        G = ttnopt_step(G, O, sw)

    return G

def tn_CUR(G, O):
    # Fix the tensors at the nodes to get proper cross approximation of O
    # Will re-evaluate points that have been evaluated already but who cares.
    # G: Tensor network
    # O: Objective function (should have cache)
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
    return G
