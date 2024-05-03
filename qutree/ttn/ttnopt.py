from qutree.ttn.tensor_network import *
from qutree.ttn.grid import *

class Logger:
    def __init__(self):
        self.df = pd.DataFrame()
    
    def __call__(self, dic):
        self.df = pd.concat([self.df, pd.DataFrame(dic, index=[0])], ignore_index=True)

class Objective:
    def __init__(self, Err):
        self.Err = Err
        self.logger = Logger()
    
    def __call__(self, x, **kwargs):
        f = self.Err(x)
        xs = {f'x{i+1}': x[i] for i in range(len(x))}
        if kwargs:
            self.logger({**xs, 'f': f, **kwargs})
        return f

def ttnopt_step(G, O, sw, dfs = []):
    G = G.copy()
    for edge in star_sweep(G):
        if is_leaf(edge, G):
            continue

        edges = pre_edges(G, edge)
        grids = collect(G, edges, 'grid')
        grid = cartesian_product(grids).permute()
        ranks = collect(G, edges, 'r')
        kwargs = {'time': sw, 'node': edge[0], 'size': 1}
        A = quTensor(grid.evaluate(O, **kwargs).reshape(ranks), edges)

        next, cross_inv = maxvol_grids(A, G, edge)

        # save results
        G.edges[edge]['grid'] = next
        G.edges[edge]['A'] = cross_inv
        G.nodes[edge[0]]['grid'] = grid
        G.nodes[edge[0]]['A'] = A

        # update dfs
#        O.logger.snapshot(G)
        dfs.append(tngrid_to_df(G, O))
    return G, dfs

def ttnopt(G, O, nsweep = 6):
    dfs = []
    for sw in range(nsweep):
        G, dfs = ttnopt_step(G, O, sw, dfs)
    df = pd.DataFrame()
    if len(dfs) > 0:
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