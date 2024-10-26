
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qutree.ttn.grid import *
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import imageio
import os

def plot_xyz(xyz, f, ranges = None):
    # Create a 3D scatter plot with colors based on f_values
    xyz = xyz.grid
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=f,  # Use f_values as colors
            colorscale='Viridis',  # Choose a colorscale
            colorbar=dict(title='Function Value')  # Add colorbar with a title
        )
    )])

    # Set plot layout with fixed axis range
    if ranges is None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X Axis', autorange=True),  # Fix x-axis range to [0, 1]
                yaxis=dict(title='Y Axis', autorange=True),  # Fix y-axis range to [0, 1]
                zaxis=dict(title='Z Axis', autorange=True),   # Fix z-axis range to [0, 1]
                aspectmode='cube',  # Ensure that aspect ratio is maintained
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X Axis', range=ranges[0], autorange=False),  # Fix x-axis range to [0, 1]
                yaxis=dict(title='Y Axis', range=ranges[1], autorange=False),  # Fix y-axis range to [0, 1]
                zaxis=dict(title='Z Axis', range=ranges[2], autorange=False),   # Fix z-axis range to [0, 1]
                aspectmode='cube',  # Ensure that aspect ratio is maintained
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

    # Show the plot
    return fig

def plot_tt_diagram(tn, draw_ranks = True):
    nleaves = len(up_leaves(tn))
    pos0 = [np.array([-(-i - 1), 0.1]) for i in range(-nleaves, 0)]
    pos1 = [np.array([-i, 0]) for i in range(nleaves)]
    pos12 = pos0 + pos1
    pos = {i: pos12[i] for i in range(-nleaves, nleaves)}
#    plt.draw() 
    plt.gca().set_aspect(15)  # 'equal' ensures that one unit in x is equal to one unit in y
    fig = nx.draw(tn, pos = pos, with_labels=False, node_size = 250, font_size = 8)
    if draw_ranks:
        ranks = nx.get_edge_attributes(tn, 'r')
        nx.draw_networkx_edge_labels(tn, pos, edge_labels=ranks, font_size=14)
#    plt.subplots_adjust(left=0.1, right=0.9, bottom=10.1, top=20.9)
    return fig

def plot_tn_xyz(tn, fun, q_to_x = None):
    # collect grids
    gs = nx.get_node_attributes(tn, 'grid')
    gs = list(gs.values()) # dict to list
    grid = direct_sum(gs)
    if not q_to_x is None:
        grid = grid.transform(q_to_x)
    fig = plot_xyz(grid, grid.evaluate(fun))
    fig.show()

def tn_to_df(tn, fun):
    # collect grids
    gs = nx.get_node_attributes(tn, 'grid')
    gs = list(gs.values()) # dict to list
    xyz, f, node = [], [], []
    for id, grid in enumerate(gs):
        for i, point in enumerate(grid.grid):
            xyz.append(point)
            node.append(id)
    return pd.DataFrame({'xyz': xyz, 'node': node})

def plot_tree(G, draw_ranks = True):
    G = add_layer_index(G)
    nleaves = len(up_leaves(G))
    grid = np.linspace(0, 1, nleaves)
    pos = {i : (0, 0) for i in sorted(G.nodes)}
    for node in sorted(G.nodes):
        layer = G.nodes[node]["layer"]
        y = -layer
        x = 0.
        if node < 0:
            id = -node - 1
            x = grid[id]
        else:
            cs = children(G, node)
            children_pos = np.array([pos[c][0] for c in cs])
            x = np.mean(children_pos, axis=0)
        pos[node] = (x, y)
    
    nx.draw(G, pos = pos, with_labels=False, node_size = 500)
    if draw_ranks:
        ranks = nx.get_edge_attributes(G, 'r')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=ranks, font_size = 18)
    plt.draw()

    
def tngrid_to_df(G, O):
    # Get node attributes as a dictionary
    node_attributes = nx.get_node_attributes(G, 'grid')
    node_attributes = {k: v.grid for k, v in node_attributes.items() if v is not None}

    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(node_attributes.items()), columns=['node', 'grid'])

    # Create a new DataFrame with the exploded values and reset the index
    df = df.explode('grid').reset_index()

    df['f'] = df['grid'].apply(lambda x: O.Err(x))

    # Remove the 'index' column
    df = df.drop(columns='index')

    # Convert the 'grid' column into separate columns
    L = len(df['grid'].iloc[0])
    df[['x{}'.format(i+1) for i in range(L)]] = df['grid'].apply(pd.Series)

    # Drop the original 'grid' column
    df.drop(columns='grid', inplace=True)
    return df

def concat_pandas(dfs):
    for t, df in enumerate(dfs):
        df['time'] = t
    df = pd.concat(dfs).reset_index()
    df["size"] = 1
    return df

def grid_animation(df, color = 'f'):
    fig = px.scatter_3d(df, x="x1", y="x2", z="x3", animation_frame="time", animation_group="node",
               size="size", color=color, hover_name="time",
               size_max=15,
                width=1000, height=800)
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 20
    fig.update_layout(scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
    ))

    camera = dict(
        eye=dict(x=1.5, y=-1.0, z=1.2),  # Set the position of the camera
        center=dict(x=0, y=0, z=0),       # Set the point the camera is looking at
        up=dict(x=0, y=0, z=1)            # Set the upward direction of the camera
    )
    fig.update_layout(scene_camera=camera)
    return fig

def grid_animation_to_gif(df, color='f', gif_filename='animation.gif', frames_folder = '.frames'):
    os.makedirs(frames_folder, exist_ok=True)
    unique_times = df['time'].unique()
    unique_times = unique_times[2:35]
    for time in unique_times:
        sub_df = df[df['time'] == time]

        # Generate the plot
        fig = px.scatter_3d(sub_df, x="x1", y="x2", z="x3", color=color, size_max=10, width=1000, height=800)
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                                     xaxis=dict(showticklabels=False),
                                     yaxis=dict(showticklabels=False),
                                     zaxis=dict(showticklabels=False)))
        camera = dict(
            eye=dict(x=1.5, y=-1.0, z=1.2),  # Set the position of the camera
            center=dict(x=0, y=0, z=0),       # Set the point the camera is looking at
            up=dict(x=0, y=0, z=1)            # Set the upward direction of the camera
        )
        
        # Adjust transparency settings if needed
        fig.update_traces(marker=dict(opacity=0.75, size = 6))  # Adjust opacity here
        fig.update_layout(scene_camera=camera)
        fig.write_image(os.path.join(frames_folder, f'frame_{time:04d}.png'), format = 'png')

    # Generate GIF
    images = [imageio.imread(os.path.join(frames_folder, f'frame_{i:04d}.png')) for i in unique_times]
    imageio.mimsave(gif_filename, images, fps=10)  # Adjust fps as needed

    # Cleanup images
    for filename in os.listdir(frames_folder):
        os.remove(os.path.join(frames_folder, filename))
    os.rmdir(frames_folder)

    return gif_filename