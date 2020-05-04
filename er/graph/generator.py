"""A collection of graph generators.

All the graph generators here create a :class:`networkx.Graph` instance with
integer node labels increasing from the center (the central node will have index
equal to 0).
"""

import math

import networkx as nx
import numpy as np
import scipy.io


def random_regular(num_nodes):
    """Generates a random 3-regular graph with `num_nodes` nodes.

    Args:
        num_nodes: The desired number of nodes in the graph.

    Returns:
        A networkx.Graph instance.
    """
    graph = nx.generators.random_regular_graph(3, num_nodes)
    graph.graph["label"] = "Random regular"

    return graph


def hexagonal_lattice(num_nodes, periodic=False):
    """Generates an hexagonal lattice graph with about `num_nodes` nodes.

    Args:
        num_nodes: The desired number of nodes. Note that the produced lattice
            may not have the exact number of nodes specified.
        periodic: Whether the generated lattice should be periodic (the nodes on
            the contour will be connected each other). False by default.
    Returns:
        A networkx.Graph instance.
    """
    m = int(math.sqrt(num_nodes / 2))
    n = m if not periodic else m + m % 2

    lattice = nx.generators.hexagonal_lattice_graph(m, n, periodic)

    lattice.graph["label"] = "Hexagonal lattice"

    if periodic:
        lattice.graph["label"] = "Periodic hexagonal lattice"

        # Recompute the positions
        rows = range(2 * m + 2)
        cols = range(n + 1)
        ii = (i for i in cols for j in rows)
        jj = (j for i in cols for j in rows)
        xx = (0.5 + i + i // 2 + (j % 2) * ((i % 2) - .5) for i in cols
              for j in rows)
        h = math.sqrt(3) / 2
        yy = (h * j for i in cols for j in rows)
        # exclude nodes not in G
        pos = {(i, j): (x, y)
               for i, j, x, y in zip(ii, jj, xx, yy) if (i, j) in lattice}
        nx.set_node_attributes(lattice, pos, "pos")

        # Clear contraction attributes
        for _, data in lattice.nodes(data=True):
            if "contraction" in data:
                del data["contraction"]


    # Split the position in two distinct keys
    for node, data in lattice.nodes(data=True):
        data["x"], data["y"] = data["pos"]
        del data["pos"]

    return _relabel_nodes_by_distance(lattice, spatial_center=True)


def from_matfile(filename, adj_key="C", nodes_key="nodes"):
    """Generates a graph based on a Matlab file (.mat).

    The Matlab file must contain the adjacency matrix (`C`) and a list of node
    coordinates (`nodes`).

    Args:
        filename: The path to the .mat file.
        adj_key: The name of the cell containing the adjacency matrix.
        nodes_key: The name of the cell containing the nodes.
    Returns:
        A networkx.Graph instance.
    """
    mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    graph = nx.Graph(mat[adj_key])  # build graph from adjacency matrix

    xs = {}
    ys = {}
    for n, pos in enumerate(mat[nodes_key]):
        xs[n], ys[n] = pos[:2]

    nx.set_node_attributes(graph, xs, name="x")
    nx.set_node_attributes(graph, ys, name="y")

    graph.graph["label"] = "Reconstructed ER"

    return _relabel_nodes_by_distance(graph)


def _relabel_nodes_by_distance(graph, spatial_center=False):
    if spatial_center:
        nodes = list(graph.nodes)
        coords = np.array([(graph.nodes[node]["x"], graph.nodes[node]["y"])
                           for node in nodes])
        x = 0.5 * (coords[:, 0].max() - coords[:, 0].min())
        y = 0.5 * (coords[:, 1].max() - coords[:, 1].min())
        index = np.argmin(np.sum((coords - [x, y])**2, axis=1))
        central_node = nodes[index]
    else:
        central_node = nx.center(graph)[0]

    node_distance = nx.shortest_path_length(graph, central_node)
    mapping = {
        node: i
        for i, node in enumerate(sorted(node_distance, key=node_distance.get))
    }

    return nx.relabel_nodes(graph, mapping)
