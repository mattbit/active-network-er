"""Algorithms and utilities for graphs."""

import networkx as nx


def k_cores(G: nx.Graph, k: int, inplace=False):
    """Return the components of the k-core graph.

    Arguments:
        G: An undirected graph.
        k: The order of the core.
        inplace: If `True`, the k-core graph will be built directly by
            modifying the instance G.
    Returns:
        The core graph and a generator of its connected components.
    """
    g = G.copy() if not inplace else G

    # Simple k-core algorithm
    tbr = [node for node, deg in g.degree() if deg < k]
    while len(tbr) > 0:
        g.remove_nodes_from(tbr)
        tbr = [node for node, deg in g.degree() if deg < k]

    return g, nx.connected_components(g)
