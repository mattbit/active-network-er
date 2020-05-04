"""Algorithms and utilities for graphs."""

import networkx as nx


def k_cores(graph: nx.Graph, k: int, inplace=False):
    """Return the components of the k-core graph.

    Parameters
    ----------
    graph : networkx.Graph
        An undirected graph.
    k : int
        The order of the core.
    inplace: bool
        If `True`, the k-core graph will be built directly by
        modifying the instance `graph`.
    Returns
    -------
    (core_graph, connected_components)
        The core graph and a generator of its connected components.
    """
    g = graph.copy() if not inplace else graph

    # Simple k-core algorithm
    tbr = [node for node, deg in g.degree() if deg < k]
    while len(tbr) > 0:
        g.remove_nodes_from(tbr)
        tbr = [node for node, deg in g.degree() if deg < k]

    return g, nx.connected_components(g)
