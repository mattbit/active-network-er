from collections import defaultdict

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from er.utils import get_nodes_pos, vals_by_dist


def _hex_periodic_edges(graph):
    """Finds the edges joining the opposite boundary nodes."""

    def _length(edge):
        u, v = edge
        x_u = (graph.nodes[u]["x"], graph.nodes[u]["y"])
        x_v = (graph.nodes[v]["x"], graph.nodes[v]["y"])

        return np.hypot(*np.subtract(x_u, x_v))

    return [edge for edge in graph.edges if _length(edge) > 1.5]


def heatmap(graph: nx.Graph, node_values=None, edge_values=None, cmap=None,
            ax=None, colorbar=True, vmin=None, vmax=None):
    if node_values is None and edge_values is None:
        raise ValueError(
            "At least one of nodes or edges values must be given.")

    if node_values is not None and edge_values is None:
        # Take the average value for the edges.
        edge_values = [np.mean(node_values[[u, v]]) for u, v in graph.edges]
    elif edge_values is not None and node_values is None:
        # Take the average value for the nodes.
        node_values = np.zeros(len(graph.nodes))
        count = np.zeros(len(graph.nodes))
        for i, (u, v) in enumerate(graph.edges):
            node_values[u] += edge_values[i]
            count[u] += 1
            node_values[v] += edge_values[i]
            count[v] += 1

        node_values = node_values / count

    if "label" in graph.graph and graph.graph["label"] == "Periodic hexagonal lattice":
        periodic_edges = _hex_periodic_edges(graph)
        edges = [edge for edge in graph.edges if edge not in periodic_edges]
        # Clip the edge values.
        edge_values = [edge_values[i] for i, edge in enumerate(graph.edges)
                       if edge in edges]
    else:
        edges = graph.edges

    if cmap is None:
        cmap = cmocean.cm.tempo

    pos = get_nodes_pos(graph)

    if vmin is None:
        vmin = min(*node_values, *edge_values)
    if vmax is None:
        vmax = max(*node_values, *edge_values)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])

    nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=edges, width=1,
                           edge_vmin=vmin, edge_vmax=vmax, edge_cmap=cmap,
                           edge_color=edge_values)

    p = nx.draw_networkx_nodes(graph, ax=ax, pos=pos,
                               vmin=vmin, vmax=vmax, cmap=cmap,
                               linewidths=0, node_size=1, node_color=node_values)

    if colorbar:
        cb = plt.colorbar(p, fraction=0.05, pad=0.04)
        cb.ax.tick_params(
            length=1.5,
            # pad=5,
            width=0.5,
            size=5,
            direction="in",
            color="white"
        )
        cb.ax.set_axisbelow(False)
        cb.outline.set_visible(False)
    else:
        cb = None

    return fig, ax, cb


def by_distance(graph, values, size=10, source=0, cmap=None, vmin=None,
                vmax=None):
    data = vals_by_dist(graph, values, source)

    if cmap is None:
        cmap = cmocean.cm.tempo

    if vmin is None:
        vmin = data.value.min()

    if vmax is None:
        vmax = data.value.max()

    # Prepare the scatter plot.
    means = data.groupby("distance", as_index=False).value.mean()

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = data.value.apply(norm).apply(cmap)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(data.distance, data.value, c=colors, s=size, alpha=0.5)
    ax.plot(means.distance, means.value, linestyle=":", linewidth=0.5, c="k")

    return fig, ax


def paths(graph, paths, labels=True, vmin=None, vmax=None, ax=None):
    edge_counts = defaultdict(lambda: 0)
    node_counts = defaultdict(lambda: 0)
    for path in paths:
        for a, b in path.edges():
            edge_counts[(a, b)] += 1
            edge_counts[(b, a)] += 1

        for node in path.nodes:
            node_counts[node] += 1

    values = [edge_counts[edge] for edge in graph.edges]

    fig, ax, cb = heatmap(graph, edge_values=values,
                          vmin=vmin, vmax=vmax, ax=ax)

    if labels:
        source = paths[0].start_node()
        target = paths[0].end_node()

        node_label(ax, graph, source, "S", loc=(-3, -3))
        node_label(ax, graph, target, "T")

    return fig, ax, cb


def node_label(ax, graph, node, label=None, loc=(.5, .5), color="#d40000", size=10):
    x, y = graph.nodes[node]["x"], graph.nodes[node]["y"]

    ax.scatter(x, y, s=size, facecolors=color, zorder=10, alpha=0.72,
               linewidths=0)

    if label is not None:
        ax.text(x + loc[0], y + loc[1], label,
                fontweight="bold", color=color, fontsize=size)


def structure(graph, edge_size=1, node_size=1, edge_color="#bbbbbb", node_color="#cccccc"):
    _periodic = _hex_periodic_edges(graph)
    edges = [edge for edge in graph.edges if edge not in _periodic]

    pos = get_nodes_pos(graph)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    nx.draw_networkx(graph, edgelist=edges, pos=pos, with_labels=False, ax=ax,
                     node_size=node_size, width=edge_size,
                     edge_color=edge_color, node_color=node_color)

    return fig, ax


def animation(graph, values, times, interval=None, title=None, size_factor=1):
    sizes = np.array(values) * size_factor
    if len(sizes.shape) == 2:
        sizes = np.array([sizes])

    fig, ax = structure(graph, edge_size=2, node_size=2, node_color="#dddddd")
    fig.set_size_inches(15, 15)
    pos = get_nodes_pos(graph)
    pos = get_nodes_pos(graph)
    xs = [pos[i][0] for i in range(graph.number_of_nodes())]
    ys = [pos[i][1] for i in range(graph.number_of_nodes())]

    nodes = np.empty(len(sizes), dtype=object)
    for i, vv in enumerate(sizes):
        nodes[i] = ax.scatter(xs, ys, s=vv[0], marker=".", edgecolors=None)
        nodes[i].set_alpha(0.7)
        nodes[i].set_zorder(10)

    text = ax.text(0, 1.01, "Time {:7.2f} s".format(times[0]), size=18,
                   family="Source Code Pro", transform=ax.transAxes)

    ax.text(0.5, 1.01, title, transform=ax.transAxes, size=18)

    pbar = tqdm(total=len(times), desc="Processing frames...")

    def _init():
        return (text, *nodes)

    def _update(frame):
        time, *vv = frame
        pbar.update()
        text.set_text("Time {:7.2f} s".format(time))

        for i, vals in enumerate(vv):
            nodes[i].set_sizes(vals, dpi=300)

        return (text, *nodes)

    if interval is None:
        interval = 1000 * (times[1] - times[0])

    return FuncAnimation(fig, _update, frames=zip(times, *sizes),
                         interval=interval, init_func=_init,
                         save_count=len(times), blit=True)
