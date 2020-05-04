import itertools
from collections import defaultdict

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from . import graph


def plot_graph(graph, filename):
    source = 0
    positions = np.array(list(nx.get_node_attributes(graph, "pos").values()))

    xs = []
    ys = []
    for u, v in graph.edges:
        xs.append([positions[u, 0], positions[v, 0]])
        ys.append([positions[u, 1], positions[v, 1]])

    output_file(filename)
    title = graph.graph["label"] if "label" in graph.graph else None
    p = figure(title=title)
    nodes = p.circle(positions[:, 0], positions[:, 1])
    p.circle(*positions[source], color="red")
    p.multi_line(xs, ys)
    p.tools.append(HoverTool(renderers=[nodes]))

    return p


def values_by_distance(distances, values, cmap=None, vmin=None, vmax=None):
    """Plots values by distance from the source node."""
    if cmap is None:
        cmap = cmocean.cm.tempo

    if vmin is None:
        vmin = min(val for vals in values for val in vals)

    if vmax is None:
        vmax = max(val for vals in values for val in vals)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Prepare the scatter plot.
    colors = []
    ds = []
    vs = []
    for dist, vals in zip(distances, values):
        ds += [dist] * len(vals)
        vs += vals
        colors += [cmap(norm(val)) for val in vals]

    mean_values = list(map(np.mean, values))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(ds, vs, c=colors, s=1, alpha=0.5)
    ax.plot(distances, mean_values, linestyle=":", linewidth=0.5, c="k")

    ax.ticklabel_format(style="sci", axis="y",
                        scilimits=(0, 2), useMathText=True)
    ax.get_yaxis().get_offset_text().set_x(-0.175)

    return fig, ax


def comparison(xs, ys, labels, colors=None):
    if colors is None:
        colors = range(len(xs))

    print(colors)
    cmap = matplotlib.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))

    fig, ax = plt.subplots(1, 1)
    for x, y, label, clr in zip(xs, ys, labels, colors):
        ax.plot(x, y, c=cmap(norm(clr)), label=label)

    ax.legend()

    return fig, ax


def hcurve(start, end, rad=None, **kwargs):
    """Draws a bezier curve with horizontal control points."""
    if rad is None:
        rad = (end[0] - start[0]) / 2

    nodes = [start, (start[0] + rad, start[1]), (end[0] - rad, end[1]), end]

    moves = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    path = Path(nodes, moves)

    return PathPatch(path, **kwargs)
