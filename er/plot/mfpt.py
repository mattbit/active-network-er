import math

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_mfpt_distance_comparison(simulations, key, label="{}"):
    """Plots average MFPT by distance for different parameter values.

    Args:
        simulations: A list of MFPTSimulation objects.
        key: A callable that receive the simulation as argument and returns
            the parameter value.
        label: The format used for printing the legend.

    Returns:
        A tuple of the matplotlib Figure and Axes.
    """
    param_values = [key(s) for s in simulations]

    fig, ax = plt.subplots(1, 1, figsize=(6.29, 2))
    cmap = matplotlib.cm.viridis
    norm = matplotlib.colors.LogNorm(
        vmin=min(param_values), vmax=max(param_values))

    for s in simulations:
        param = key(s)
        dists, times = s.result.mfpt_by_distance(s.model.graph)
        mean_times = list(map(np.mean, times))

        ax.plot(
            dists,
            mean_times,
            linewidth=1,
            color=cmap(norm(param)),
            label=label.format(param))

    ax.set_ylabel("$\\bar{\\tau}_{S \\to T}$   [s]")
    ax.set_xlabel("$d(S, T)$")

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 1), useMathText=True)
    ax.get_yaxis().get_offset_text().set_x(-0.055)

    return fig, ax
