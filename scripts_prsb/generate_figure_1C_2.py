"""Generate figure 1C (Mean First Passage Time).

Produces the MFPT heatmaps and plots for figure 2. The network model has
switching edges (timescale of 30 ms) and the walker waits an exponential time
in the nodes (timescale of 100 ms).
The result is calculated by averaging 5000 simulations.

The same code can be used to generate figure 2, by changing the configuration.
"""

import cmocean
import pandas as pd
from tqdm import tqdm
import networkx as nx
from config import FIGURES_PATH
import matplotlib.pyplot as plt

import er.plot as erplot
from er.utils import vals_by_dist
from er.simulation import MFPTSimulation
from er.utils import load_graph, load_data, data_path
from er.model import SwitchingNetwork, ExponentialWalker


# %% Configuration
# ================

TAU = 0.03
VMAX = 3000
GRAPH = "er"


# %% Run required simulations
# ===========================

graph = load_graph(GRAPH)

out = data_path(f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}')
if not out.exists():
    network = SwitchingNetwork(graph, timescale=TAU, memory=False)
    walker = ExponentialWalker(timescale=0.1)
    sim = MFPTSimulation(network, walker, num_sims=5000)
    print(f'Running simulation: {out}')
    res = sim.run()
    res.to_csv(str(out))


# %% MFPT heatmap
# ===============

data = load_data(f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}')
# data = load_data(f'MFPT/{GRAPH}_UndirectedNetwork_ExponentialWalker')

mfpt = data.groupby("node").FPT.mean()
fig, ax, cb = erplot.graph.heatmap(
    graph, mfpt, cmap=cmocean.cm.delta, vmax=VMAX)

erplot.graph.node_label(ax, graph, 0, "S", size=5)
cb.remove()
fig.savefig(FIGURES_PATH.joinpath("MFPT_heatmap_{}.svg".format(GRAPH)))
plt.show()

fig, ax, cb = erplot.graph.heatmap(
    graph, mfpt, cmap=cmocean.cm.delta, vmax=VMAX)
ax.remove()
cb.ax.set_ylabel("$\\bar{\\tau}_{S \\to T}$   (s)")
fig.savefig(FIGURES_PATH.joinpath("MFPT_heatmap_colorbar.svg"))


# %% Distance plots
# =================

mfpt = data.groupby("node").FPT.mean()
fig, ax = erplot.graph.by_distance(
    graph, mfpt, cmap=cmocean.cm.delta, vmax=VMAX)

ax.set_ylim([0, VMAX])
ax.set_xlim([0, 50])
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_ylabel("$\\bar{\\tau}_{S \\to T}$   (s)")
ax.set_xlabel("$d(S, T)$")

fig.savefig(FIGURES_PATH.joinpath("MFPT_plot_{}.svg".format(GRAPH)),
            pad_inches=0.012)
plt.show()

# %% Extra plots for MFPT related quantities
# ==========================================

means = vals_by_dist(graph, mfpt).groupby(
    'distance', as_index=False).value.mean()

fig, ax = plt.subplots()
ax.plot(means.distance, means.value)
ax.set_yscale('log')
ax.set_xlabel('Distance d(S, T)')
ax.set_ylabel('Average MFPT (log scale)')
fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_MFPT_log_scale.svg'))

# %%

conn = [nx.node_connectivity(graph, 0, node) for node in tqdm(graph.nodes)]

dist_ = pd.DataFrame({'node': list(graph.nodes), 'distance': list(
    nx.shortest_path_length(graph, 0).values())})
conn_ = pd.DataFrame({'node': list(graph.nodes), 'connectivity': conn})
mfpt_ = pd.DataFrame(mfpt).reset_index()
data = mfpt_.merge(dist_, on='node').merge(conn_, on='node')
conn_by_dist = data.groupby('distance').connectivity.mean().reset_index()

fig, ax = plt.subplots()
ax.plot(conn_by_dist.distance, conn_by_dist.connectivity)
ax.set_xlabel('Distance')
ax.set_ylabel('Average node connectivity')
fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_node_connectivity_by_dist.svg'))


fig, ax, cb = erplot.graph.heatmap(
    graph, conn_.connectivity, cmap=cmocean.cm.thermal)
fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_node_connectivity_heatmap.svg'))
