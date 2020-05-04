"""Generate figure 1D (MFPT vs switching timescale)."""

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import FIGURES_PATH

import er.plot as erplot
from er.simulation import MFPTSimulation
from er.utils import load_graph, data_path, load_data
from er.model import SwitchingNetwork, UndirectedNetwork, ExponentialWalker


# %% Configuration
# ================

GRAPH = 'er'
TAUS = np.concatenate([[0.001],
                       np.arange(0.01, 0.3, step=0.05),
                       np.arange(0.3, 3.01, step=0.1)])
DISTANCE = 25


# %% Run the required simulations
# ===============================

graph = load_graph(GRAPH)


out = data_path(f'MFPT/{GRAPH}_UndirectedNetwork_ExponentialWalker')
if not out.exists():
    network = UndirectedNetwork(graph)
    walker = ExponentialWalker(timescale=0.1)
    sim = MFPTSimulation(network, walker, num_sims=5000)
    results = sim.run()
    results.to_csv(str(out))

for tau in TAUS:
    out = data_path(
        f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}')
    if out.exists():
        continue

    print(f'Running simulation: {out}')
    network = SwitchingNetwork(graph, timescale=tau, memory=False)
    walker = ExponentialWalker(timescale=0.1)
    sim = MFPTSimulation(network, walker, num_sims=1000)
    results = sim.run()
    results.to_csv(str(out))


# %% Plot the nodes at the given distance
# =======================================

node_distance = nx.shortest_path_length(graph, 0)
distances = pd.DataFrame.from_dict(
    node_distance, orient='index', columns=['distance'])
distances.index.name = 'node'

nodes = [node
         for node, distance in node_distance.items() if distance == DISTANCE]

fig, ax = erplot.graph.structure(graph)
erplot.graph.node_label(ax, graph, 0, label='0')
for node in nodes:
    erplot.graph.node_label(ax, graph, node, label='')
plt.show()


# %% MFPT vs switching rate
# =========================

graph = load_graph(GRAPH)
mfpts = []

nodes_d = distances[distances.distance == DISTANCE].index.values

for tau in tqdm(TAUS):
    df = load_data(
        f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}')
    mfpts.append(df[df.node.isin(nodes_d)].FPT.mean())

df = load_data(f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{0.03:e}')
active30ms_mfpt = df[df.node.isin(nodes_d)].FPT.mean()

df = load_data(f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{0.3:e}')
active300ms_mfpt = df[df.node.isin(nodes_d)].FPT.mean()

df = load_data(f'MFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{3:e}')
active3s_mfpt = df[df.node.isin(nodes_d)].FPT.mean()

# Undirected
df = load_data(f'MFPT/{GRAPH}_UndirectedNetwork_ExponentialWalker')
df = df.join(distances, on='node')
undirected_mfpt = df[df.distance == DISTANCE].FPT.mean()


fig, ax = plt.subplots()
ax.plot(TAUS, mfpts, label='Active Network')
ax.hlines(undirected_mfpt, 0, max(TAUS), linestyle='--',
          linewidth=1, label='Undirected model')

ax.set_ylabel(f"MFPT at $d(S, T) = {DISTANCE}$  (s)")
ax.set_xlabel("$\\tau_{\\mathit{switch}}$  (s)", usetex=True)
ax.legend()
plt.ylim(0, max(mfpts))

fig.savefig(FIGURES_PATH.joinpath(f'MFPT_vs_switching_{DISTANCE}.svg'))
pd.DataFrame({"tau": TAUS, f"MFPT at {DISTANCE}": mfpts}).to_csv(
    FIGURES_PATH.joinpath(f'MFPT_vs_switching_{DISTANCE}.csv'), index=False)

plt.show()

print(f'Undirected network: {undirected_mfpt/60:.2f} minutes')
print(f'Active network (30 ms): {active30ms_mfpt/60:.2f} minutes')
print(f'Active network (300 ms): {active300ms_mfpt/60:.2f} minutes')
print(f'Active network (3 s): {active3s_mfpt/60:.2f} minutes')
