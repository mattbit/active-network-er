import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import FIGURES_PATH

import er.plot as erplot
from er.simulation import MFPTSimulation
from er.utils import load_graph, data_path, load_data
from er.model import (SwitchingNetworkConstantRate,
                      UndirectedNetwork, ExponentialWalker)

# %% Configuration
# ================

GRAPH = 'hex'
TAUS = np.concatenate([[0.001],
                       np.arange(0.01, 0.075, step=0.005),
                       np.arange(0.75, 0.3, step=0.01),
                       np.arange(0.3, 2.3, step=0.1)])
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
        f'MFPT/{GRAPH}_SwitchingNetworkConstantRate_ExponentialWalker_tau{tau:e}_sim1000')
    if out.exists():
        continue

    print(f'Running simulation: {out}')
    network = SwitchingNetworkConstantRate(graph, timescale=tau, memory=False)
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

nodes = [node for node, distance in node_distance.items() if distance == 25]

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
        f'MFPT/{GRAPH}_SwitchingNetworkConstantRate_ExponentialWalker_tau{tau:e}_sim1000')
    mfpts.append(df[df.node.isin(nodes_d)].FPT.mean())


# Undirected
df = load_data(f'MFPT/{GRAPH}_UndirectedNetwork_ExponentialWalker')
df = df.join(distances, on='node')
undirected_mfpt = df[df.distance == DISTANCE].FPT.mean()

# %%

fig, ax = plt.subplots()
ax.plot(TAUS, mfpts, label='Active Network')
ax.hlines(undirected_mfpt, 0, max(TAUS), linestyle='--',
          linewidth=1, label='Undirected model')
ax.hlines(2 * undirected_mfpt, 0, max(TAUS), linestyle=':',
          linewidth=1, label='2 × MFPT of undirected model')

ax.set_ylabel(f"MFPT at $d(S, T) = {DISTANCE}$  (s)")
ax.set_xlabel("$\\tau_{switch}$  (s)", usetex=True)
ax.legend()
plt.ylim(0, 3500)

fig.savefig(FIGURES_PATH.joinpath(
    f'MFPT_vs_switching_{DISTANCE}_constant_rate.svg'))
pd.DataFrame({"tau": TAUS, f"MFPT at {DISTANCE}": mfpts}).to_csv(
    FIGURES_PATH.joinpath(f'MFPT_vs_switching_{DISTANCE}_constant_rate.csv'),
    index=False)
plt.show()
