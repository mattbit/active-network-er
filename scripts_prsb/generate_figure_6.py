import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from cmocean import cm
import matplotlib.pyplot as plt
from config import FIGURES_PATH

import er.plot
from er.graph import Trajectory
from er.simulation import TrajectoryGenerator
from er.utils import count_nodes, load_graph, load_data, data_path
from er.model import SwitchingNetwork, UndirectedNetwork, ExponentialWalker


# %% Configuration
# ================

GRAPH = 'hex'
SOURCE = 0
TARGET = 1471
NUM_TRAJS = 10000
MAX_TIME = 120
TAUS = [0.03, 0.3, .3]


# %% Run required simulations
# ===========================

graph = load_graph(GRAPH)

for tau in TAUS:
    out = data_path(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
    if out.exists():
        continue

    print(f'Generating trajectories: {out}')
    network = SwitchingNetwork(graph, timescale=tau)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(NUM_TRAJS, MAX_TIME)
    trajs.to_parquet(str(out))


out = data_path(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = UndirectedNetwork(graph)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(NUM_TRAJS, MAX_TIME)
    trajs.to_parquet(str(out))


# %% Create kymographs
# ====================

def create_kymographs(trajs, path, graph):
    trajs = trajs.sort_values(['id', 'time'])
    node_distance = nx.shortest_path_length(graph, 0)
    distances = pd.DataFrame.from_dict(node_distance,
                                       orient='index', columns=['distance'])
    distances.index.name = 'node'

    # Calculate node occupation.
    times = np.arange(0, 10., .05)
    counts = np.zeros((graph.number_of_nodes(), len(times)), dtype=int)
    for i, t in enumerate(tqdm(times)):
        for node, count in count_nodes(t, trajs).iteritems():
            counts[node, i] = count

    assert counts[0, 0] == NUM_TRAJS  # all particles in the source at time 0

    # Figure for path occupation
    path_nodes = np.array(path.nodes)

    fig_path, ax_path = plt.subplots()
    ax_path.pcolormesh(times, np.arange(path_nodes.size), counts[path_nodes],
                       cmap=cm.thermal, vmin=0, vmax=100)
    ax_path.set_xlabel('Time (s)')
    ax_path.set_ylabel('Distance from source')

    # Figure for average occupation
    d_counts = np.zeros((len(path.nodes), len(times)))
    dists = distances[distances.distance < d_counts.shape[0]]

    for d, _group in dists.groupby('distance'):
        d_counts[d] = counts[_group.index.values].mean(axis=0)

    fig_avg, ax_avg = plt.subplots()
    ax_avg.pcolormesh(times, np.arange(d_counts.shape[0]), d_counts,
                      cmap=cm.thermal, vmin=0, vmax=100)

    ax_avg.set_xlabel('Time (s)')
    ax_avg.set_ylabel('Distance from source')

    return fig_path, fig_avg


# %% Plot source and target on the graph
# ======================================

fig, ax = er.plot.graph.structure(graph)
er.plot.graph.node_label(ax, graph, SOURCE, 'S')
er.plot.graph.node_label(ax, graph, TARGET, 'T')
fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_kymograph_path.svg'))

nodes = [node for node, dist in nx.shortest_path_length(graph, 0).items()
         if dist == 15]
fig, ax = er.plot.graph.structure(graph)
er.plot.graph.node_label(ax, graph, 0, 'S')
for node in nodes:
    er.plot.graph.node_label(ax, graph, node)
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_kymograph_nodes_at_distance_15.svg'))


# %% Switching network
# ====================

path = Trajectory(nx.shortest_path(graph, SOURCE, TARGET))

for tau in TAUS:
    trajs = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
    fig_path, fig_avg = create_kymographs(trajs, path, graph)
    fig_path.savefig(FIGURES_PATH.joinpath(
        f'{GRAPH}_kymograph_tau{tau:.2f}_path.svg'))
    fig_avg.savefig(FIGURES_PATH.joinpath(
        f'{GRAPH}_kymograph_tau{tau:.2f}_avg.svg'))


# %% Undirected network
# =====================

trajs = load_data(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
fig_path, fig_avg = create_kymographs(trajs, path, graph)
fig_path.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_kymograph_undirected_path.svg'))
fig_avg.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_kymograph_undirected_avg.svg'))
