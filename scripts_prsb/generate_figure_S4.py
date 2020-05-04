import numpy as np
import networkx as nx
from scipy import stats
from tqdm import trange
import matplotlib.pyplot as plt
from config import FIGURES_PATH

import er.plot
from er.simulation import TrajectoryGenerator
from er.utils import load_graph, load_data, data_path
from er.model import SwitchingNetwork, ExponentialWalker


# %% Configuration

GRAPH = 'hex'
TAU = 1e9  # “frozen” active network
MAX_TIME = 60
NUM_RUNS = 10  # number of independent simulations to run

# %% Run the required simulations
# ===============================

graph = load_graph(GRAPH)

for n_run in range(1, NUM_RUNS + 1):
    out = data_path(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_uniform_10_max_time{MAX_TIME}_run{n_run}', suffix='.parquet')
    if out.exists():
        continue

    print(f'Generating trajectories: {out}')
    # Start from uniform distribution, 10 particles per node.
    network = SwitchingNetwork(graph, timescale=TAU)
    walker = ExponentialWalker(timescale=0.1)
    num_walkers = 10 * network.graph.number_of_nodes()
    start_nodes = np.repeat(list(network.graph.nodes), 10)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(num_walkers, MAX_TIME,
                                   start_nodes=start_nodes)
    trajs.to_parquet(str(out))


# %% Analyse graph structure: attractors and limit cycles
# =======================================================

network = SwitchingNetwork(graph, timescale=1, memory=False)
ccs = None
while not ccs or len(ccs[0]) == 1:
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes(data=True))
    digraph.add_edges_from(network.edges(0))

    ccs = sorted(nx.attracting_components(digraph), key=len, reverse=True)

for i in [6]:
    component = ccs[0]
    nodes = nx.ancestors(digraph, next(iter(component)))
    nodes |= component
    subgraph = digraph.subgraph(nodes)

    fig, ax = er.plot.graph.structure(graph)
    fig.set_size_inches((4, 4))
    pos = nx.get_node_attributes(graph, 'pos')
    pos = er.plot.graph.get_nodes_pos(digraph)
    nx.draw_networkx(subgraph, nodelist=nodes, pos=pos, with_labels=False, ax=ax,
                     node_size=2, arrowsize=7, width=1,
                     edge_color='firebrick', node_color='firebrick')
    nx.draw_networkx(subgraph.subgraph(component), nodelist=component, pos=pos, with_labels=False,
                     ax=ax,
                     node_size=10, width=1, arrowsize=7, node_color='black')

    component = ccs[i]
    nodes = nx.ancestors(digraph, next(iter(component)))
    nodes |= component
    subgraph = digraph.subgraph(nodes)
    nx.draw_networkx(subgraph, nodelist=nodes, pos=pos, with_labels=False, ax=ax,
                     node_size=2, arrowsize=7, width=1,
                     edge_color='rebeccapurple', node_color='rebeccapurple')
    nx.draw_networkx(subgraph.subgraph(component), nodelist=component, pos=pos, with_labels=False,
                     ax=ax,
                     node_size=10, width=1, arrowsize=7, node_color='black')

    fig.tight_layout(pad=0)

fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_attractors_examples.svg'))

# %% Attractor size distribution
# ==============================

qs = []
for _ in trange(1000):
    network.reset()
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes(data=True))
    digraph.add_edges_from(network.edges(0))

    for component in nx.attracting_components(digraph):
        nodes = nx.ancestors(digraph, next(iter(component)))
        nodes |= component
        subgraph = digraph.subgraph(nodes)
        q = 0
        for node in nodes:
            if digraph.out_degree[node] > 0:
                weight = subgraph.out_degree[node] / digraph.out_degree[node]
            else:
                weight = 1
            q += weight

        qs.append(q / len(component))


fig, ax = plt.subplots(figsize=(4, 4))

attr_dens, bins, _ = ax.hist(qs, bins=np.arange(3, 201, 1), density=True,
                             label='Empirical distribution')
ax.set_xlabel('Attractor mass (# of nodes)')
attr_size = (bins[:-1] + bins[1:]) / 2

β, c = -np.polyfit(attr_size[:100] - 2, np.log(attr_dens[:100]), deg=1)
ax.plot(attr_size[:100], np.exp(-β * (attr_size[:100] - 2) - c), c='navy',
        linestyle='--', label='Exponential fit')
ax.legend()
fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_attractor_size_distribution.svg'))


# %% Distribution of particles
# ============================

M = graph.number_of_nodes()
hist = np.zeros((NUM_RUNS, M + 1))
all_nodes = list(graph.nodes)

# I am taking the average distribution over NUM_RUNS independent simulations
for n_run in range(1, 1 + NUM_RUNS):
    trajs = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_uniform_10_max_time{MAX_TIME}_run{n_run}', suffix='.parquet')

    trajs = trajs.set_index('time', drop=False).sort_index()
    counts = trajs.groupby('id', as_index=False).last().groupby(
        'node').count().reindex(all_nodes, fill_value=0)
    hist[n_run - 1], bins = np.histogram(counts.id, bins=np.arange(M + 2))

hist_norm = np.sum(hist, axis=0) / np.sum(hist)

sim_vals = hist_norm
sim_xvals = (bins[:-1] + bins[1:]) / 2

# Theoretical functions


def p_k_particles(k, N, M, p=1 / 8):
    ms = np.arange(M // 4 + 1)
    ϕ = stats.binom.pmf(ms, M, p)
    ps = np.concatenate(([0], 1 / ms[1:]))

    return np.sum(((1 - ms / M) * (k == 0) + ms / M * stats.binom.pmf(k, N, ps)) * ϕ)


def ρ_c(m):
    if np.isscalar(m):
        m = np.array([m])
    m = np.asarray(m, dtype='float64')

    return np.exp(-β * m - c) * (m >= 2)


def p_node(k):
    m = np.arange(2, 10000)
    return np.sum(ρ_c(m) * (m - 1)) * (k == 0) + (k > 0) * ρ_c(k)


# Figures
ks = np.arange(0, 10 * M) / 10
A = 1 / (p_node(ks)).sum()

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(sim_xvals[:200], sim_vals[:200], width=1, label='Simulation')

ax.scatter(np.arange(0, 200, 10), A * p_node(np.arange(0, 200, 10) / 10),
           c='navy', zorder=3, label='Theoretical')

ax.scatter(np.arange(0, 201, 4), [p_k_particles(k, N, M, p=0.5**deg)
                                  for k in range(0, 201, 4)],
           marker='o', edgecolors='navy', linewidth=1, facecolors='none', zorder=3, label='Theoretical (no topology)')

ax.set_xlabel('Particles per node')
ax.set_yscale('log')
ax.legend()
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_particle_distribution_frozen_active_network.svg'))

# Fluctuation
ks = np.arange(40)
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(ks, stats.binom.pmf(ks - 10, 30, 1 / 3), width=1, align='center')
ax.set_xlabel('Number of particles')
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_particle_fluctuation_frozen_active_network.svg'))
