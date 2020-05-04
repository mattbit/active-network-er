import numpy as np
import pandas as pd
from tqdm import tqdm
from cmocean import cm
from scipy import stats
import matplotlib.pyplot as plt
from config import FIGURES_PATH

import er.plot
from er.simulation import TrajectoryGenerator
from er.utils import load_graph, load_data, data_path
from er.model import UndirectedNetwork, SwitchingNetwork, ExponentialWalker


# %% Configuration
# ================

GRAPH = 'hex'
TAU = 3.
NUM_TRAJS = 10000
MAX_TIME = 200


# %% Run the required simulations
# ===============================

graph = load_graph(GRAPH)

# Undirected network
out = data_path(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = UndirectedNetwork(graph)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(NUM_TRAJS, MAX_TIME)
    trajs.to_parquet(str(out))

# Active network
out = data_path(
    f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = SwitchingNetwork(graph, timescale=TAU)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(NUM_TRAJS, MAX_TIME)
    trajs.to_parquet(str(out))


# If you're skeptical like me you may think that in the active network the time
# required to reach the steady-state is sooo long that we can't see it in our
# simulations, and what we're plotting is just a transient state and maybe
# it's just a question of waiting more to see a uniform distribution.
# To verify that this is not true, you can play with different initial states
# for the simulation. For example, trajectories can be generated starting from
# a uniform distribution (instead of releasing all particles in a single node),
# this can be used to verify that the steady state of the active network is
# indeed non-uniform. In fact even if we start from the uniform state, the
# distribution of particles will not be maintained, it will deviate from the
# nice uniform state.
# Here is some ready-to-use code if you want to generate trajectories in this
# way, starting with 5 particles in each node in the network.
#
# -----8<----------8<----------8<----------8<----------8<----------8<----------
# network = SwitchingNetwork(graph, timescale=TAU)
# walker = ExponentialWalker(timescale=0.1)
# num_walkers = 5 * network.graph.number_of_nodes()
# start_nodes = np.repeat(list(network.graph.nodes), 5)
# generator = TrajectoryGenerator(network, walker)
# trajs = generator.trajectories(num_walkers, TIME_MAX, start_nodes=start_nodes)
# -----8<----------8<----------8<----------8<----------8<----------8<----------


# %% Plotting helpers
# ===================

def plot_particle_distribution(trajs, times, nodes, num_bins=101,
                               snapshots=[0, 10]):
    times = np.asarray(times)
    bins = np.arange(num_bins)

    heatmap = np.zeros((len(times), len(bins) - 1))
    trajs = trajs.set_index('time').sort_index()

    for i, time in enumerate(tqdm(times)):
        dx = trajs[:time].groupby('id', as_index=False).last()
        trajs
        counts = dx.groupby('node').id.count().reindex(all_nodes, fill_value=0)
        heatmap[i], _ = np.histogram(counts, bins=bins, density=True)

    # Snapshots (particle distribution at a fixed time)
    fig_snapshots, axes = plt.subplots(
        nrows=len(snapshots), sharex=True, figsize=(4, 4))

    for ax, t in zip(axes, snapshots):
        ax.bar(bins[:-1], heatmap[t], width=1,
               color=cm.thermal(heatmap[t].clip(0, .25) / .25))
        ax.set_ylim(0, 0.25)
        ax.set_ylabel(f't = {t} s')

    axes[0].set_ylim(0, 1)
    ax.set_xlim(-0.5, 20)
    ax.set_xlabel('Particles per node')

    # Time evolution heatmap
    fig_heatmap, ax_h = plt.subplots(figsize=(4, 3))
    ax_h.pcolormesh(bins[:-1] - 0.5, times, heatmap,
                    vmax=0.25, cmap=cm.thermal)
    ax_h.set_xlim(-0.5, 20)
    ax_h.set_ylabel('Time (s)')
    ax_h.set_xlabel('Particles per node')
    ax_h.invert_yaxis()

    return fig_heatmap, fig_snapshots


# %% Prepare the trajectories
# ===========================
nodes = pd.DataFrame({'node': node, 'x': attrs['x'], 'y': attrs['y']}
                     for node, attrs in graph.nodes(data=True))
all_nodes = list(graph.nodes)

# Load the trajectories
trajs_active = load_data(
    f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')

trajs_undirected = load_data(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')


# %% Plot the time evolution of the distributions
# ===============================================

times = np.arange(MAX_TIME + 1)

# Undirected
fig_time_evol, fig_snapshots = plot_particle_distribution(
    trajs_undirected, times, all_nodes, snapshots=[0, 30, 60, 200])
fig_snapshots.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_particle_distribution_snapshots_undirected.svg'))
fig_time_evol.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_particle_distribution_time_evolution_undirected.svg'))

# Active network
fig_time_evol, fig_snapshots = plot_particle_distribution(
    trajs_active, times, all_nodes, snapshots=[0, 30, 60, 200])
fig_snapshots.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_particle_distribution_snapshots_active_tau{TAU:e}.svg'))
fig_time_evol.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_particle_distribution_time_evolution_active_tau{TAU:e}.svg'))


# %% Steady state distributions
# =============================

# Realisations on the network (undirected)
dx = trajs_undirected.groupby('id', as_index=False).last()
counts = dx.groupby('node').count().reindex(all_nodes, fill_value=0)
counts = counts.join(nodes, on='node')

fig, ax = plt.subplots(figsize=(4, 4))
er.plot.graph.heatmap(graph, counts.id, ax=ax, vmin=0, vmax=10)
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_realisation_steady_state_undirected.svg'))

# Distribution and fit (undirected)
N = counts.id.sum()
M = graph.number_of_nodes()

bins = np.arange(22) - 0.5
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(counts.id, bins=bins, density=True, rwidth=0.8, label='Simulation')
ax.scatter(np.arange(21), [stats.binom.pmf(k, N, 1 / M) for k in range(21)],
           marker='.', c='navy', zorder=2, s=49,
           label='Theoretical')
ax.legend()
ax.set_xlabel('Particles per node')
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_steady_state_distribution_undirected.svg'))


# Realisations on the network (active network)
dx = trajs_active.groupby('id', as_index=False).last()
counts = dx.groupby('node').count().reindex(all_nodes, fill_value=0)
counts = counts.join(nodes, on='node')

fig, ax = plt.subplots(figsize=(4, 4))
er.plot.graph.heatmap(graph, counts.id, ax=ax, vmin=0, vmax=10)
fig.tight_layout(pad=0)
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_realisation_steady_state_active_tau{TAU:e}.svg'))

# Distribution (active)
N = counts.id.sum()
M = graph.number_of_nodes()

bins = np.arange(22) - 0.5
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(counts.id, bins=bins, density=True, rwidth=0.8, label='Simulation')
ax.legend()
ax.set_xlabel('Particles per node')
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_steady_state_distribution_active_tau{TAU:e}.svg'))
