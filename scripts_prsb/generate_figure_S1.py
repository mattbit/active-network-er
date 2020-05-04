import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange
from config import FIGURES_PATH
import matplotlib.pyplot as plt

from er.simulation import TrajectoryGenerator
from er.utils import load_graph, load_data, data_path, reduce_trajs_steps
from er.model import UndirectedNetwork, SwitchingNetwork, ExponentialWalker


# %% Configuration
# ================

GRAPH = 'hex'
NUM_TRAJS = 1000
TARGET = 146
TAUS = [0.03, 0.3, 3.]
NUM_SIMS = 1000


# %% Run required simulations
# ===========================

graph = load_graph(GRAPH)
distance = nx.shortest_path_length(graph, 0, TARGET)

for tau in TAUS:
    out = data_path(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{NUM_TRAJS}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    if out.exists():
        continue

    print(f'Generating trajectories: {out}')

    network = SwitchingNetwork(graph, timescale=tau)
    walker = ExponentialWalker(timescale=0.1)

    # We generate simulations with NUM_TRAJS particles and keep only the first
    # to arrive. We repeat this NUM_SIMS times to obtain the statistics of the
    # first particle to arrive.
    trajs = []
    for sim_id in trange(NUM_SIMS):
        network.reset()
        walker.reset()
        generator = TrajectoryGenerator(network, walker)
        traj = generator.trajectories_to_target(NUM_TRAJS, TARGET, keep=1)
        traj['sim_id'] = sim_id
        trajs.append(traj)

    df = pd.concat(trajs)
    df.to_parquet(str(out))

# Same for the undirected network
out = data_path(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = UndirectedNetwork(graph, timescale=tau)
    walker = ExponentialWalker(timescale=0.1)
    trajs = []
    for sim_id in trange(NUM_SIMS):
        network.reset()
        walker.reset()
        generator = TrajectoryGenerator(network, walker)
        traj = generator.trajectories_to_target(NUM_TRAJS, TARGET, keep=1)
        traj['sim_id'] = sim_id
        trajs.append(traj)

    df = pd.concat(trajs)
    df.to_parquet(str(out))


# Generate first 1000 trajs reaching the target

tau = 3.
out = data_path(
    f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N1000_trajectories_S0_T{TARGET}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')

    network = SwitchingNetwork(graph, timescale=tau)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories_to_target(1000, TARGET)
    trajs.to_parquet(str(out))


out = data_path(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N1000_trajectories_S0_T{TARGET}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = UndirectedNetwork(graph, timescale=tau)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories_to_target(1000, TARGET)
    trajs.to_parquet(str(out))


# Simulations for N = [100, 1000, 10000]
tau = 3.
for num_trajs in [100, 1000, 10000]:
    out = data_path(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{num_trajs}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    if out.exists():
        print(f'Generating trajectories: {out}')

        network = SwitchingNetwork(graph, timescale=tau)
        walker = ExponentialWalker(timescale=0.1)

        trajs = []
        for sim_id in trange(NUM_SIMS):
            network.reset()
            walker.reset()
            generator = TrajectoryGenerator(network, walker)
            traj = generator.trajectories_to_target(num_trajs, TARGET, keep=1)
            traj['sim_id'] = sim_id
            trajs.append(traj)

        df = pd.concat(trajs)
        df.to_parquet(str(out))

    # Same for the undirected network
    out = data_path(
        f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{num_trajs}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    if not out.exists():
        print(f'Generating trajectories: {out}')
        network = UndirectedNetwork(graph, timescale=tau)
        walker = ExponentialWalker(timescale=0.1)
        trajs = []
        for sim_id in trange(NUM_SIMS):
            network.reset()
            walker.reset()
            generator = TrajectoryGenerator(network, walker)
            traj = generator.trajectories_to_target(num_trajs, TARGET, keep=1)
            traj['sim_id'] = sim_id
            trajs.append(traj)

        df = pd.concat(trajs)
        df.to_parquet(str(out))


# %% Length of fastest
# ====================

df = load_data(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
dfagg = df.groupby(['sim_id', 'id']).agg({'node': 'count', 'time': 'max'})

fig, ax = plt.subplots()
bins = range(0, dfagg.node.max() + dfagg.node.max() % 2 + 1, 2)
ax.hist(dfagg.node, bins, density=True,
        label='Undirected network', color='gray')

for tau in TAUS:
    df = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{NUM_TRAJS}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    dfagg = df.groupby(['sim_id', 'id']).agg({'node': 'count', 'time': 'max'})
    dfred = df.groupby(['sim_id', 'id'], as_index=False).apply(
        reduce_trajs_steps)
    dfagg = dfred.groupby(['sim_id', 'id']).agg(
        {'node': 'count', 'count': 'sum', 'time': 'max'})

    bins = range(0, dfagg.node.max() + dfagg.node.max() % 2 + 1, 2)
    ax.hist(dfagg.node, bins, density=True,
            label=f'Active Network ($\\tau$ = {tau} s)', alpha=0.72)

ax.legend()
ax.set_xlim(0, 100)
ax.set_xlabel('Path length (# of edges)')
ax.set_ylabel('Density')

fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_length_fastest_traj.svg'))


# %% Arrival time of the fastest
# ==============================

df = load_data(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{NUM_TRAJS}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
dfagg = df.groupby(['sim_id', 'id']).agg({'node': 'count', 'time': 'max'})
num_sims = df.sim_id.unique().size

fig, ax = plt.subplots()
ax.hist(dfagg.time, bins=50, range=(0, 10), density=True,
        label='Undirected network', color='gray')
for tau in TAUS:
    df = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{NUM_TRAJS}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    dfagg = df.groupby(['sim_id', 'id']).agg({'node': 'count', 'time': 'max'})

    ax.hist(dfagg.time, bins=50, range=(0, 10),
            label=f'Active Network ($\\tau$ = {tau} s)', alpha=0.72)

ax.legend()
ax.set_xlabel('Arrival time')
ax.set_ylabel('Density')
fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_arrival_time_fastest_traj.svg'))


# %% Trajectory characteristics
# =============================

trajs = load_data(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N1000_trajectories_S0_T{TARGET}', suffix='.parquet')
trajs = trajs.sort_values('time')

dfred = trajs.groupby('id', as_index=False).apply(reduce_trajs_steps)
dfred['delta_t'] = dfred.groupby('id').time.diff().values
sorted_trajs = dfred.groupby('id').agg(
    {'node': 'count', 'time': 'max', 'delta_t': 'mean'}).sort_values('time').reset_index()
sorted_trajs = sorted_trajs.rename(columns={'node': 'length'})
res = {'undirected': sorted_trajs}

tau = 3.
trajs = load_data(
    f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N1000_trajectories_S0_T{TARGET}', suffix='.parquet')
trajs = trajs.sort_values('time')

dfred = trajs.groupby('id', as_index=False).apply(reduce_trajs_steps)
dfred['delta_t'] = dfred.groupby('id').time.diff().values
sorted_trajs = dfred.groupby('id').agg(
    {'node': 'count', 'time': 'max', 'delta_t': 'mean'}).sort_values('time').reset_index()
sorted_trajs = sorted_trajs.rename(columns={'node': 'length'})
res['active'] = sorted_trajs


fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(res['undirected'].index, res['undirected'].delta_t * 1000,
        linewidth=1, label='Undirected network')
ax.plot(res['active'].index, res['active'].delta_t * 1000,
        linewidth=1, label='Active network ($\\tau$ = 3 s)')

ax.set_yticks(np.arange(0, 600, 100))
ax.set_ylim(0)
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Order of arrival')
ax.legend()

fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_trajectory_characteristics_time.svg'))


fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(res['undirected'].index, res['undirected'].length,
        linewidth=1, label='Undirected network')
ax.plot(res['active'].index, res['active'].length,
        linewidth=1, label='Active network ($\\tau$ = 3 s)')

ax.set_yscale('log')
ax.set_ylabel('Length (# of edges)')
ax.set_xlabel('Order of arrival')
ax.legend()

fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_trajectory_characteristics_length.svg'))


# %% Distribution of length based on number of particles
# ======================================================

tau = 3.

fig, ax = plt.subplots()
for n in [10000, 1000, 100]:
    trajs = load_data(
        f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{num_trajs}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    dfred = trajs.groupby(['sim_id', 'id'], as_index=False).apply(
        reduce_trajs_steps)

    dfagg = trajs.groupby(['sim_id', 'id']).agg(
        {'node': 'count', 'time': 'max'})

    bins = range(0, 110, 2)
    ax.hist(dfagg.node, bins, density=True, label=f'N = {n}', alpha=0.72)

ax.set_xlabel('Path length (# of edges)')
ax.legend()
fig.savefig(f'{GRAPH}_length_fastest_traj_by_N_undirected.svg')


fig, ax = plt.subplots()
for n in [10000, 1000, 100]:
    trajs = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{num_trajs}_1st_trajectories_S0_T{TARGET}', suffix='.parquet')
    dfred = trajs.groupby(['sim_id', 'id'], as_index=False).apply(
        reduce_trajs_steps)
    dfagg = dfred.groupby(['sim_id', 'id']).agg(
        {'node': 'count', 'time': 'max'})

    bins = range(0, 105, 2)
    ax.hist(dfagg.node, bins, density=True, label=f'N = {n}', alpha=0.72)

ax.set_xlabel('Path length (# of edges)')
ax.legend()
fig.savefig(FIGURES_PATH.joinpath(
    f'{GRAPH}_length_fastest_traj_by_N_tau{tau}.svg'))
