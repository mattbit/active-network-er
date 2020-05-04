"""Generate a movie of the particles motion."""

import numpy as np
from tqdm import tqdm
from config import FIGURES_PATH

import er.plot as erplot
from er.utils import load_graph, load_data, data_path, reduce_trajs_steps, count_nodes


# %% Configuration
# ================

GRAPH = 'hex'
TAU = 3
NUM_TRAJS = 10000
MAX_TIME = 120


# %% Run required simulation
# ==========================

graph = load_graph(GRAPH)

out = data_path(
    f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_N{NUM_TRAJS}_max_time{MAX_TIME}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = SwitchingNetwork(graph, timescale=TAU)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(NUM_TRAJS, MAX_TIME)
    trajs.to_parquet(str(out))
else:
    trajs = load_data(str(out.with_suffix('')), suffix='.parquet')


# %% Create the movie
# ===================

df = reduce_trajs_steps(trajs).copy()
df['step'] = df.groupby('id').cumcount()  # add step count
df['prev_node'] = df.groupby('id').node.shift().fillna(0).astype(int)

# Generate frames
MAX_TIME = 30
times = np.linspace(0, MAX_TIME, int(MAX_TIME * 20))
values = np.zeros((len(times), graph.number_of_nodes()))
for i, t in enumerate(tqdm(times)):
    for node, count in count_nodes(t, df).iteritems():
        values[i, node] = count
values[0, 0] = NUM_TRAJS  # All particles in source node at time 0

_title = f'$\\tau_{{switch}}$ = {TAU} s,\t$\\tau_{{node}}$ = 100 ms'
ani = erplot.graph.animation(graph, values, times, title=_title)
ani.save(FIGURES_PATH.joinpath(
    f'{GRAPH}_movie_tau{TAU}_max{MAX_TIME}.mp4'), dpi=300)
