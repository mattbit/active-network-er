import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import FIGURES_PATH

from er.simulation import TrajectoryGenerator
from er.utils import (load_graph, load_data, data_path,
                      collapse_traps, count_by_source)
from er.model import SwitchingNetwork, ExponentialWalker


# %% Configuration
# ================

TAUS = [0.03, 0.3, 3]
N = 10000
MAX_TIME = 120

GRAPH = 'hex'
TARGET = 146

# %% Run the required simulations
# ===============================

graph = load_graph(GRAPH)

for tau in TAUS:
    out = data_path(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{N}_max_time{MAX_TIME}', suffix='.parquet')
    if out.exists():
        continue

    print(f'Generating trajectories: {out}')
    network = SwitchingNetwork(graph, timescale=tau)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(N, MAX_TIME)
    trajs.to_parquet(str(out))


# %% Generate the figures
# =======================

for tau in TAUS:
    df = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{N}_max_time{MAX_TIME}',
        suffix='.parquet')

    df = df.sort_values(["id", "time"])
    df = collapse_traps(df)
    df["step"] = df.groupby("id").cumcount()  # add step count
    df["prev_node"] = df.node.shift()

    ts = np.linspace(0, 120, 1200)
    neighbors, count = count_by_source(TARGET, ts, df, graph)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.stackplot(ts, count, labels=neighbors,
                 colors=['#1b9e77', '#d95f02', '#7570b3'])
    ax.plot(ts, pd.Series(count.sum(axis=0)).rolling(
        50).mean(), linestyle="--", c="navy", linewidth=1)
    ax.legend()
    plt.show()

    fig.savefig(FIGURES_PATH.joinpath(f'{GRAPH}_count_in_node_tau{tau}.svg'))
