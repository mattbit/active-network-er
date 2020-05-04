import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import FIGURES_PATH

from er.simulation import TrajectoryGenerator
from er.utils import load_graph, load_data, data_path
from er.model import SwitchingNetwork, UndirectedNetwork, ExponentialWalker


# %% Configuration
# ================

TAUS = np.concatenate([[0.001],
                       np.arange(0.01, 3.01, step=0.1)])

N = 1000
MAX_TIME = 120

GRAPH = 'hex'
TARGET = 146

# %% Run the required simulations
# ===============================

graph = load_graph(GRAPH)

out = data_path(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{N}_max_time{MAX_TIME}', suffix='.parquet')
if not out.exists():
    print(f'Generating trajectories: {out}')
    network = UndirectedNetwork(graph)
    walker = ExponentialWalker(timescale=0.1)
    generator = TrajectoryGenerator(network, walker)
    trajs = generator.trajectories(N, MAX_TIME)
    trajs.to_parquet(str(out))


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


# %% Produce the figures
# ======================

def theoretical_trapping_prob(tau_switch, tau_wait=0.1):
    q = tau_wait / tau_switch
    numerator = 24 * q**3 + 34 * q**2 + 11 * q + 1
    denominator = 192 * q**3 + 130 * q**2 + 23 * q + 1
    return numerator / denominator


def theoretical_backtracking_prob(tau_switch, tau_wait=0.1):
    q = tau_wait / tau_switch
    p_trap = theoretical_trapping_prob(tau_switch, tau_wait)
    return (1 - p_trap) * (7 / 12) * q / (2 * q + 1)


def trapping_fraction(traj):
    return (traj.node.diff() == 0).mean()


def backtracking_fraction(traj):
    n0 = traj.node
    n1 = traj.node.shift(-1)
    n2 = traj.node.shift(-2)

    return ((n0 == n2) & (n0 != n1)).mean()


trajs = load_data(
    f'trajs/{GRAPH}_UndirectedNetwork_ExponentialWalker_N{N}_max_time{MAX_TIME}',
    suffix='.parquet')
trajs = trajs.sort_values(['id', 'time'])
bt_avg_u = np.mean([backtracking_fraction(traj)
                    for _, traj in trajs.groupby('id')])
tp_avg_u = np.mean([trapping_fraction(traj)
                    for _, traj in trajs.groupby("id")])

bt_avg = []
tp_avg = []
for tau in tqdm(TAUS):
    trajs = load_data(
        f'trajs/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{tau:e}_N{N}_max_time{MAX_TIME}',
        suffix='.parquet')
    bt_avg.append(trajs.groupby('id').apply(backtracking_fraction).mean())
    tp_avg.append(trajs.groupby('id').apply(trapping_fraction).mean())
tp_avg = np.array(tp_avg)
bt_avg = np.array(bt_avg)

# %%

fig, ax = plt.subplots()
ax.plot(TAUS, tp_avg)
ax.plot(TAUS, theoretical_trapping_prob(TAUS), linestyle="--")
ax.set_xlabel('$\\tau_{\\mathit{switch}}$')
fig.savefig(FIGURES_PATH.joinpath('trapping_prob.svg'))
plt.show()

# %%

fig, ax = plt.subplots()
ax.plot(TAUS, bt_avg)
ax.plot(TAUS, theoretical_backtracking_prob(TAUS), linestyle="--")
ax.set_xlabel('$\\tau_{\\mathit{switch}}$')
fig.savefig(FIGURES_PATH.joinpath('backtracking_prob.svg'))
plt.show()
# %%
