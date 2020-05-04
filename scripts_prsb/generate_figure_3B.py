"""Extreme First Passage Time.

Produces the MEFPT plots for figure 3. The network model has switching edges
(timescale of 30 ms) and the walker waits an exponential time in the nodes
(timescale of 100 ms). The result is based on 5000 simulations.
"""

import numpy as np
from config import FIGURES_PATH
from scipy.optimize import leastsq

import er.plot as erplot
from er.simulation import MEFPTSimulation
from er.model import SwitchingNetwork, ExponentialWalker
from er.utils import data_path, load_graph, load_data, vals_by_dist


# %% Configuration
# ================

TAU = 0.03
NS = [100, 1000, 10000]

GRAPH = 'er'
TARGET = 146


# %% Run required simulations
# ===========================

graph = load_graph(GRAPH)

for n in NS:
    out = data_path(
        f'MEFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_N{n}')
    if out.exists():
        continue

    print(f'Running simulation: {out}')
    network = SwitchingNetwork(graph, timescale=TAU)
    walker = ExponentialWalker(timescale=0.1)
    sim = MEFPTSimulation(network, walker, n, num_sims=5000)
    res = sim.run()
    res.to_csv(str(out))


# %% MEFPT for different number of walkers
# ========================================

xs = []
ys = []
labels = []
for n in NS:
    data = load_data(
        f'MEFPT/{GRAPH}_SwitchingNetwork_ExponentialWalker_tau{TAU:e}_N{n}')
    mefpt = data.groupby('node').EFPT.mean()
    df = vals_by_dist(graph, mefpt)
    df = df.groupby('distance', as_index=False).value.mean()

    xs.append(df.distance)
    ys.append(df.value)
    labels.append(f'N = {n}')

fig, ax = erplot.comparison(xs, ys, labels)
ax.set_xlabel('$d(S, X)$')
ax.set_ylabel('$\\tau^{ex}_S(X)$ [s]')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.grid(color='.9', axis='y')


# %% Fit of the log law
# =====================

def log_fit_N(d, N, c1, c2):
    return c1 * np.power(d, 2) / (c2 + np.log(N))


def fit_error(params, func, xs, ys, ns):
    errors = [y - func(x, n, *params)
              for x, y, n in zip(xs, ys, ns)]

    return np.concatenate(errors)


(c1, c2), _ = leastsq(fit_error, (1, 1), args=(
    log_fit_N, xs[:30], ys[:30], [100, 1000]))

print("""c1 = {:.2f}\nc2 = {:.2f}""".format(c1, c2))

for x, n in zip(xs, NS):
    ax.plot(x, log_fit_N(x, n, c1, c2), linewidth=1, linestyle="--", c="navy")


fig.savefig(FIGURES_PATH.joinpath(f'MEFPT_plot_{GRAPH}.svg'))
