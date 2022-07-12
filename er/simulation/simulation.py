import abc
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils import unique_id
from ..model.walker import Walker
from ..model.network import NetworkModel


class Simulation:
    """The base abstract class for simulations.

    Parameters
    ----------
    walker : er.model.walker.Walker
        A walker agent that moves on the network.
    network : er.model.network.NetworkModel
        The network model used for simulating the motion.
    simulations : int
        The number of simulations to perform. Default is 1000.
    """

    def __init__(self, network: NetworkModel, walker: Walker, num_sims=1000):
        self.network = network
        self.walker = walker
        self.num_sims = num_sims

    @abc.abstractmethod
    def run():
        """Run the simulation."""


class ConcurrentSimulation(Simulation):
    """The base class for concurrent simulations."""

    def run(self, progress=True):
        """Runs the simulations concurrently."""
        results = []
        with ProcessPoolExecutor() as e:
            futures = [e.submit(self._run, i) for i in range(self.num_sims)]

            pbar = tqdm(total=self.num_sims, disable=(not progress))
            pbar.update(0)
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
            pbar.close()

        return pd.concat(results)

    @abc.abstractmethod
    def _run(self):
        """Run a single simulation."""


class MFPTSimulation(ConcurrentSimulation):
    """Mean First Passage Time Simulation.

    Simulates independent walkers on a graph and finds the Mean First Passage
    Time for all the nodes. Since all the walkers move on independent networks
    this simulation works well with memoryless implementations of the network.

    Parameters
    ----------
    walker : er.model.walker.Walker
        A walker agent that moves on the network.
    network : er.model.network.NetworkModel
        The network model used for simulating the motion.
    simulations : int
        The number of simulations to perform.
    """

    def _run(self, id=None):
        if id is None:
            id = unique_id()

        network = self.network
        walker = self.walker

        fpt = np.full(self.network.size(), np.inf)  # first passage times
        fpt[walker.node] = walker.time

        # Run the simulation.
        while np.any(fpt == np.inf):
            node, time = walker.step(network)

            # Update the first passage times.
            if fpt[node] > time:
                fpt[node] = time

        data = pd.DataFrame({"id":   id,
                             "node": range(len(fpt)),
                             "FPT":  fpt})

        return data


class MEFPTSimulation(ConcurrentSimulation):
    """Mean Extreme First Passage Time simulation.

    Simulates many particles and calculates the mean time required for the
    first one to hit the target. It is a generalization of the MFPT simulation
    for multiple particles which are diffusing at the same time.
    This models an activation process where one particle is sufficient to
    activate a target/receptor.

    If you only need to simulate a single particle, use
    :class:`~er.simulation.MFPTSimulation` instead, since it is optimized
    for single particle analysis.
    """

    def __init__(self, network: NetworkModel, walker: Walker, num_walkers: int,
                 num_sims=1000):
        self.network = network
        self.walker = walker
        self.num_walkers = num_walkers
        self.num_sims = num_sims

    def _run(self, id=None):
        if not id:
            id = unique_id()

        network = self.network
        walkers = [copy(self.walker) for _ in range(self.num_walkers)]

        fpt = np.full(network.size(), np.inf)
        fpt[self.walker.node] = self.walker.time

        times = np.array([w.time for w in walkers], dtype=float)

        steps = 0
        while times.min() < fpt.max():
            for i, walker in enumerate(walkers):
                node, times[i] = walker.step(network)
                fpt[node] = min(fpt[node], times[i])

            steps += 1

            if steps % 5000 == 0 and network.memory:
                network.clear_memory(times.min())

        data = pd.DataFrame({"id":   id,
                             "node": range(len(fpt)),
                             "EFPT": fpt})

        return data
