import numpy as np
import pandas as pd
from copy import copy
from tqdm import tqdm

from er.graph import Trajectory
from er.model.walker import Walker
from er.model.network import NetworkModel


class TrajectoryGenerator:
    """Trajectory generator.

    TrajectoryGenerator simulates an arbitrary number of walkers on a network
    and returns the trajectories of each walker.
    """

    def __init__(self, network: NetworkModel, walker: Walker):
        """Initialize the TrajectoryGenerator.

        Parameters
        ----------
        network : NetworkModel
            The network model.
        walker : Walker
            The walker model.
        """
        self.network = network
        self.walker = walker

    def trajectories(self, num_walkers, max_time, start_nodes=None,
                     progress=True):
        """Simulates the walkers motion for a limited time.

        Parameters
        ----------
        num_walkers : int
            The number of walkers to simulate.
        max_time : float
            The time at which the simulation will be interrupted.
        progress : bool
            Whether to show the progress bar during the simulation.

        Returns
        -------
        trajectories : pd.DataFrame
            A :class:`pandas.DataFrame` containing the trajectories.
        """
        network = self.network
        walkers = [copy(self.walker) for _ in range(num_walkers)]
        if start_nodes is not None:
            if len(start_nodes) != len(walkers):
                raise Exception(
                    "Number of starting nodes must be equal to walkers")
            for i, node in enumerate(start_nodes):
                walkers[i].start = node
                walkers[i].node = node

        times = np.array([w.time for w in walkers], dtype=float)

        trajs = [Trajectory([w.node], [w.time], id=k)
                 for k, w in enumerate(walkers)]
        sim_time = 0
        steps = 0

        if progress:
            bar = tqdm(total=max_time)

        while sim_time < max_time:
            for id, walker in enumerate(walkers):
                if walker.time >= max_time:
                    continue

                node, times[id] = walker.step(network)
                trajs[id].append(node, times[id])

            steps += 1
            sim_time = times.min()

            if progress:
                bar.n = round(sim_time, 2)
                bar.update(0)

            # Clear memory
            if steps % 10000 == 0 and getattr(network, 'memory', False):
                network.clear_memory(sim_time)

        return pd.concat([traj.to_dataframe() for traj in trajs])

    def trajectories_to_target(self, num_walkers, target, keep=None,
                               progress=True):
        """Simulates the walkers motion until they hit a given target.

        Parameters
        ----------
        num_walkers : int
            The number of walkers to simulate.
        target :
            The target node label.
        keep : int
            The number of trajectories to keep (sorted by arrival time).
            For example, if `keep` is 10 only the trajectories of the first
            10 walkers to arrive at the target will be returned. If set to
            `None`, all the trajectories will be returned.

        Returns
        -------
        data : pandas.Dataframe
            A :class:`pandas.DataFrame` containing the trajectories.
        """
        if keep is None:
            keep = num_walkers

        network = self.network
        walkers = np.array([copy(self.walker) for _ in range(num_walkers)])

        active_idx = set(range(num_walkers))
        times = np.array([w.time for w in walkers], dtype=float)

        paths = np.array([Trajectory([w.node], times=[w.time], id=i)
                          for i, w in enumerate(walkers)])

        sim_time = 0.
        arrival_times = np.full_like(walkers, fill_value=np.inf)
        steps = 0

        while True:
            completed = set()
            for id in active_idx:
                walker = walkers[id]

                node, times[id] = walker.step(network)
                paths[id].append(node, times[id])

                if node == target:
                    completed.add(id)
                    arrival_times[id] = times[id]

            active_idx.difference_update(completed)

            if active_idx:
                sim_time = times[list(active_idx)].min()
            else:
                # All walkers arrived
                break

            if np.sum(arrival_times < sim_time) >= keep:
                # First N walkers have arrived
                break

            # Clear memory
            if steps % 10000 == 0 and hasattr(network, 'clear_memory'):
                network.clear_memory(sim_time)

            steps += 1

        # Simulation ended, now retrieve the trajectories.
        idx = np.argsort(arrival_times)[:keep]
        data = pd.concat([path.to_dataframe() for path in paths[idx]])

        return data
