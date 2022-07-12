"""
Network models represent the structure on which walkers move. Walkers interact
with the network models by means of the
:meth:`~er.model.network.NetworkModel.neighbors` method.
"""
import bisect
import random
from abc import abstractmethod

import networkx as nx
import numpy as np


class NetworkModel:
    """The base network model."""

    def __init__(self, graph: nx.Graph, **params):
        """Create a NetworkModel.

        Parameters
        ----------
        graph : networkx.Graph
            A Graph instance representing the network structure.
        """
        self.graph = graph.copy()
        self.params = params

    def size(self) -> int:
        """Returns the number of nodes in the network."""
        return self.graph.number_of_nodes()

    def reset(self) -> None:
        """Resets the network state."""
        return

    @property
    def meta(self) -> dict:
        return {
            "model": self.__class__.__name__,
            "params": self.params,
            "graph": self.graph.name,
        }

    @abstractmethod
    def neighbors(self, node, time=0):
        """Returns a list of accessible neighbors of `node`.

        Parameters
        ----------
        node :
            The node from which the neighbors are searched.
        time : float
            The current time, it is required if the network structure varies
            with time.
        """
        raise NotImplementedError()


class UndirectedNetwork(NetworkModel):
    """Undirected network model.

    UndirectedNetwork is a simple model for the motion on an undirected graph.
    Walkers can go through the edges in both directions, independently of time.
    """

    def __init__(self, graph: nx.Graph, **params):
        super().__init__(graph, **params)

        # We precompute the adjacency dict, for faster access.
        self.adj_dict = nx.to_dict_of_lists(graph)

    def neighbors(self, node, time=0):
        return self.adj_dict[node]


class SwitchingNetwork(NetworkModel):
    """Network model with switching edge direction.

    SwitchingNetwork implements a directed network where the direction of edges
    changes in time as a Poisson's process. Walkers moving on this network can
    only move through outward edges (if any).
    """

    def __init__(self, graph: nx.Graph, **params):
        """Create a SwitchingNetwork instance."""
        super().__init__(graph, **params)

        if "timescale" not in params:
            raise ValueError("The parameter `timescale` is needed!")

        self.timescale = params["timescale"]
        self.memory = params["memory"] if "memory" in params else True

        switch_class = Switch if self.memory else MemorylessSwitch

        for source, target, data in self.graph.edges(data=True):
            data["switch"] = switch_class(source, self.timescale)

    def reset(self):
        for _, _, data in self.graph.edges(data=True):
            data["switch"].reset()

    def clear_memory(self, min_time):
        """Clears the memory of the switching events before the given time."""
        if not self.memory:
            return

        for _, _, data in self.graph.edges(data=True):
            data["switch"].clear(min_time)

    def neighbors(self, node, time):
        return [
            n for n, data in self.graph[node].items() if data["switch"].open(node, time)
        ]

    def edges(self, time):
        for source, target, data in self.graph.edges(data=True):
            is_open = data["switch"].open(source, time)
            edge = (source, target) if is_open else (target, source)
            yield edge


class SwitchingNetworkConstantRate(SwitchingNetwork):
    """Network model with switching edge direction and constant flow.

    This class is a slightly modified version of
    :class:`~er.model.network.SwitchingNetwork`. In this implementation, the
    flow is not split across outward edges. In practical terms it means that
    for a node with :math:`N` total edges, of which :math:`N_{outward}` are
    outward directed, the particle will move to a neighbor with probability
    :math:`\\frac{N_{outward}}{N}` and remain in the current node with
    probability :math:`\\frac{1- N_{outward}}{N}`.
    """

    def neighbors(self, node, time):
        return [
            n if data["switch"].open(node, time) else node
            for n, data in self.graph[node].items()
        ]


class MemorylessSwitch:
    """A switch object that does not have memory of past events.

    Once its state is queried with `open`, it is not possible to retrieve the
    value of a previous state in time.
    """

    def __init__(self, source, timescale):
        self.source = source
        self.status = random.randint(0, 1)
        self.timescale = timescale
        self.time = 0

    def open(self, source, time):
        interval = time - self.time
        if interval < 0:
            raise Exception("Switch has no memory of previous time!")

        p_switch = (1 - np.exp(-2 * interval / self.timescale)) / 2

        if np.random.rand() <= p_switch:
            self.status = (self.status + 1) % 2

        self.time = time

        return not (source == self.source) ^ (self.status)  # xnor

    def reset(self):
        self.time = 0.0
        self.status = random.randint(0, 1)


class Switch:
    """A switch object that keeps memory of the past states."""

    def __init__(self, source, timescale, batch=1000):
        self.source = source
        self.init = 1 if random.random() < 0.5 else 0
        self.timescale = timescale
        self.batch = batch
        self.switch_times = np.array([0.0])

    def reset(self):
        self.init = 1 if random.random() < 0.5 else 0
        self.switch_times = np.array([0.0])

    def open(self, source, time):
        # Expand lifespan if required.
        while self.switch_times[-1] < time:
            self._expand_times()

        if time < self.switch_times[0]:
            raise Exception("No memory for time {}".format(time))

        return not (
            (source == self.source)
            ^ (1 + self.init + bisect.bisect(self.switch_times, time)) % 2
        )  # xnor

    def _expand_times(self):
        times = np.random.exponential(self.timescale, self.batch)
        np.cumsum(times, out=times)
        times += self.switch_times[-1]

        self.switch_times = np.concatenate((self.switch_times, times))

    def clear(self, time):
        """Clears the memory of events before given time.

        Args:
            time: The time limit.
        """
        if self.switch_times[-1] < time:
            interval = time - self.switch_times[-1]
            num_events = np.random.poisson(interval / self.timescale)
            self.init = (self.open(self.source, self.switch_times[-1]) + num_events) % 2
            self.switch_times = np.array([time])
        else:
            self.init = self.open(self.source, time)
            i = bisect.bisect(self.switch_times, time)
            self.switch_times = np.concatenate(([time], self.switch_times[i:]))
