"""
Walkers represent agents moving on the network. Different implementations
reflect the different logic used to choose the next move and the parameters of
motion (e.g. waiting time in the nodes, edge crossing time).
"""
import random
from abc import abstractmethod

from er.model.network import NetworkModel


class Walker:
    """Base graph walker."""
    _default_params = {}

    def __init__(self, start=0, **kwargs):
        """Walker.

        Arguments:
            start: Starting node label.
        """
        self.start = start
        self.node = start
        self.time = 0
        params = self._default_params.copy()
        params.update(kwargs)
        self.params = params

    def reset(self) -> None:
        self.node = self.start
        self.time = 0

    @property
    def meta(self) -> dict:
        return dict(model=self.__class__.__name__, params=self.params)

    @abstractmethod
    def step(self, network: NetworkModel):  # pragma: no cover
        """Move the walker of one step on the network."""
        raise NotImplementedError()


class RandomWalker(Walker):
    """RandomWalker is a standard random walker.

    The random walker steps to a random neighboring node.
    """
    _default_params = {'timescale': 1}

    def __init__(self, start=0, **params):
        """Initialize the RandomWalker.

        Arguments:
            start: Starting node.
            timescale: The (constant) time for each step.
        """
        super().__init__(start, **params)
        self._timescale = self.params['timescale']

    def step(self, network: NetworkModel):
        """Move the walker of one step on the network."""
        nodes = network.neighbors(self.node, self.time)
        if nodes:
            self.node = random.choice(nodes)

        self.time += self._timescale

        return self.node, self.time


class ExponentialWalker(Walker):
    """ExponentialWalker spends an exponential time in nodes.

    This walker steps to a random neighbor in (random) exponential time.
    """
    _default_params = {'timescale': 1}

    def __init__(self, start=0, **params):
        """Initialize an ExponentialWalker.

        Arguments:
            start: Starting node.
            timescale: The exponential timescale needed for the step.
        """
        super().__init__(start, **params)
        self._lambd = 1. / self.params['timescale']

    def step(self, network: NetworkModel):
        """Move the walker of one step on the network."""
        nodes = network.neighbors(self.node, self.time)
        if nodes:
            self.node = random.choice(nodes)

        self.time += random.expovariate(self._lambd)

        return self.node, self.time
