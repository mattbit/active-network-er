from collections import deque

import numpy as np
from pandas import DataFrame

from er.utils import unique_id


class Trajectory:
    """Trajectory represents a path on a graph."""

    def __init__(self, nodes, times=None, id=None):
        """Creates a Trajectory instance.

        Args:
            nodes: A sequence of the nodes composing the path.
            times: Optional arrival time in each node.
        """
        if times is None:
            times = range(len(nodes))

        if len(nodes) != len(times):
            raise ValueError("Nodes and times lengths do not match!")

        self.nodes = deque(nodes)
        self.times = deque(times)
        self.id = id if id is not None else unique_id()

        self._sorted = False

    def append(self, node, time=None):
        """Append a new node to the trajectory."""
        if time is None:
            time = self.times[-1] + 1.

        self.nodes.append(node)
        self.times.append(time)

        self._sorted = False  # Invalidate sorting

    def start_node(self):
        """Returns the starting node of the trajectory."""
        if not self._sorted:
            self._sort()

        return self.nodes[0]

    def end_node(self):
        """Returns the ending node of the trajectory."""
        if not self._sorted:
            self._sort()

        return self.nodes[-1]

    def duration(self):
        """Returns the trajectory time duration."""
        return max(self.times) - min(self.times)

    def edges(self):
        """Returns a list of edges on which the particle moved.

        The list may contain self loops (e.g. 3 → 3) if the particle was
        trapped in a node.
        """
        if not self._sorted:
            self._sort()

        n = list(self.nodes)  # deque does not support slicing

        return list(zip(n[:-1], n[1:]))

    def traps(self):
        """Returns a list of nodes in which the particle got trapped.

        The list may contain multiple occurrences of the same node, meaning
        that the trapping happened multiple times.
        """
        return [s for s, t in self.edges() if s == t]

    def time(self):
        """Returns the current trajectory time."""
        return max(self.times)

    def __len__(self):
        """Trajectory length as number of steps.

        Note that multiple steps spent in a trapped node (failed escape) are
        taken into account, thus this does not correspond to the effective
        distance travelled.
        """
        return len(self.nodes) - 1

    def __str__(self):
        """Returns a string representation of the trajectory."""
        if not self._sorted:
            self._sort()

        path = "{}".format(self.nodes[0])

        for i in range(1, len(self.nodes)):
            path += " → {}".format(self.nodes[i])

        return path

    def __iter__(self):
        """Returns an iterable of (node, time) tuples."""
        if not self._sorted:
            self._sort()

        return zip(self.nodes, self.times)

    def __contains__(self, node):
        """Whether the trajectory passes through a given node."""
        return node in self.nodes

    def data(self):
        """Provides a dict containing the main characteristics."""
        return {
            "duration": self.duration(),
            "start_node": self.start_node(),
            "end_node": self.end_node(),
            "length": len(self),
            "traps": len(self.traps())
        }

    def to_dataframe(self):
        """Represents the trajectory as DataFrame."""
        if not self._sorted:
            self._sort()

        df = DataFrame({"id": self.id, "node": self.nodes, "time": self.times})

        return df

    def _sort(self):
        decorated = list(zip(self.times, self.nodes))
        decorated.sort()
        self.times, self.nodes = [deque(l) for l in zip(*decorated)]

        self._sorted = True

    def fpt(self, target):
        """Returns the first passage time through `target`."""
        if not self._sorted:
            self._sort()

        try:
            i = self.nodes.index(target)
        except ValueError:
            return np.nan

        return self.times[i]
