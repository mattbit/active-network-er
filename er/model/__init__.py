"""
The model of motion is organized into two distinct (but interacting) parts: a
model for the network structure and a model for the moving agent (walker). The
network model (:class:`~er.model.network.NetworkModel`) is used to represent
the structure on which the agents move (the underlining graph, edges direction,
time evolution of the network). Instead, the agent model
(:class:`~er.model.walker.Walker`) implements the logic of the motion (e.g. how
is the next move chosen, how much time does it take to cross an edge).
Different implementations and combination of these entities make it possible to
flexibly represent different models of motion.
"""
from .network import (NetworkModel, UndirectedNetwork,
                      SwitchingNetwork, SwitchingNetworkConstantRate)
from .walker import Walker, RandomWalker, ExponentialWalker

__all__ = ['NetworkModel', 'UndirectedNetwork', 'SwitchingNetwork',
           'SwitchingNetworkConstantRate', 'Walker', 'RandomWalker',
           'ExponentialWalker']
