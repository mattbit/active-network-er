import unittest
from unittest.mock import patch

from er.model.walker import Walker, RandomWalker, ExponentialWalker


class TestWalker(unittest.TestCase):

    def test_walker_init(self):
        w = Walker()
        self.assertEqual(w.time, 0)
        self.assertEqual(w.node, 0)

        w = Walker(start=10)
        self.assertEqual(w.time, 0)
        self.assertEqual(w.node, 10)

    def test_random_walker_params(self):
        w = RandomWalker()
        self.assertEqual(w.params['timescale'], 1)

        w = RandomWalker(timescale=2.5)
        self.assertEqual(w.params['timescale'], 2.5)

    @patch("er.model.network.NetworkModel")
    def test_random_walker(self, network):
        w = RandomWalker(timescale=1)
        network.neighbors.return_value = []
        w.step(network)
        self.assertEqual(w.time, 1)
        self.assertEqual(w.node, 0)
        network.neighbors.assert_called_with(0, 0)

        network.neighbors.return_value = [1, 2]
        w.step(network)
        self.assertEqual(w.time, 2)
        self.assertIn(w.node, [1, 2])
        network.neighbors.assert_called_with(0, 1)
        prev_node = w.node

        network.neighbors.return_value = [3]
        w.step(network)
        self.assertEqual(w.time, 3)
        self.assertEqual(w.node, 3)
        network.neighbors.assert_called_with(prev_node, 2)

    @patch("er.model.network.NetworkModel")
    def test_random_walker_timescale(self, network):
        w = RandomWalker(timescale=0.5)
        network.neighbors.return_value = []
        w.step(network)
        self.assertEqual(w.time, 0.5)
        w.step(network)
        w.step(network)
        self.assertEqual(w.time, 1.5)

    def test_exponential_walker_params(self):
        w = ExponentialWalker()
        self.assertEqual(w.params['timescale'], 1)

        w = ExponentialWalker(timescale=2.5)
        self.assertEqual(w.params['timescale'], 2.5)

    @patch("er.model.network.NetworkModel")
    def test_exponential_walker(self, network):
        w = ExponentialWalker(timescale=1000)

        network.neighbors.return_value = []
        w.step(network)
        self.assertGreater(w.time, 1)
        self.assertEqual(w.node, 0)

        network.neighbors.assert_called_with(0, 0)
        prev_time = w.time

        network.neighbors.return_value = [1, 2]
        w.step(network)
        self.assertGreater(w.time, prev_time)
        self.assertIn(w.node, [1, 2])
        network.neighbors.assert_called_with(0, prev_time)
        prev_time = w.time
        prev_node = w.node

        network.neighbors.return_value = [3]
        w.step(network)
        self.assertGreater(w.time, prev_time)
        self.assertEqual(w.node, 3)
        network.neighbors.assert_called_with(prev_node, prev_time)

    @patch("er.model.network.NetworkModel")
    def test_exponential_walker_timescale(self, network):
        w = ExponentialWalker(timescale=0.1)
        network.neighbors.return_value = [2]

        for _ in range(10000):
            w.step(network)

        self.assertAlmostEqual(0.1, w.time / 10000, places=1)
