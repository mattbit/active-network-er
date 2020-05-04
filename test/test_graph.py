import random
import unittest

import networkx as nx
import numpy as np

from er.graph import Trajectory, generator
from er.graph.utils import k_cores


class TestGraphGenerator(unittest.TestCase):

    def test_random_regular(self):
        g = generator.random_regular(1032)
        self.assertEqual(g.number_of_nodes(), 1032)

        degrees = np.array([d for _, d in g.degree()])
        self.assertTrue(np.all(degrees == 3))

    def test_hexagonal_lattice(self):
        g = generator.hexagonal_lattice(1000)
        self.assertBetween(g.number_of_nodes(), (900, 1100))

        mean_degree = np.mean([d for n, d in g.degree()])
        self.assertAlmostEqual(mean_degree, 3, places=0)

    def test_hexagonal_periodic_lattice(self):
        g = generator.hexagonal_lattice(1000, periodic=True)
        self.assertBetween(g.number_of_nodes(), (900, 1100))

        mean_degree = np.mean([d for n, d in g.degree()])
        self.assertEqual(mean_degree, 3.00)

    def test_loadmat(self):
        g = generator.from_matfile("test/data/graph.mat")
        self.assertEqual(g.number_of_nodes(), 1920)
        self.assertEqual(g.number_of_edges(), 2787)

    def assertBetween(self, num, range):
        self.assertGreaterEqual(num, range[0])
        self.assertLessEqual(num, range[1])


class TestGraphTrajectory(unittest.TestCase):

    def test_append(self):
        t = Trajectory([1, 2])
        t.append(3)
        t.append(4)
        self.assertSequenceEqual(t.nodes, [1, 2, 3, 4])
        self.assertSequenceEqual(t.times, [0, 1, 2, 3])

        t = Trajectory([1, 2], [0.1, 0.2])
        t.append(3, 0.3)
        self.assertSequenceEqual(t.nodes, [1, 2, 3])
        self.assertSequenceEqual(t.times, [.1, .2, .3])

    def test_init(self):
        # ID
        id = random.randint(0, 1000)
        t = Trajectory([0], id=id)
        self.assertEqual(t.id, id)

        # Invalid
        self.assertRaises(ValueError, Trajectory, [1, 2], [1, 2, 3])

    def test_times(self):
        t = Trajectory([10, 21, 34])
        self.assertSequenceEqual(t.times, [0, 1, 2])

        t = Trajectory([10, 21, 34], [0.1, 0.2, 0.3])
        self.assertSequenceEqual(t.times, [0.1, 0.2, 0.3])

    def test_edges(self):
        t = Trajectory([1, 2, 3, 3, 4])
        self.assertSequenceEqual(t.edges(), [(1, 2), (2, 3), (3, 3), (3, 4)])

    def test_traps(self):
        t = Trajectory([1, 2, 2, 3, 4, 4, 4, 5])
        self.assertSequenceEqual(t.traps(), [2, 4, 4])

    def test_endpoints(self):
        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0.5, 0.9])
        self.assertEqual(t.start_node(), 0)
        self.assertEqual(t.end_node(), 3)

        # Auto sorting.
        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0., 0.1])
        self.assertEqual(t.start_node(), 2)

        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0., 0.1])
        self.assertEqual(t.end_node(), 1)

    def test_duration(self):
        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0.5, 0.9])
        self.assertEqual(t.duration(), 0.9 - 0.2)

        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0., 0.1])
        self.assertEqual(t.duration(), 0.3)

    def test_time(self):
        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0.5, 0.9])
        self.assertEqual(t.time(), 0.9)

        t = Trajectory([0, 1, 2, 3], [0.2, 0.3, 0., 0.1])
        self.assertEqual(t.time(), 0.3)

    def test_length(self):
        t = Trajectory([1])
        self.assertEqual(len(t), 0)

        t = Trajectory([0, 1, 2, 3])
        self.assertEqual(len(t), 3)

    def test_fpt(self):
        t = Trajectory([1, 2, 3, 2, 3, 3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(t.fpt(3), 0.3)

        t = Trajectory([1, 2, 2, 3, 3, 3], [0.1, 0.4, 0.2, 0.6, 0.5, 0.3])
        self.assertEqual(t.fpt(3), 0.3)
        self.assertTrue(np.isnan(t.fpt(1000)))

    def test_contains(self):
        t = Trajectory([1, 2, 3, 2, 3, 3], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        self.assertTrue(1 in t)
        self.assertTrue(3 in t)
        self.assertFalse(0.2 in t)
        self.assertFalse(5 in t)


class TestGraphUtils(unittest.TestCase):

    def test_kcores(self):
        # Check that results correspond to those of networkx.
        G = nx.gnm_random_graph(100, 300)
        core_graph = nx.algorithms.k_core(G, 4)
        result, _ = k_cores(G, 4)

        self.assertEqual(result.edges, core_graph.edges)
