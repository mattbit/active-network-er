import copy
import random
import unittest
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

from er.model.network import (MemorylessSwitch, NetworkModel, Switch,
                              SwitchingNetwork, SwitchingNetworkConstantRate,
                              UndirectedNetwork)


class TestNetworkModel(unittest.TestCase):

    @patch("networkx.Graph")
    def test_interface(self, graph):
        graph.number_of_nodes.return_value = 103
        graph.copy.return_value = graph

        n = NetworkModel(graph)

        self.assertEqual(n.size(), 103)
        self.assertRaises(NotImplementedError, n.neighbors, 0)


class TestUndirectedNetwork(unittest.TestCase):
    def test_neighbors(self):
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2), (2, 1)])

        n = UndirectedNetwork(graph)
        self.assertCountEqual(n.neighbors(0), [1, 2])
        self.assertCountEqual(n.neighbors(0, 100), [1, 2])
        self.assertCountEqual(n.neighbors(1), [0, 2])


class TestSwitchingNetwork(unittest.TestCase):

    def test_initialization(self):
        g = nx.Graph([(0, 1), (1, 2), (0, 2)])
        self.assertRaises(ValueError, SwitchingNetwork, g)  # missing timescale

        n = SwitchingNetwork(g, timescale=0.1)
        for source, target, data in n.graph.edges(data=True):
            self.assertIsInstance(data["switch"], Switch)
            self.assertEqual(0.1, data["switch"].timescale)

        n = SwitchingNetwork(g, timescale=0.1, memory=False)
        for source, target, data in n.graph.edges(data=True):
            self.assertIsInstance(data["switch"], MemorylessSwitch)
            self.assertEqual(0.1, data["switch"].timescale)

    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    def test_switching(self, switch01, switch02, switch03):
        g = nx.Graph([(0, 1), (0, 2), (0, 3)])
        n = SwitchingNetwork(g, timescale=0.5)

        n.graph.edges[(0, 1)]["switch"] = switch01
        n.graph.edges[(0, 2)]["switch"] = switch02
        n.graph.edges[(0, 3)]["switch"] = switch03

        switch01.open.return_value = False
        switch02.open.return_value = True
        switch03.open.return_value = True

        neighbors = n.neighbors(0, 0.131)
        self.assertCountEqual(neighbors, [2, 3])
        switch01.open.assert_called_with(0, 0.131)
        switch02.open.assert_called_with(0, 0.131)
        switch03.open.assert_called_with(0, 0.131)

        neighbors = n.neighbors(1, 0.132)
        switch01.open.assert_called_with(1, 0.132)

    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    def test_reset(self, switch1, switch2):
        g = nx.Graph([(0, 1), (0, 2)])
        n = SwitchingNetwork(g, timescale=0.5)
        n.graph.edges[(0, 1)]["switch"] = switch1
        n.graph.edges[(0, 2)]["switch"] = switch2

        n.reset()

        switch1.reset.assert_called_once_with()
        switch2.reset.assert_called_once_with()

    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    def test_clear_memory(self, switch1, switch2):
        g = nx.Graph([(0, 1), (0, 2)])
        n = SwitchingNetwork(g, timescale=0.5)
        n.graph.edges[(0, 1)]["switch"] = switch1
        n.graph.edges[(0, 2)]["switch"] = switch2

        n.clear_memory(4.2)

        switch1.clear.assert_called_once_with(4.2)
        switch2.clear.assert_called_once_with(4.2)

    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    def test_skip_clear_memory(self, switch1, switch2):
        g = nx.Graph([(0, 1), (0, 2)])
        n = SwitchingNetwork(g, timescale=0.5, memory=False)
        n.graph.edges[(0, 1)]["switch"] = switch1
        n.graph.edges[(0, 2)]["switch"] = switch2

        n.clear_memory(4.2)

        switch1.clear.assert_not_called()
        switch2.clear.assert_not_called()


class TestMemorylessSwitch(unittest.TestCase):

    def test_status(self):
        init_status = np.empty(1000)

        for i in range(1000):
            s = MemorylessSwitch(10, 0.5)
            init_status[i] = s.status

            self.assertEqual(s.open(10, 0.), s.status)
            self.assertEqual(s.open(11, 0.), not s.status)

        # Initial status is random uniform
        self.assertAlmostEqual(init_status.mean(), 0.5, places=1)

    def test_switching(self):
        s = MemorylessSwitch(10, 0.5)
        status = np.empty(1000)

        for i in range(1000):
            status[i] = s.open(11, i)

        self.assertAlmostEqual(status.mean(), 0.5, places=1)

    def test_memoryless(self):
        s = MemorylessSwitch(10, 0.5)
        status = s.open(10, 123.1)

        self.assertEqual(s.open(10, 123.1), status)
        self.assertRaises(Exception, s.open, 10, 111.)

    def test_reset(self):
        s = MemorylessSwitch(10, 0.5)
        self.assertEqual(s.time, 0.)

        s.open(10, 11.)
        self.assertEqual(s.time, 11.)

        s.reset()
        self.assertEqual(s.time, 0.)


class TestSwitch(unittest.TestCase):

    def test_status(self):
        init_status = np.empty(1000)

        for i in range(1000):
            s = Switch(10, 0.5)
            init_status[i] = s.open(10, 0.)
            self.assertEqual(s.open(11, 0.), not s.open(10, 0.))

        # Initial status is uniform
        self.assertAlmostEqual(init_status.mean(), 0.5, places=1)

    def test_switching(self):
        s = Switch(10, 0.5, batch=10)
        status = s.open(10, 0.1)
        self.assertEqual(11, len(s.switch_times))

        s.open(10, 100.1)
        self.assertGreater(len(s.switch_times), 11)
        self.assertEqual(1, len(s.switch_times) % 10)

        self.assertEqual(status, s.open(10, 0.1))
        self.assertRaises(Exception, s.open, 10, -0.1)

    def test_clear(self):
        for _ in range(10):
            s = Switch(10, 0.5, batch=10)
            status14s = s.open(10, 14.)
            status29s = s.open(10, 29.)

            s.clear(10.)
            self.assertEqual(status14s, s.open(10, 14.))
            self.assertEqual(status29s, s.open(10, 29.))

            s.clear(14.)
            self.assertEqual(status14s, s.open(10, 14.))
            self.assertEqual(not status29s, s.open(11, 29.))

            s.clear(30.)
            self.assertRaises(Exception, s.open, 10, 29)

    def test_reset(self):
        s = Switch(10, 0.5, batch=1000)
        s.open(10, 123.1)
        self.assertEqual(1001, len(s.switch_times))

        s.reset()
        self.assertEqual(1, len(s.switch_times))


class TestSwitchingNetworkConstantRate(unittest.TestCase):
    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    @patch("er.model.network.Switch")
    def test_switching(self, switch01, switch02, switch03):
        g = nx.Graph([(0, 1), (0, 2), (0, 3)])
        n = SwitchingNetworkConstantRate(g, timescale=0.5)

        n.graph.edges[(0, 1)]["switch"] = switch01
        n.graph.edges[(0, 2)]["switch"] = switch02
        n.graph.edges[(0, 3)]["switch"] = switch03

        switch01.open.return_value = False
        switch02.open.return_value = True
        switch03.open.return_value = True

        neighbors = n.neighbors(0, 0.131)
        self.assertCountEqual(neighbors, [0, 2, 3])
        switch01.open.assert_called_with(0, 0.131)
        switch02.open.assert_called_with(0, 0.131)
        switch03.open.assert_called_with(0, 0.131)

        neighbors = n.neighbors(1, 0.132)
        switch01.open.assert_called_with(1, 0.132)
