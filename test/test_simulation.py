import logging
import unittest
from unittest.mock import call, patch

import numpy as np
from pandas import DataFrame

from er.model.walker import Walker
from er.simulation import (ConcurrentSimulation, MEFPTSimulation,
                           MFPTSimulation, TrajectoryGenerator)


class WalkerStub(Walker):
    def __init__(self, *trajs, id=0):
        self.id = id
        self.trajs = trajs
        self._step = 0
        self.time = 0.
        self.node = 0

    def step(self, network):
        self.node, self.time = self.trajs[self.id][self._step]
        self._step += 1

        return self.node, self.time

    def __copy__(self):
        m = self.__class__(*self.trajs, id=self.id)
        self.id += 1

        return m


class TestFPT(unittest.TestCase):

    @patch("er.model.network.NetworkModel")
    @patch("er.model.walker.Walker")
    def test_mfpt_run(self, network, walker):
        network.size.return_value = 3
        walker.step.side_effect = [(0, 0.1), (1, 0.2), (1, 0.3), (2, 0.4)]

        sim = MFPTSimulation(network, walker)
        data = sim._run(11)

        self.assertIsInstance(data, DataFrame)
        self.assertEqual(data.id[0], 11)
        self.assertSequenceEqual(list(data.node), [0, 1, 2])
        self.assertSequenceEqual(list(data.FPT), [0.1, 0.2, 0.4])

    @patch("er.model.network.NetworkModel")
    def test_mefpt_run(self, network):
        network.size.return_value = 3

        t1 = [(2, 0.1), (1, 0.2), (0, 0.3), (2, 0.4)]
        t2 = [(2, 0.15), (1, 0.18), (1, 0.3), (2, 0.5)]
        walker = WalkerStub(t1, t2)

        sim = MEFPTSimulation(network, walker, 2, 1)
        data = sim._run(11)

        self.assertIsInstance(data, DataFrame)
        self.assertEqual(data.id[0], 11)
        self.assertSequenceEqual(list(data.node), [0, 1, 2])
        self.assertSequenceEqual(list(data.EFPT), [0.0, 0.18, 0.1])


class TestTrajectoryGenerator(unittest.TestCase):

    @patch("er.model.network.NetworkModel")
    def test_trajectories(self, network):
        t1 = [(2, 3.1), (1, 6.2), (1, 6.5), (0, 9.3), (2, 12.4)]
        t2 = [(0, 2.2), (1, 3.3), (0, 9.9), (2, 11.1)]
        walker = WalkerStub(t1, t2)

        gen = TrajectoryGenerator(network, walker)
        trajs = gen.trajectories(2, 10)

        self.assertListEqual(
            list(trajs[trajs.id == 0].node.values), [0, 2, 1, 1, 0, 2])
        self.assertListEqual(
            list(trajs[trajs.id == 0].time.values), [0.0, 3.1, 6.2, 6.5, 9.3, 12.4])
        self.assertSequenceEqual(
            list(trajs[trajs.id == 1].node.values), [0, 0, 1, 0, 2])
        self.assertSequenceEqual(
            list(trajs[trajs.id == 1].time.values), [0.0, 2.2, 3.3, 9.9, 11.1])

    @patch("er.model.network.NetworkModel")
    def test_clear_memory(self, network):
        walker = WalkerStub([(t, t / 100 + 0.01) for t in range(25000)])

        self.assertTrue(hasattr(network, "clear_memory"))
        gen = TrajectoryGenerator(network, walker)
        gen.trajectories(1, 200)
        calls = network.clear_memory.mock_calls
        self.assertEqual(calls, [call(100), call(200)])

    @patch("er.model.network.NetworkModel")
    def test_trajs_to_target(self, network):
        t1 = [(2, 3.1), (1, 6.2), (2, 6.5), (0, 9.3), (2, 12.4)]
        t2 = [(0, 2.2), (1, 3.3), (0, 9.9), (2, 11.1), (1, 12.)]
        walker = WalkerStub(t1, t2)

        gen = TrajectoryGenerator(network, walker)
        trajs = gen.trajectories_to_target(2, 2)

        self.assertListEqual(
            list(trajs[trajs.id == 0].node.values), [0, 2])
        self.assertListEqual(
            list(trajs[trajs.id == 0].time.values), [0.0, 3.1])
        self.assertSequenceEqual(
            list(trajs[trajs.id == 1].node.values), [0, 0, 1, 0, 2])
        self.assertSequenceEqual(
            list(trajs[trajs.id == 1].time.values), [0.0, 2.2, 3.3, 9.9, 11.1])

        self.assertEqual(2, trajs.id.nunique())

    @patch("er.model.network.NetworkModel")
    def test_trajs_to_target_keep_1(self, network):
        t1 = [(0, 2.2), (1, 3.3), (0, 9.9), (2, 11.1), (1, 12.)]
        t2 = [(2, 3.1), (1, 6.2), (2, 6.5), (0, 9.3), (2, 12.4)]
        t3 = [(1, 1.2), (1, 2.2), (2, 2.5), (0, 9.3), (2, 12.4)]
        walker = WalkerStub(t1, t2, t3)

        gen = TrajectoryGenerator(network, walker)
        trajs = gen.trajectories_to_target(3, 2, keep=1)
        print(trajs)
        self.assertListEqual(list(trajs.node.values), [0, 1, 1, 2])
        self.assertListEqual(list(trajs.time.values), [0.0, 1.2, 2.2, 2.5])

        self.assertEqual(1, trajs.id.nunique())
