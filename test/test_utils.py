import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from er.utils import collapse_traps

class TestUtils(unittest.TestCase):

    def test_collapse(self):
        trajs = pd.DataFrame({
            "id":   [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "node": [0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 1],
            "time": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5]
        })

        collapsed = collapse_traps(trajs)
        t1 = collapsed[collapsed.id == 1]
        t2 = collapsed[collapsed.id == 2]

        self.assertListEqual(list(t1.node), [0, 1, 2, 0])
        self.assertListEqual(list(t1.time), [1, 3, 4, 6])
        self.assertListEqual(list(t1.traps), [1, 0, 1, 0])

        self.assertListEqual(list(t2.node), [0, 1])
        self.assertListEqual(list(t2.time), [1, 5])
        self.assertListEqual(list(t2.traps), [3, 0])
