"""
General utils.
"""

import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

from config import DATA_PATH


def unique_id():
    """Returns a unique integer identifier."""
    return uuid.uuid4().int & (1 << 64) - 1


def dict_zip(a, b):
    ab = {}
    keys = set(a.keys()).intersection(b.keys())
    for key in keys:
        ab[key] = (a[key], b[key])

    return ab


def load_graph(name):
    return nx.read_graphml("resources/graphs/{}.graphml".format(name),
                           node_type=int)


def load_data(name, suffix=".csv"):
    path = data_path(name, suffix)

    if suffix.lower() in ['.parquet', '.parq']:
        return pd.read_parquet(str(path))

    return pd.read_csv(str(path))


def data_path(name, suffix=".csv"):
    return DATA_PATH.joinpath(name + suffix)


def reduce_trajs_steps(dataframe):
    df = dataframe.copy()
    runs = (df.node.diff() != 0).cumsum()
    df["count"] = 1
    agg = {col: "first" for col in df.columns}
    agg.update({"time": "min", "count": "count"})

    return df.groupby(runs, as_index=False).agg(agg)


def vals_by_dist(graph, values, source=0):
    """Returns the MFPT over the nodes at the same distance from the source."""
    df = pd.DataFrame({"value": values})

    for node, distance in nx.shortest_path_length(graph, source).items():
        df.loc[node, "distance"] = distance

    return df


def get_nodes_pos(graph, as_array=False):
    """Returns the position of the nodes in the graph.

    Args:
        graph: The instance of networkx.Graph.
        as_array: If False (default), a dictionary will be returned; if True,
            a numpy.array.
    """
    xs = nx.get_node_attributes(graph, "x")
    ys = nx.get_node_attributes(graph, "y")

    pos = dict_zip(xs, ys)

    if as_array:
        return np.array([pos[i] for i in range(len(pos))])

    return pos


def count_by_source(node, times, df, graph, window=None, progress=True):
    if window is None:
        window = df.time.diff().max() + 1

    neighbors = list(graph.neighbors(node))
    count = np.zeros((3, len(times)))

    for j, time in enumerate(tqdm(times, disable=(not progress))):
        min_time = time - window
        dx = df[(df.time <= time) & (df.time >= min_time)].groupby("id").last()
        c = dx[dx.node == node].groupby("prev_node").step.count()

        for i, neigh in enumerate(neighbors):
            count[i][j] = c[neigh] if neigh in c else 0

    return neighbors, count


def count_nodes(time, df):
    dx = df[df.time <= time].groupby("id", as_index=False).last()

    return dx.groupby("node").id.count()


def collapse_traps(df):
    """Collapse trapped state in dataframe.

    Multiple rows in the dataframe corresponding to failed attempts to escape
    from a trap node are collapsed into a single record with the arrival time
    in the node and the count of escape attempts (traps).
    """
    runs = ((df.node.diff() != 0) | (df.id.diff() != 0)).cumsum()

    df["traps"] = 1
    agg = {col: "first" for col in df.columns}
    agg.update({"time": "min", "traps": "count"})
    df = df.groupby(runs, as_index=False).agg(agg)
    df["traps"] -= 1

    return df
