#!/usr/bin/env python3

import networkx as nx

from louvain import Louvain
from collections import defaultdict

def makeSampleGraph():
    g = nx.Graph()
    g.add_edge("a", "b", weight=1.)
    g.add_edge("a", "c", weight=1.)
    g.add_edge("a", "d", weight=1.)
    g.add_edge("b", "d", weight=1.)
    g.add_edge("c", "d", weight=1.)

    g.add_edge("b", "g", weight=1.)

    g.add_edge("e", "f", weight=1.)
    g.add_edge("e", "g", weight=1.)
    g.add_edge("e", "h", weight=1.)
    g.add_edge("f", "h", weight=1.)
    g.add_edge("g", "h", weight=1.)

    return g

if __name__ == "__main__":
    sample_graph = makeSampleGraph()
    louvain = Louvain()
    partition = louvain.getBestPartition(sample_graph)

    p = defaultdict(list)
    for node, com_id in partition.items():
        p[com_id].append(node)

    for com, nodes in p.items():
        print(com, nodes)
