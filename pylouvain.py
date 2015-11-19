#!/usr/bin/env python3
# coding: utf-8

import sys
import copy

from itertools import combinations
from itertools import permutations
from collections import defaultdict

class Louvain(object):

    @classmethod
    def show(cls, node2com):
        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)

        for com_id, nodes in com2node.items():
            print("com_id: {}, nodes: {}".format(com_id, ", ".join([str(node) for node in nodes])), file=sys.stderr)

    @classmethod
    def run(cls, graph):
        node2com, edge_weights = cls._setNode2Com(graph)

        node2com = cls._runInitFirstPhase(node2com, edge_weights, graph)
        node2coms = [node2com]
        cls.show(node2com)

        new_node2com, new_edge_weights = cls._runSecondPhase(node2com, edge_weights)
        node2coms.append(new_node2com)
        cls.show(new_node2com)

        while True:
            new_node2com = cls._runFirstPhase(new_node2com, new_edge_weights)
            node2coms.append(copy.deepcopy(new_node2com))
            _new_node2com, _new_edge_weights = cls._runSecondPhase(new_node2com, new_edge_weights)
            sys.exit()

    @classmethod
    def _runInitFirstPhase(cls, node2com, edge_weights, graph):
        node2com, edge_weights = cls._setNode2Com(graph)
        all_edge_weights = graph.size(weight="weight")

        status = True
        while status:
            print(sorted(graph.nodes()))
            statuses = []
            for node in sorted(graph.nodes()):
                statuses = []
                com_id = node2com[node]
                # neigh_clusters = list(set([node2com[neigh_node] for neigh_node in graph[node].keys()]))
                neigh_nodes = list(graph[node].keys())

                max_delta = 0.
                max_com_id = com_id
                max_n = None
                for neigh_node in sorted(neigh_nodes):
                    node2com_copy = copy.deepcopy(node2com)
                    node2com_copy[node] = node2com_copy[neigh_node]

                    # 隣接クラスタに移動した際のModularityの差分
                    # delta_q = (1. / (2 * all_edge_weights)) * (cls.getNodeWeightInCluster(node, node2com_copy, edge_weights, graph) -
                    #                                    ((cls.getTotWeight(com_id, node2com_copy, edge_weights) * cls.getNodeWeights(node, graph)) / (2 * all_edge_weights)))
                    delta_q = 2 * cls.getNodeWeightInCluster(node, node2com_copy, edge_weights, graph) - cls.getTotWeight(node, node2com_copy, edge_weights) * cls.getNodeWeights(node, graph) / all_edge_weights
                    print("neighbor_id: {}, k_i_in: {}, tot: {}, k_i: {}, max_delta: {}".format(neigh_node,
                                                                               cls.getNodeWeightInCluster(node, node2com_copy, edge_weights, graph),
                                                                               cls.getTotWeight(node, node2com_copy, edge_weights),
                                                                               cls.getNodeWeights(node, graph),
                                                                               delta_q))

                    if delta_q > max_delta:
                        max_delta = delta_q
                        max_com_id = node2com_copy[neigh_node]
                        max_n = copy.deepcopy(node2com_copy)

                print("k_i_in: {}, tot: {}, k_i: {}, max_delta: {}".format(cls.getNodeWeightInCluster(node, node2com_copy, edge_weights, graph),
                                                            cls.getTotWeight(node, max_n, edge_weights),
                                                            cls.getNodeWeights(node, graph),
                                                            max_delta))
                print(node2com, max_com_id)
                node2com[node] = max_com_id
                print(com_id, max_com_id)
                statuses.append(com_id != max_com_id)

            if sum(statuses) == 0:
                break

        return node2com

    @classmethod
    def _runFirstPhase(cls, node2com, all_edge_weights):
        pass

    @classmethod
    def _runSecondPhase(cls, node2com, edge_weights):
        # クラスタ内のノードの一括集約
        #     1. cluster内のノードを一括集約. cluster内のエッジの重みの総和を2乗して，自身へのエッジとする 
        #     2. cluster間ノードはcluster内のノードが別のクラスタへ接続していたノードへのエッジの総和とする
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda : defaultdict(float))

        # ノードの集約
        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        # cluster内エッジの重み
        # for com_id in com2node.keys():
        #     cluster_weight = cls.getWeightsInCluster(com_id, node2com, edge_weights)
        #     new_edge_weights[com_id][com_id] = cluster_weight ** 2

        # cluster間エッジの重み
        nodes = list(node2com.keys())
        for edge in permutations(nodes, 2):
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][edge[1]]

        for node in new_node2com.keys():
            new_edge_weights[node][node] = (new_edge_weights[node][node] / 2) ** 2

        return new_node2com, new_edge_weights

    @classmethod
    def getTotWeight(cls, node, node2com, edge_weights):
        # 街灯ノードを除いたクラスタの隣接エッジの総和
        nodes = []
        for n, com_id in node2com.items():
            if com_id == node2com[node] and node != n:
                nodes.append(n)

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    @classmethod
    def getWeightsInCluster(cls, com_id, node2com, edge_weights):
        nodes = [node for node, com in node2com.items() if com == com_id]
        edges = combinations(nodes, 2)
        cluster_weight = sum([edge_weights[edge[0]][edge[1]] for edge in edges])

        return cluster_weight, nodes

    @classmethod
    def getNodeWeightInCluster(cls, node, node2com, edge_weights, graph):
        # nodeに接続しているcluster内のedgeの重みの総和
        neigh_nodes = graph[node].items()
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]["weight"]

        return weights

    @classmethod
    def getNodeWeights(cls, node, graph):
        return graph.degree(node, weight="weight")

    @classmethod
    def _setNode2Com(cls, graph):
        # initialize 
        node2com = {}
        edge_weights = defaultdict(lambda : defaultdict(float))
        for idx, node in enumerate(sorted(graph.nodes())):
            node2com[node] = idx
            for edge in graph[node].items():
                # edge = (node, {"weight" : edge_weight})
                edge_weights[node][edge[0]] = edge[1]["weight"]

        return node2com, edge_weights
