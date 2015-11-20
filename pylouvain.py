#!/usr/bin/env python3
# coding: utf-8

import sys
import copy

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
        all_edge_weights = graph.size(weight="weight")

        node2com = cls._runFirstPhase(node2com, edge_weights)
        cls.show(node2com)

        partition = copy.deepcopy(node2com)
        new_node2com, new_edge_weights = cls._runSecondPhase(node2com, edge_weights)
        cls.show(new_node2com)

        while True:
            new_node2com = cls._runFirstPhase(new_node2com, new_edge_weights)
            partition = cls._updatePartition(new_node2com, partition)
            _new_node2com, _new_edge_weights = cls._runSecondPhase(new_node2com, new_edge_weights)
            if new_node2com == _new_node2com:
                break
        return partition

    @classmethod
    def computeModularity(cls):
        pass

    @classmethod
    def _updatePartition(cls, new_node2com, partition):
        # new_node2com : {'古いcom_id' : "新しいcom_id"}
        reverse_partition = {v:k for k,v in partition.items()}
        for old_com_id, new_com_id in new_node2com.items():
            partition[reverse_partition[old_com_id]] = new_com_id
        return partition

    @classmethod
    def _runFirstPhase(cls, node2com, edge_weights):
        all_edge_weights = sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        status = True
        while status:
            statuses = []
            for node in sorted(list(node2com.keys())):
                statuses = []
                com_id = node2com[node]
                neigh_nodes = sorted([edge[0] for edge in cls.getNeighborNodes(node, edge_weights)])

                max_delta = 0.
                max_com_id = com_id
                max_n = None
                communities = {}
                for neigh_node in sorted(neigh_nodes):
                    node2com_copy = copy.deepcopy(node2com)
                    if node2com_copy[neigh_node] in communities:
                        continue
                    communities[node2com_copy[neigh_node]] = 1
                    node2com_copy[node] = node2com_copy[neigh_node]

                    # 隣接クラスタに移動した際のModularityの差分
                    delta_q = 2 * cls.getNodeWeightInCluster(node, node2com_copy, edge_weights) - cls.getTotWeight(node, node2com_copy, edge_weights) * cls.getNodeWeights(node, edge_weights) / all_edge_weights
                    # print("neighbor_id: {}, k_i_in: {}, tot: {}, k_i: {}, max_delta: {}".format(neigh_node,
                    #                                                           cls.getNodeWeightInCluster(node, node2com_copy, edge_weights),
                    #                                                           cls.getTotWeight(node, node2com_copy, edge_weights),
                    #                                                           cls.getNodeWeights(node, edge_weights),
                                                                               delta_q))

                    if delta_q > max_delta:
                        max_delta = delta_q
                        max_com_id = node2com_copy[neigh_node]
                        max_n = copy.deepcopy(node2com_copy)

                #print("k_i_in: {}, tot: {}, k_i: {}, max_delta: {}".format(cls.getNodeWeightInCluster(node, node2com_copy, edge_weights),
                #                                            cls.getTotWeight(node, max_n, edge_weights),
                #                                            cls.getNodeWeights(node, edge_weights),
                #                                            max_delta))
                #print(node2com, max_com_id)
                node2com[node] = max_com_id
                #print(com_id, max_com_id)
                statuses.append(com_id != max_com_id)

            if sum(statuses) == 0:
                break

        return node2com

    @classmethod
    def _runSecondPhase(cls, node2com, edge_weights):
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda : defaultdict(float))

        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        for edge in permutations(nodes, 2):
            if edge[0] not in edge_weights: continue
            if edge[1] not in edge_weights[edge[0]]: continue
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][edge[1]]

        for node in new_node2com.keys():
            new_edge_weights[node][node] = (new_edge_weights[node][node] / 2) ** 2

        return new_node2com, new_edge_weights

    @classmethod
    def getTotWeight(cls, node, node2com, edge_weights):
        nodes = []
        for n, com_id in node2com.items():
            if com_id == node2com[node] and node != n:
                nodes.append(n)

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    @classmethod
    def getNeighborNodes(cls, node, edge_weights):
        if node not in edge_weights:
            return 0
        return list(edge_weights[node].items())

    @classmethod
    def getNodeWeightInCluster(cls, node, node2com, edge_weights):
        neigh_nodes = cls.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]

        return weights
    
    @classmethod
    def getNodeWeights(cls, node, edge_weights):
        return sum([weight for weight in edge_weights[node].values()])

    @classmethod
    def _setNode2Com(cls, graph):
        # initialize 
        node2com = {}
        edge_weights = {}
        for idx, node in enumerate(sorted(graph.nodes())):
            node2com[node] = idx
            edge_weights[node] = {}
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]["weight"]

        return node2com, edge_weights
