"""
Plotting of the dependencies.

https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
"""

import pydotplus as pydot
from collections import OrderedDict
from typing import (
    Any,
    NewType,
    NamedTuple,
    Optional,
    Dict,
    Tuple,
    List,
    Type,
)
from dds.structures import KSException, FunctionInteractions, DDSPath, PyHash

def build_graph(fis: FunctionInteractions) -> pydot.Dot:
    g = pydot.Dot("interactions", graph_type="digraph", )


class Node(NamedTuple):
    path: DDSPath
    node_hash: PyHash

class Edge(NamedTuple):
    from_hash: PyHash
    to_hash: PyHash
    is_implicit: True

class Graph(NamedTuple):
    fnodes: List[Node]
    deps: List[Edge]

def _structure(fis: FunctionInteractions) -> Graph:
    nodes: OrderedDict[PyHash, Node] = OrderedDict()
    deps: List[Edge] = []

    # Returns the last node evaluated
    def traverse(fis_: FunctionInteractions) -> Optional[Node]:
        sig = fis_.fun_return_sig
        if sig in nodes:
            return nodes[sig]
        # Recurse
        sub_calls = [traverse(sub_fis) for sub_fis in fis_.parsed_body]
        sub_nodes: List[Node] = [n for n in sub_calls if n is not None]
        # Implicit dependencies
        for (n1, n2) in zip(sub_nodes[:-1], sub_nodes[1:]):
            deps.append(Edge(n1.node_hash, n2.node_hash, True))
        if fis_.store_path is None:
            return sub_nodes[-1] if sub_nodes else None
        else:
            # We are returning a path -> create a node
            res_node = Node(fis_.store_path, sig)
            for sub_n in sub_nodes:
                deps.append(Edge(sub_n.node_hash, res_node.node_hash, False))
            return res_node

    traverse(fis)
    return Graph(list(nodes.values()), deps)
