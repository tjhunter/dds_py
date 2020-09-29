"""
Plotting of the dependencies.

https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
"""
import pathlib
from collections import OrderedDict
from typing import NamedTuple, Optional, List, Tuple

import pydotplus as pydot  # type: ignore

from dds.structures import FunctionInteractions, DDSPath, PyHash


def build_graph(fis: FunctionInteractions) -> pydot.Dot:
    s = _structure(fis)
    g = pydot.Dot("interactions", graph_type="digraph", rankdir="BT")
    for n in s.fnodes:
        g.add_node(pydot.Node(name=str(n.path), **NODE_STYLE))
    for e in s.deps:
        style = IMPLICIT_EDGE_STYLE if e.is_implicit else EXPLICIT_EDGE_STYLE
        g.add_edge(pydot.Edge(src=str(e.from_path), dst=str(e.to_path), **style))
    return g


def draw_graph(fis: FunctionInteractions, out: pathlib.Path) -> None:
    out.write_bytes(build_graph(fis).create(format=out.suffix[1:]))


NODE_STYLE = {"shape": "octagon"}

IMPLICIT_EDGE_STYLE = {"style": "dashed"}

EXPLICIT_EDGE_STYLE = {"style": "solid"}


class Node(NamedTuple):
    path: DDSPath
    node_hash: PyHash


class Edge(NamedTuple):
    from_path: DDSPath
    to_path: DDSPath
    is_implicit: bool


class Graph(NamedTuple):
    fnodes: List[Node]
    deps: List[Edge]


def _structure(fis: FunctionInteractions) -> Graph:
    nodes: OrderedDict[PyHash, Node] = OrderedDict()
    deps: OrderedDict[Tuple[PyHash, PyHash], Edge] = OrderedDict()

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
            k = (n1.node_hash, n2.node_hash)
            if k not in deps:
                deps[k] = Edge(n1.path, n2.path, True)
        if fis_.store_path is None:
            return sub_nodes[-1] if sub_nodes else None
        else:
            # We are returning a path -> create a node
            res_node = Node(fis_.store_path, sig)
            for sub_n in sub_nodes:
                k = (sub_n.node_hash, res_node.node_hash)
                if k not in deps or deps[k].is_implicit:
                    deps[k] = Edge(sub_n.path, res_node.path, False)
            return res_node

    traverse(fis)
    return Graph(list(nodes.values()), list(deps.values()))
