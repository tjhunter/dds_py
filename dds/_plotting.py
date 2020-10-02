"""
Plotting of the dependencies.

https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
"""
import pathlib
from collections import OrderedDict
from typing import NamedTuple, Optional, List, Tuple, Set

import pydotplus as pydot  # type: ignore

from dds.structures import FunctionInteractions, DDSPath, PyHash


NORMAL_NODE_STYLE = {"shape": "box"}

BLOB_NODE_STYLE = {"shape": "box", "fillcolor": "#E0E0E0", "style": "filled"}

EVAL_NODE_STYLE = {"shape": "box", "fillcolor": "#90EE90", "style": "filled"}

IMPLICIT_EDGE_STYLE = {"style": "dashed"}

EXPLICIT_EDGE_STYLE = {"style": "solid"}


def build_graph(
    fis: FunctionInteractions, present_blobs: Optional[Set[PyHash]]
) -> pydot.Dot:
    s = _structure(fis)
    g = pydot.Dot("interactions", graph_type="digraph", rankdir="BT")
    for n in s.fnodes:
        style = NORMAL_NODE_STYLE
        if present_blobs:
            style = BLOB_NODE_STYLE if n.node_hash in present_blobs else EVAL_NODE_STYLE
        g.add_node(pydot.Node(name=str(n.path), **style))
    for e in s.deps:
        style = IMPLICIT_EDGE_STYLE if e.is_implicit else EXPLICIT_EDGE_STYLE
        g.add_edge(pydot.Edge(src=str(e.from_path), dst=str(e.to_path), **style))
    return g


def draw_graph(
    fis: FunctionInteractions, out: pathlib.Path, present_blobs: Optional[Set[PyHash]]
) -> None:
    out.write_bytes(build_graph(fis, present_blobs).create(format=out.suffix[1:]))


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
    def traverse(fis_: FunctionInteractions) -> List[Node]:
        sig = fis_.fun_return_sig
        if sig in nodes:
            return [nodes[sig]]
        # Recurse
        sub_calls = [traverse(sub_fis) for sub_fis in fis_.parsed_body]
        sub_nodes: List[Node] = [n for l_nodes in sub_calls for n in l_nodes]
        # Implicit dependencies
        for (n1, n2) in zip(sub_nodes[:-1], sub_nodes[1:]):
            k = (n1.node_hash, n2.node_hash)
            if k not in deps:
                deps[k] = Edge(n1.path, n2.path, True)
        if fis_.store_path is None:
            return sub_nodes
        else:
            # We are returning a path -> create a node
            res_node = Node(fis_.store_path, sig)
            nodes[sig] = res_node
            for sub_n in sub_nodes:
                k = (sub_n.node_hash, res_node.node_hash)
                if k not in deps or deps[k].is_implicit:
                    deps[k] = Edge(sub_n.path, res_node.path, False)
            return [res_node]

    traverse(fis)
    return Graph(list(nodes.values()), list(deps.values()))
