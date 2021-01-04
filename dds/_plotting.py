"""
Plotting of the dependencies.

https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py
"""
import pathlib
from collections import OrderedDict
from typing import NamedTuple, Optional, List, Tuple, Set, NewType, Dict

import pydotplus as pydot  # type: ignore

from dds.structures import FunctionInteractions, DDSPath, PyHash

EdgeType = NewType("EdgeType", int)
DirectEdge = EdgeType(1)
ImplicitEdge = EdgeType(2)
IndirectEdge = EdgeType(3)

_edge_styles: Dict[EdgeType, Dict[str, str]] = {
    DirectEdge: {"style": "solid"},
    IndirectEdge: {"style": "dashed"},
    ImplicitEdge: {"style": "dotted"},
}

NORMAL_NODE_STYLE = {"shape": "box"}

BLOB_NODE_STYLE = {"shape": "box", "fillcolor": "#E0E0E0", "style": "filled"}

EVAL_NODE_STYLE = {"shape": "box", "fillcolor": "#90EE90", "style": "filled"}


def build_graph(
    fis: FunctionInteractions,
    present_blobs: Optional[Set[PyHash]],
    indirect_refs: Dict[DDSPath, PyHash],
) -> pydot.Dot:
    s = _structure(fis, indirect_refs)
    g = pydot.Dot("interactions", graph_type="digraph", rankdir="BT")
    indirect_hashes = set(indirect_refs.values())
    for n in s.fnodes:
        style = NORMAL_NODE_STYLE
        if n.node_hash in indirect_hashes:
            style = BLOB_NODE_STYLE
        if present_blobs:
            style = BLOB_NODE_STYLE if n.node_hash in present_blobs else EVAL_NODE_STYLE
        g.add_node(pydot.Node(name=str(n.path), **style))
    for e in s.deps:
        style = _edge_styles[e.edge_type]
        g.add_edge(pydot.Edge(src=str(e.from_path), dst=str(e.to_path), **style))
    return g


def draw_graph(
    fis: FunctionInteractions,
    out: pathlib.Path,
    present_blobs: Optional[Set[PyHash]],
    indirect_refs: Dict[DDSPath, PyHash],
) -> None:
    out.write_bytes(
        build_graph(fis, present_blobs, indirect_refs).create(format=out.suffix[1:])
    )


class Node(NamedTuple):
    path: DDSPath
    node_hash: PyHash


class Edge(NamedTuple):
    from_path: DDSPath
    to_path: DDSPath
    edge_type: EdgeType


class Graph(NamedTuple):
    fnodes: List[Node]
    deps: List[Edge]


def _structure(
    fis: FunctionInteractions, indirect_refs: Dict[DDSPath, PyHash]
) -> Graph:
    nodes: OrderedDict[PyHash, Node] = OrderedDict()
    all_refs: Dict[DDSPath, PyHash] = dict(indirect_refs)
    # The head nodes for each function
    head_nodes: OrderedDict[PyHash, List[Node]] = OrderedDict()
    # The set of all known dependencies to a node
    node_deps: OrderedDict[PyHash, Set[PyHash]] = OrderedDict()
    deps: OrderedDict[Tuple[PyHash, PyHash], Edge] = OrderedDict()

    # Returns the list of head nodes:
    # All the nodes that can be evaluated independently inside a function.
    def traverse(fis_: FunctionInteractions) -> List[Node]:
        sig = fis_.fun_return_sig
        if sig in head_nodes:
            return head_nodes[sig]
        # Recurse
        sub_calls: List[Tuple[List[Node], FunctionInteractions]] = [
            (traverse(sub_fis), sub_fis) for sub_fis in fis_.parsed_body
        ]
        sub_nodes: List[Node] = sorted(
            list(
                dict(
                    [(n.node_hash, n) for (l_nodes, _) in sub_calls for n in l_nodes]
                ).values()
            ),
            key=lambda n: n.node_hash,
        )
        # Add implicit dependencies between context-dependent nodes.
        # Current algorithm is not very smart: anything that has parameters is assumed to be context-dependent
        # (even if the parameters are known at introspection time)
        start_nodes: List[Node] = sub_calls[0][0] if sub_calls else []
        sub_set: Set[PyHash] = set(
            [k for n in sub_nodes for k in node_deps[n.node_hash]]
        )
        for (l1, fi) in sub_calls[1:]:
            # l1: List[Node]
            # fi: FunctionInteractions
            # If it is a context-independent function, add it to the list of potential implicit dependencies
            if len(fi.arg_input.named_args) == 0:
                start_nodes += l1
            # Otherwise, there is an implicit dep: introduce a single dep here
            else:
                for n1 in start_nodes:
                    for n2 in l1:
                        k1 = n1.node_hash
                        k2 = n2.node_hash
                        if k1 not in node_deps:
                            node_deps[k1] = set()
                        if k2 not in node_deps:
                            node_deps[k2] = set()
                        k = (k1, k2)
                        if (
                            k not in deps
                            and k2 not in node_deps[k1]
                            and k1 not in node_deps[k2]
                            and k1 not in sub_set
                            and k2 not in sub_set
                        ):
                            deps[k] = Edge(n1.path, n2.path, ImplicitEdge)
                            node_deps[k2].add(k1)
                            node_deps[k2].update(node_deps[k1])
                # Restart the list of deps just based on the last implicit node
                # It is an approximation of the actual computation flow, but enough for UI purposes
                start_nodes = l1
        if fis_.store_path is None:
            return sub_nodes
        else:
            # We are returning a path -> create a node
            res_node = Node(fis_.store_path, sig)
            nodes[sig] = res_node
            all_refs[fis_.store_path] = sig
            sub_set.update([n.node_hash for n in sub_nodes])
            node_deps[res_node.node_hash] = sub_set
            for sub_n in sub_nodes:
                k = (sub_n.node_hash, res_node.node_hash)
                if k not in deps or deps[k].edge_type != DirectEdge:
                    deps[k] = Edge(sub_n.path, res_node.path, DirectEdge)
                node_deps[res_node.node_hash].update(node_deps[sub_n.node_hash])
            # Add the indirect references
            for p in fis_.indirect_deps:
                assert p in all_refs, p
                sig2 = all_refs[p]
                if sig2 not in nodes:
                    nodes[sig2] = Node(p, sig2)
                k = (sig2, res_node.node_hash)
                if k not in deps:
                    deps[k] = Edge(p, res_node.path, IndirectEdge)
            return [res_node]

    traverse(fis)
    return Graph(list(nodes.values()), list(deps.values()))
