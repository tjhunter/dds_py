"""
Utilities related to structures
"""
# from __future__ import annotations

import pathlib
from collections import OrderedDict
from typing import (
    Callable,
    Any,
    Optional,
    List,
    Tuple,
)
from typing import Union

from .structures import (
    DDSPath,
    KSException,
    FunctionInteractions,
    PyHash,
)


class DDSPathUtils(object):
    @staticmethod
    def create(p: Union[str, pathlib.Path]) -> DDSPath:
        if isinstance(p, str):
            if not p or p[0] != "/":
                raise KSException(
                    f"Provided path {p} is not absolute. All paths must be absolute"
                )
            # TODO: more checks
            return DDSPath(p)
        if isinstance(p, pathlib.Path):
            if not p.is_absolute():
                raise KSException(
                    f"Provided path {p} is not absolute. All paths must be absolute"
                )
            return DDSPath(p.absolute().as_posix())
        raise NotImplementedError(f"Cannot make a path from object type {type(p)}: {p}")


class _PrintNode(object):
    def __init__(
        self, value: Optional[Any] = None, children: "Optional[List[_PrintNode]]" = None
    ):
        if children is None:
            children = []
        self.value, self.children = value, children


class FunctionInteractionsUtils(object):
    @classmethod
    def all_store_paths(cls, fi: FunctionInteractions) -> OrderedDict[DDSPath, PyHash]:
        res: List[Tuple[DDSPath, PyHash]] = []
        if fi.store_path is not None:
            res.append((fi.store_path, fi.fun_return_sig))
        for fi0 in fi.parsed_body:
            if isinstance(fi0, FunctionInteractions):
                res += cls.all_store_paths(fi0).items()
        return OrderedDict(res)

    @classmethod
    def pprint_tree(
        cls, fi: FunctionInteractions, printer: Callable[[str], None]
    ) -> None:
        def pprint_tree_(
            node: _PrintNode, _prefix: str = "", _last: bool = True
        ) -> None:
            s = _prefix + ("`- " if _last else "|- ") + str(node.value)
            printer(s)
            _prefix += "   " if _last else "|  "
            child_count = len(node.children)
            for i, child in enumerate(node.children):
                _last = i == (child_count - 1)
                pprint_tree_(child, _prefix, _last)

        def to_nodes(fi_: FunctionInteractions) -> _PrintNode:
            path = (
                f"@ {fi_.fun_return_sig}"
                if fi_.store_path is None
                else f"{fi_.store_path} <- {fi_.fun_return_sig}"
            )
            # TODO: add full path
            name = f"Fun {fi_.fun_path} {path}"
            call_ctx = fi_.arg_input.inner_call_key
            nodes = (
                ([_PrintNode(value=f"Ctx {call_ctx}")] if call_ctx is not None else [])
                + [
                    _PrintNode(value=f"Arg {arg_name}: {arg_key}")
                    for (arg_name, arg_key) in fi_.arg_input.named_args.items()
                ]
                + [
                    _PrintNode(value=f"Dep {ed.local_path} -> {ed.path}: {ed.sig}")
                    for ed in fi_.external_deps
                ]
                + [
                    to_nodes(fi0)
                    for fi0 in fi_.parsed_body
                    if isinstance(fi0, FunctionInteractions)
                ]
            )

            return _PrintNode(value=name, children=nodes)

        pprint_tree_(to_nodes(fi))
