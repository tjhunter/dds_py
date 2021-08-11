"""
Utilities related to structures
"""
import itertools
import logging
import pathlib
from collections import OrderedDict
from pathlib import PurePosixPath
from typing import Callable, Any, Optional, List, Tuple, Set
from typing import Union

from .structures import (
    DDSPath,
    DDSException,
    FunctionInteractions,
    PyHash,
    LocalDepPath,
    FunctionIndirectInteractions,
    SupportedType,
    CanonicalPath,
    DDSErrorCode,
)

_logger = logging.getLogger(__name__)


class DDSPathUtils(object):
    @staticmethod
    def create(p: Union[str, pathlib.Path]) -> DDSPath:
        if isinstance(p, str):
            if not p or p[0] != "/":
                raise DDSException(
                    f"Provided path {p} is not absolute. All paths must be absolute",
                    DDSErrorCode.PATH_NOT_ABSOLUTE,
                )
            # TODO: more checks
            return DDSPath(p)
        if isinstance(p, pathlib.Path):
            if not p.is_absolute():
                raise DDSException(
                    f"Provided path {p} is not absolute. All paths must be absolute",
                    DDSErrorCode.PATH_NOT_ABSOLUTE,
                )
            return DDSPath(p.absolute().as_posix())
        raise NotImplementedError(f"Cannot make a path from object type {type(p)}: {p}")

    @staticmethod
    def split(p: DDSPath) -> Tuple[str, Optional[DDSPath]]:
        l = p.split("/")
        if len(l) == 1:
            return (l[0], None)
        if len(l) == 2 and l[0] == "":
            return (l[1], None)
        return (l[1], DDSPathUtils.create("/" + "/".join(l[2:])))


class _PrintNode(object):
    def __init__(
        self, value: Optional[Any] = None, children: "Optional[List[_PrintNode]]" = None
    ):
        if children is None:
            children = []
        self.value, self.children = value, children


class FunctionInteractionsUtils(object):
    @classmethod
    def non_terminal_leaves(
        cls, paths: List[DDSPath], current_prefix: Optional[DDSPath]
    ) -> List[DDSPath]:
        """
        Returns a list of paths that are terminal but also with leaves (for example: [/f, /f/g] -> [/f]).
        """
        # _logger.debug("non_terminal in: %s %s", paths, current_prefix)
        empty_path = DDSPath("/")
        res: List[DDSPath] = []
        non_empty_paths = [p for p in paths if p != empty_path]

        # There are both empty paths and non-empty paths. it should be one or the other.
        if len(paths) > len(non_empty_paths) > 0 and current_prefix is not None:
            res.append(current_prefix)

        splits = [DDSPathUtils.split(p) for p in non_empty_paths]
        # _logger.debug("non_terminal splits: %s", splits)
        groups = itertools.groupby(splits, lambda x: x[0])
        for (key, l) in groups:
            sub: List[DDSPath] = [(p if p is not None else empty_path) for (_, p) in l]
            # _logger.debug("non_terminal: %s %s", key, sub)
            sub_path: DDSPath = DDSPath(
                "/" + key
            ) if current_prefix is None else DDSPath(current_prefix + "/" + key)
            res += FunctionInteractionsUtils.non_terminal_leaves(sub, sub_path)

        return res

    @classmethod
    def all_store_paths(
        cls, fi: FunctionInteractions
    ) -> "OrderedDict[DDSPath, PyHash]":
        res: List[Tuple[DDSPath, PyHash]] = []
        if fi.store_path is not None:
            res.append((fi.store_path, fi.fun_return_sig))
        for fi0 in fi.parsed_body:
            if isinstance(fi0, FunctionInteractions):
                res += cls.all_store_paths(fi0).items()
        return OrderedDict(res)

    @classmethod
    def all_indirect_deps(cls, fis: FunctionInteractions) -> Set[DDSPath]:
        res = set(fis.indirect_deps)
        for fis_ in fis.parsed_body:
            res.update(cls.all_indirect_deps(fis_))
        return res

    @classmethod
    def pprint_tree(
        cls,
        fi: FunctionInteractions,
        present_blobs: Optional[Set[PyHash]],
        printer: Callable[[str], None],
        only_new_nodes: bool = True,
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

        printed_nodes: Set[PyHash] = set()

        def to_nodes(fi_: FunctionInteractions) -> _PrintNode:
            if present_blobs is not None:
                if fi_.fun_return_sig in present_blobs:
                    status = "<-- "
                else:
                    status = "<-* "
            else:
                status = "--- "
            sig = str(fi_.fun_return_sig)[:10]
            path = (
                f"@ {sig}"
                if fi_.store_path is None
                else f"{fi_.store_path} {status}{sig}"
            )
            # TODO: add full path
            name = f"Fun {fi_.fun_path} {path}"
            call_ctx = str(fi_.arg_input.inner_call_key)[:10]
            # Already printed -> just put the first line
            if only_new_nodes and fi_.fun_return_sig in printed_nodes:
                return _PrintNode(value="~~" + name, children=[])
            nodes = (
                ([_PrintNode(value=f"Ctx {call_ctx}")] if call_ctx is not None else [])
                + [
                    _PrintNode(value=f"Arg {arg_name}: {str(arg_key)[:10]}")
                    for (arg_name, arg_key) in fi_.arg_input.named_args.items()
                ]
                + [
                    _PrintNode(
                        value=f"Dep {ed.local_path} -> {ed.path}: {str(ed.sig)[:10]}"
                    )
                    for ed in fi_.external_deps
                    if ed.sig is not None
                ]
                + [
                    _PrintNode(value=f"Ext {ed.local_path} -> {ed.path}")
                    for ed in fi_.external_deps
                    if ed.sig is None
                ]
                + [_PrintNode(value=f"Ind {ed}") for ed in fi_.indirect_deps]
                + [
                    to_nodes(fi0)
                    for fi0 in fi_.parsed_body
                    if isinstance(fi0, FunctionInteractions)
                ]
            )
            printed_nodes.add(fi_.fun_return_sig)

            return _PrintNode(value=name, children=nodes)

        pprint_tree_(to_nodes(fi))


class LocalDepPathUtils(object):
    @staticmethod
    def tail(p: LocalDepPath) -> LocalDepPath:
        ps = p.parts[1:]
        return LocalDepPath(pathlib.PurePosixPath("/".join(ps)))

    @staticmethod
    def empty(p: LocalDepPath) -> bool:
        ps = p.parts
        if not ps or (len(ps) == 1 and ps[0] == "."):
            return True
        return False


class FunctionIndirectInteractionUtils(object):
    @staticmethod
    def all_loads(fis: FunctionIndirectInteractions) -> Set[DDSPath]:
        # Use the underlying type (python limitation)
        res: Set[DDSPath] = {
            DDSPath(p) for p in fis.indirect_deps if isinstance(p, str)
        }
        for fis0 in fis.indirect_deps:
            if isinstance(fis0, FunctionIndirectInteractions):
                res.update(FunctionIndirectInteractionUtils.all_loads(fis0))
        return res

    @staticmethod
    def all_stores(fis: FunctionIndirectInteractions) -> Set[DDSPath]:
        res: Set[DDSPath] = {fis.store_path} if fis.store_path is not None else set()
        for fis0 in fis.indirect_deps:
            if isinstance(fis0, FunctionIndirectInteractions):
                res.update(list(FunctionIndirectInteractionUtils.all_stores(fis0)))
        return res


class SupportedTypeUtils(object):
    @staticmethod
    def from_type(t: type) -> SupportedType:
        if t is None:
            return SupportedTypeUtils.from_type(type(None))
        module = t.__module__
        if module is None or module == str.__class__.__module__:
            return SupportedType(t.__name__)
        return SupportedType(module + "." + t.__name__)


class CanonicalPathUtils(object):
    @staticmethod
    def from_list(l: List[str]) -> CanonicalPath:
        return CanonicalPath(PurePosixPath("/".join(l)))

    @staticmethod
    def head(p: CanonicalPath) -> str:
        return p._path.parts[0]

    @staticmethod
    def last(p: CanonicalPath) -> str:
        return p._path.parts[-1]

    @staticmethod
    def tail(p: CanonicalPath) -> CanonicalPath:
        return CanonicalPathUtils.from_list(list(p._path.parts[1:]))

    @staticmethod
    def append(p: CanonicalPath, o: Union[str, LocalDepPath]) -> CanonicalPath:
        if isinstance(o, str):
            return CanonicalPath(p._path.joinpath(o))
        elif isinstance(o, PurePosixPath):  # LocalDepPath
            s = str(o)
            if s.startswith("/"):
                s = s[1:]
            return CanonicalPath(p._path.joinpath(s))
        else:
            raise DDSException(f"Only str or PurePosixPath expected, got {type(o)} {o}")
