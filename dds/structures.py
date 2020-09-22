from __future__ import annotations
from functools import total_ordering
from pathlib import PurePosixPath
from collections import OrderedDict

from typing import (
    TypeVar,
    Callable,
    Any,
    NewType,
    NamedTuple,
    FrozenSet,
    Optional,
    Dict,
    List,
    Tuple,
    Type,
    Union,
    # OrderedDict as OrderedDictType,
)

# from .fun_args import FunctionArgContext

# A path to an object in the DDS store.
DDSPath = NewType("DDSPath", str)


# The hash of a python object
PyHash = NewType("PyHash", str)


class KSException(BaseException):
    pass


class EvalContext(NamedTuple):
    """
    The evaluation context created when evaluating a call.
    """

    requested_paths: Dict[DDSPath, PyHash]


# The name of a codec protocol.
ProtocolRef = NewType("ProtocolRef", str)

# A URI like wrapper to put stuff somewhere.
# The exact content and schema is determined by the store.
GenericLocation = NewType("GenericLocation", str)

CodecBackend = NewType("CodecBackend", str)


class CodecProtocol(object):
    def ref(self) -> ProtocolRef:
        pass

    def handled_types(self) -> List[Type[Any]]:
        """ The list of types that this codec can handle """
        pass

    def serialize_into(self, blob: Any, loc: GenericLocation) -> None:
        """
        Simple in-memory serialization
        """
        pass

    def deserialize_from(self, loc: GenericLocation) -> Any:
        """ Simple in-memory deserialization """
        pass


class BlobMetaData(NamedTuple):
    protocol: ProtocolRef
    # TODO: creation date?


@total_ordering
class CanonicalPath(object):
    def __init__(self, p: List[str]):
        self._path = p

    def __hash__(self):
        return hash(tuple(self._path))

    def append(self, s: str) -> "CanonicalPath":
        return CanonicalPath(self._path + [s])

    def head(self) -> str:
        return self._path[0]

    def tail(self) -> "CanonicalPath":
        return CanonicalPath(self._path[1:])

    def get(self, i: int) -> str:
        return self._path[i]

    def __len__(self):
        return len(self._path)

    def __repr__(self):
        x = ".".join(self._path)
        return f"<{x}>"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __ne__(self, other):
        return not (repr(self) == repr(other))

    def __lt__(self, other):
        return repr(self) < repr(other)


# The path of a local dependency from the perspective of a function, as read from the AST
# It is always a relative path without root.
LocalDepPath = NewType("LocalDepPath", PurePosixPath)


class ExternalDep(NamedTuple):
    """
    An external dependency to a function (not a function, this is tracked by FunctionInteraction)
    """

    # The local path, as called within the function
    local_path: LocalDepPath
    # The path of the object
    path: CanonicalPath
    # The signature of the object
    sig: PyHash


class FunctionArgContext(NamedTuple):
    # The keys of the arguments that are known at call time
    named_args: OrderedDict[str, Optional[PyHash]]
    # The key of the environment when calling the function
    inner_call_key: Optional[PyHash]

    @staticmethod
    def relevant_keys(fac: "FunctionArgContext") -> List[PyHash]:
        # TODO: this just sends back a list of hashes. This is not great if the names change?
        keys = [key for (_, key) in fac.named_args.items()]
        if any([key is None for key in keys]):
            # Missing some keys in the named arguments -> rely on the inner call key for the hash
            # TODO: this should not be a bug because of the root context, but it would be good to check.
            return [] if fac.inner_call_key is None else [fac.inner_call_key]
        else:
            return keys  # type: ignore


class FunctionInteractions(NamedTuple):
    arg_input: FunctionArgContext
    # The signature of the function (function body)
    fun_body_sig: PyHash
    # The signature of the return of the function (including the evaluated args)
    fun_return_sig: PyHash
    # The external dependencies
    # TODO: merge it with parsed_body
    external_deps: List[ExternalDep]
    # In order, all the content from the parsed body of the function.
    # TODO: real type is FunctionInteractions but mypy does not support yet recursive types
    parsed_body: List["Any"]  # type: ignore
    # The path, if the output is expected to be stored
    store_path: Optional[DDSPath]
    # The path of the function
    fun_path: CanonicalPath

    @classmethod
    def all_store_paths(
        cls, fi: "FunctionInteractions"
    ) -> OrderedDict[DDSPath, PyHash]:
        res: List[Tuple[DDSPath, PyHash]] = []
        if fi.store_path is not None:
            res.append((fi.store_path, fi.fun_return_sig))
        for fi0 in fi.parsed_body:
            if isinstance(fi0, FunctionInteractions):
                res += cls.all_store_paths(fi0).items()
        return OrderedDict(res)

    @classmethod
    def pprint_tree(
        cls, fi: "FunctionInteractions", printer: Callable[[str], None]
    ) -> None:
        class Node:
            def __init__(self, value=None, children=None):
                if children is None:
                    children = []
                self.value, self.children = value, children

        def pprint_tree_(node: Node, _prefix: str = "", _last: bool = True) -> None:
            s = _prefix + ("`- " if _last else "|- ") + str(node.value)
            printer(s)
            _prefix += "   " if _last else "|  "
            child_count = len(node.children)
            for i, child in enumerate(node.children):
                _last = i == (child_count - 1)
                pprint_tree_(child, _prefix, _last)

        def to_nodes(fi_: FunctionInteractions) -> Node:
            # TODO: add full path
            name = f"Fun {fi_.fun_path} {fi_.store_path} <- {fi_.fun_return_sig}"
            nodes = [
                Node(value=f"dep: {ed.local_path} -> {ed.path}: {ed.sig}")  # type: ignore
                for ed in fi_.external_deps
            ] + [
                to_nodes(fi0)
                for fi0 in fi_.parsed_body
                if isinstance(fi0, FunctionInteractions)
            ]
            return Node(value=name, children=nodes)  # type: ignore

        pprint_tree_(to_nodes(fi))
