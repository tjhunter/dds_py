from collections import OrderedDict
from enum import Enum
from functools import total_ordering
from pathlib import PurePath
from pathlib import PurePosixPath
from typing import Any, NewType, NamedTuple, Optional, Dict, List, Tuple


class ProcessingStage(str, Enum):
    """
    The processing stages by DDS:
    - analysis: parses all the functions and finds the functions that need to be evaluated
    - store_inspect: lists the new blobs that need to be added to the store (optional)
    - eval: evaluates the new blobs and pushes them to the store
    - path_commit: commit the paths to the store
    """

    ANALYSIS = "analysis"
    STORE_INSPECT = "store_inspect"
    EVAL = "eval"
    STORE_COMMIT = "store_commit"
    PATH_COMMIT = "path_commit"

    @staticmethod
    def all_phases() -> "List[ProcessingStage]":
        return [
            ProcessingStage.ANALYSIS,
            ProcessingStage.STORE_INSPECT,
            ProcessingStage.EVAL,
            ProcessingStage.STORE_COMMIT,
            ProcessingStage.PATH_COMMIT,
        ]


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

    stats_time: Dict[ProcessingStage, float]


# The name of a codec protocol.
ProtocolRef = NewType("ProtocolRef", str)

# A URI like wrapper to put stuff somewhere.
# The exact content and schema is determined by the store.
GenericLocation = NewType("GenericLocation", str)

CodecBackend = NewType("CodecBackend", str)

SupportedType = NewType("SupportedType", str)


class CodecProtocol(object):
    def ref(self) -> ProtocolRef:
        pass

    def handled_types(self) -> List[SupportedType]:
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


class FileCodecProtocol(object):
    """
    Simpler interface that just knows how to read and write a single file from a local file system.

    This file is expected to be read, written and deleted by the store.
    """

    def ref(self) -> ProtocolRef:
        pass

    def handled_types(self) -> List[SupportedType]:
        """ The list of types that this codec can handle """
        pass

    def serialize_into(self, blob: Any, loc: PurePath) -> None:
        """
        Puts the blob into the specified path. The path is assumed to be eventually filled with a file.
        """
        pass

    def deserialize_from(self, loc: PurePath) -> Any:
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

    def __len__(self) -> int:
        return len(self._path)

    def __repr__(self) -> str:
        x = ".".join(self._path)
        return f"<{x}>"

    def __eq__(self, other: Any) -> bool:
        return repr(self) == repr(other)

    def __ne__(self, other: Any) -> bool:
        return not (repr(self) == repr(other))

    def __lt__(self, other: Any) -> bool:
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


FunctionArgContextHash = NewType(
    "FunctionArgContextHash",
    Tuple[Optional[PyHash], Tuple[Tuple[str, Optional[PyHash]], ...]],
)


class FunctionArgContext(NamedTuple):
    # The keys of the arguments that are known at call time
    named_args: "OrderedDict[str, Optional[PyHash]]"
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

    @classmethod
    def as_hashable(cls, arg_ctx: "FunctionArgContext") -> FunctionArgContextHash:
        x: Tuple[Tuple[str, Optional[PyHash]], ...] = tuple(arg_ctx.named_args.items())
        return FunctionArgContextHash((arg_ctx.inner_call_key, x))


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
    parsed_body: List["Any"]
    # The path, if the output is expected to be stored
    store_path: Optional[DDSPath]
    # The path of the function
    fun_path: CanonicalPath
    # The indirect dependencies from this function
    indirect_deps: List[DDSPath]


class FunctionIndirectInteractions(NamedTuple):
    """
    The representation of all the indirect calls.
    This is done as a preprocessing step to find all the calls that need to be resolved before calling the main
    introspection function that compute the FunctionInteractions.
    """

    fun_path: CanonicalPath
    store_path: Optional[DDSPath]
    # TODO: real type is Union[DDSPath, FunctionIndirectInteractions]
    # The DDSPath object correspond to load() calls, the other calls correspond to sub function calls.
    # They are kept in order of calling to validate the order of calls:
    # indirect calls with that refer to a function also executed must happen after the function has executed
    indirect_deps: List["Any"]
