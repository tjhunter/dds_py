from functools import total_ordering

from typing import TypeVar, Callable, Any, NewType, NamedTuple, OrderedDict, FrozenSet, Optional, Dict, List, Tuple, Type


# A path to an object in the DDS store.
DDSPath = NewType("Path", str)


# The hash of a python object
PyHash = NewType("PyHash", str)


class FunctionInteractions(NamedTuple):
    # The signature of the function (function body)
    fun_body_sig: PyHash
    # The signature of the return of the function (including the evaluated args)
    fun_return_sig: PyHash
    # If the function is called within a function, this is the signature of the input
    # (currently depending on the input of the function)
    fun_context_input_sig: Optional[PyHash]
    # # The set of input paths for the function
    # inputs: FrozenSet[Path]
    # # The set of paths that will be committed by the function
    outputs: List[Tuple[DDSPath, PyHash]]
    # # The set of objects that will be read or committed to the cache
    # cache: FrozenSet[PyHash]
    # # The input of the function (with all the arguments), in case this is top level
    # fun_inputs: OrderedDict[str, PyHash]


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

    def handled_types(self) -> List[Type]:
        """ The list of types that this codec can handle """
        pass

    def serialize_into(self, blob: Any, loc: GenericLocation):
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

