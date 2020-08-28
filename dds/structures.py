from typing import TypeVar, Callable, Any, NewType, NamedTuple, OrderedDict, FrozenSet, Optional, Dict, List, Tuple


Path = str


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
    outputs: List[Tuple[Path, PyHash]]
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
    requested_paths: Dict[Path, PyHash]

