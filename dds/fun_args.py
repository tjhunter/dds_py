import ast
import hashlib
import importlib
import inspect
import logging
from inspect import Parameter
import abc
import sys
from collections import OrderedDict
from enum import Enum
from types import ModuleType, FunctionType
import builtins
from functools import total_ordering
from typing import Tuple, Callable, Any, Dict, Set, Union, FrozenSet, Optional, \
    List, Type, NewType, NamedTuple, OrderedDict as OrderedDictType

from .structures import PyHash, DDSPath, FunctionInteractions, KSException, CanonicalPath

_logger = logging.getLogger(__name__)

from typing import Tuple, Callable, Any, Dict, Set, Union, FrozenSet, Optional, \
    List, Type, NewType, NamedTuple, OrderedDict as OrderedDictType


from .structures import PyHash, DDSPath, FunctionInteractions, KSException


class FunctionArgContext(NamedTuple):
    # The keys of the arguments that are known at call time
    named_args: OrderedDictType[str, Optional[PyHash]]
    # The key of the environment when calling the function
    inner_call_key: Optional[PyHash]


def dds_hash(x: Any) -> PyHash:
    def algo_str(s: str) -> PyHash:
        return PyHash(hashlib.sha256(s.encode()).hexdigest())

    if isinstance(x, str):
        return algo_str(x)
    if isinstance(x, int):
        return algo_str(str(x))
    if isinstance(x, list):
        return algo_str("|".join([dds_hash(y) for y in x]))
    if isinstance(x, CanonicalPath):
        return algo_str(repr(x))
    raise NotImplementedError(str(type(x)))


def get_arg_ctx(f: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> FunctionArgContext:
    if len(kwargs) > 0:
        raise NotImplementedError(f"kwargs")
    arg_sig = inspect.signature(f)
    _logger.debug(f"get_arg_ctx: {f}: arg_sig={arg_sig}")
    args_hashes = []
    for (idx, (n, p)) in enumerate(arg_sig.parameters.items()):
        p: inspect.Parameter = p
        if p.kind != Parameter.POSITIONAL_OR_KEYWORD:
            raise NotImplementedError(f"{p.kind} {f} {arg_sig}")
        if p.default != Parameter.empty:
            raise NotImplementedError(f"{p} {p.default} {f} {arg_sig}")
        if len(args) < idx:
            raise NotImplementedError(f"{len(args)} {arg_sig}")
        h = dds_hash(args[idx])
        args_hashes.append((n, h))
    return FunctionArgContext(OrderedDict(args_hashes), None)

