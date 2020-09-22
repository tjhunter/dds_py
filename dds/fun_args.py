import hashlib
import inspect
import logging
from pathlib import PurePosixPath
from collections import OrderedDict
from inspect import Parameter

from .structures import CanonicalPath


from typing import (
    Tuple,
    Callable,
    Any,
    Dict,
    Optional,
    NamedTuple,
    OrderedDict as OrderedDictType,
)


from .structures import PyHash, FunctionArgContext


_logger = logging.getLogger(__name__)


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
    if isinstance(x, tuple):
        return dds_hash(list(x))
    if isinstance(x, PurePosixPath):
        return algo_str(str(x))
    raise NotImplementedError(str(type(x)))


def get_arg_ctx(
    f: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> FunctionArgContext:
    if len(kwargs) > 0:
        raise NotImplementedError(f"kwargs")
    arg_sig = inspect.signature(f)
    _logger.debug(f"get_arg_ctx: {f}: arg_sig={arg_sig} args={args}")
    args_hashes = []
    for (idx, (n, p_)) in enumerate(arg_sig.parameters.items()):
        p: inspect.Parameter = p_
        _logger.debug(f"get_arg_ctx: {f}: idx={idx} n={n} p={p}")
        if p.kind != Parameter.POSITIONAL_OR_KEYWORD:
            raise NotImplementedError(f"{p.kind} {f} {arg_sig}")
        elif p.default != Parameter.empty and idx >= len(args):
            # Use the default argument as an input
            # This assumes that the user does not mutate the argument, which is
            # a warning/errors in most linters.
            h = dds_hash(p.default or "__none__")
            # raise NotImplementedError(f"{p} {p.default} {f} {arg_sig}")
        elif len(args) <= idx:
            raise NotImplementedError(f"{len(args)} {arg_sig}")
        else:
            h = dds_hash(args[idx])
        args_hashes.append((n, h))
    return FunctionArgContext(OrderedDict(args_hashes), None)
