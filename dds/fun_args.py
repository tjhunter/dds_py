# from __future__ import annotations

import ast
import hashlib
import inspect
import struct
import logging
from collections import OrderedDict
from inspect import Parameter
from pathlib import PurePosixPath
import numpy as np  # type: ignore
from typing import Tuple, Callable, Any, Dict, List, Optional, NewType

from .structures import CanonicalPath
from .structures import PyHash, FunctionArgContext

_logger = logging.getLogger(__name__)

HashKey = NewType("HashKey", str)


def dds_hash_commut(i: List[Tuple[HashKey, PyHash]]) -> PyHash:
    """
    Takes a dictionary-like structure of keys and values and returns a hash of it with the
    following commutatitivity property: the hash is stable under permutation of elements
    in the key.
    """

    def digest(kv: Tuple[HashKey, PyHash]) -> str:
        b = hashlib.sha256(kv[0].encode("utf-8"))
        b.update(kv[1].encode("utf-8"))
        return b.hexdigest()

    assert i
    res = digest(i[0])
    for c in i[1:]:
        _res = int(res, 16) ^ int(digest(c), 16)
        res = "{:x}".format(_res)
    return PyHash(res)


def _algo_str(s: str) -> PyHash:
    return _algo_bytes(s.encode("utf-8"))


def _algo_bytes(b: bytes) -> PyHash:
    return PyHash(hashlib.sha256(b).hexdigest())


def dds_hash(x: Any) -> PyHash:

    if isinstance(x, str):
        return _algo_str(x)
    if isinstance(x, float):
        return _algo_bytes(struct.pack("!d", x))
    if isinstance(x, int):
        return _algo_bytes(struct.pack("!l", x))
    if isinstance(x, list):
        return _algo_str("|".join([dds_hash(y) for y in x]))
    if isinstance(x, CanonicalPath):
        return _algo_str(repr(x))
    if isinstance(x, tuple):
        return dds_hash(list(x))
    if isinstance(x, PurePosixPath):
        return _algo_str(str(x))
    raise NotImplementedError(str(type(x)))


def get_arg_list(
    f: Callable,  # type: ignore
) -> List[str]:
    arg_sig = inspect.signature(f)
    return list(arg_sig.parameters.keys())


def get_arg_ctx(
    f: Callable,  # type: ignore
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
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


def get_arg_ctx_ast(
    f: Callable,  # type: ignore
    args: List[ast.AST],
) -> "OrderedDict[str, Optional[PyHash]]":
    """
    Gets the arg context based on the AST.
    """
    arg_sig = inspect.signature(f)
    _logger.debug(f"get_arg_ctx: {f}: arg_sig={arg_sig} args={args}")
    args_hashes: List[Tuple[str, Optional[PyHash]]] = []
    for (idx, (n, p_)) in enumerate(arg_sig.parameters.items()):
        p: inspect.Parameter = p_
        _logger.debug(f"get_arg_ctx: {f}: idx={idx} n={n} p={p}")
        h: Optional[PyHash]
        if p.kind != Parameter.POSITIONAL_OR_KEYWORD:
            raise NotImplementedError(f"{p.kind} {f} {arg_sig}")
        elif p.default != Parameter.empty and idx >= len(args):
            # Use the default argument as an input
            # This assumes that the user does not mutate the argument, which is
            # a warning/errors in most linters.
            h = dds_hash(p.default or "__none__")
            # raise NotImplementedError(f"{p} {p.default} {f} {arg_sig}")
        elif len(args) <= idx:
            # TODO: it should be dealt with kwargs here
            # raise NotImplementedError(f"{f} {len(args)} {arg_sig}")
            h = None
        else:
            # Get the AST node:
            node = args[idx]
            # NameConstant for python 3.5 - 3.7
            if isinstance(node, (ast.Constant, ast.NameConstant)):
                # We can deal with some constant nodes
                default_ob = node.value if node.value is not None else "__none__"
                h = dds_hash(default_ob)
            else:
                # Cannot deal with it for the time being
                h = None
        args_hashes.append((n, h))
    return OrderedDict(args_hashes)
