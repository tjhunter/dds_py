# from __future__ import annotations

import ast
from collections import deque
import hashlib
import inspect
import logging
import struct
import dataclasses
from collections import OrderedDict
from inspect import Parameter
from pathlib import PurePosixPath
from typing import Tuple, Callable, Any, Dict, List, Optional, NewType, Union, Deque
import datetime

from .structures import CanonicalPath, DDSException, DDSErrorCode
from .structures import PyHash, FunctionArgContext, ArgName
from ._config import get_option

_logger = logging.getLogger(__name__)

HashKey = NewType("HashKey", str)


def dds_hash_commut(i: List[Tuple[HashKey, PyHash]]) -> Optional[PyHash]:
    """
    Takes a dictionary-like structure of keys and values and returns a hash of it with the
    following commutativity property: the hash is stable under permutation of elements
    in the key.

    Returns None if the input is empty
    """
    if not i:
        return None

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
    """
    Converts a python object to a hash.

    This function does not make use of the general __hash__ function of python:
    * it is meant to offer a more cryptographically stronger solution (full 2^256 space)
    * it is more restricted to the sort of inputs that are expected to be quickly hashed

    The expectation is that all the inputs tend to be primitive types or known structured types
    (named tuples or dataclass types). All other inputs are meant to be ignored unless their
    type has been flagged in an accepted module.

    """
    max_sequence_size: int = get_option("hash.max_sequence_size")

    # Some hints for tracing the offending object.
    trace: Deque[Union[int, str]] = deque()

    def current_path() -> str:
        return ".".join([str(i) if not isinstance(i, str) else i for i in list(trace)])

    def check_len(x: Any) -> None:
        if len(x) > max_sequence_size:
            raise DDSException(
                f"Object of type {type(x)} is a sequence of length {len(x)}. "
                f"Only sequences of length less than {max_sequence_size} are supported. "
                "This behaviour can be adjusted with the 'hash.max_sequence_size' option."
                f" Path hint: <{current_path()}>",
                DDSErrorCode.SEQUENCE_TOO_LONG,
            )

    def _dds_hash(elt: Any, path_item: Union[int, str, None]) -> PyHash:
        if path_item is not None:
            trace.append(path_item)
        res = _dds_hash0(elt)
        if trace:
            trace.pop()
        return res

    def _hash_dict_tuple(k: Any, v: Any) -> str:
        # Supposing for now that any dictionary key is well-behaved with respect to being converted to a string.
        if isinstance(k, str):
            n = k
        else:
            n = str(k)
        return _dds_hash(k, None) + "|" + _dds_hash(v, n)

    def _dds_hash0(elt: Any) -> PyHash:
        if elt is None:
            # TODO: this is not robust to adversarial changes.
            return PyHash(hashlib.sha256("__DDS_NONE__".encode("utf-8")).hexdigest())
        if isinstance(elt, str):
            return _algo_str(elt)
        if isinstance(elt, float):
            return _algo_bytes(struct.pack("!d", elt))
        if isinstance(elt, int):
            return _algo_bytes(struct.pack("!l", elt))
        if isinstance(elt, CanonicalPath):
            return _algo_str(repr(elt))
        if isinstance(elt, list):
            check_len(elt)
            return _algo_str(
                "|".join([_dds_hash(y, idx) for (idx, y) in enumerate(elt)])
            )
        if isinstance(elt, tuple):
            check_len(elt)
            return _dds_hash(list(elt), None)
        if isinstance(elt, PurePosixPath):
            return _algo_str(str(elt))
        if isinstance(elt, OrderedDict):
            check_len(elt)
            # Directly using the ordering of the items in the dictionary.
            return _dds_hash([_hash_dict_tuple(k, v) for (k, v) in elt.items()], None)
        if isinstance(elt, dict):
            # Starting from python 3.7 the order of the keys in a dictionary is the order of insertion
            # This code should return reliable results for CPython 3.6 and python 3.7+ (i.e. the overwhelming
            # majority of python interpreters out there).
            # Not going to check for obscure corner cases for now.
            check_len(elt)
            return _dds_hash([_hash_dict_tuple(k, v) for (k, v) in elt.items()], None)
        if dataclasses.is_dataclass(elt):
            names: List[str] = [f.name for f in dataclasses.fields(elt)]
            # TODO: this is not entirely accurate. The error message will show a 'list' type, but it is actually
            # a dataclass.
            check_len(names)
            vals = [_dds_hash(getattr(elt, n), n) for n in names]
            return _dds_hash(
                [_hash_dict_tuple(name, h) for (name, h) in zip(names, vals)], None
            )
        if isinstance(
            elt,
            (
                datetime.datetime,
                datetime.date,
                datetime.time,
                datetime.timedelta,
                datetime.timezone,
                datetime.tzinfo,
            ),
        ):
            # TODO: there may be some confusion because we use the same representation for the object
            # and its string
            return _dds_hash(repr(elt), None)
        msg = (
            f"The type {type(elt)} is currently not supported. The only supported types are "
            f"'well-known' types that are part of the standard data structures in the python library. "
            f"If you think your data type should be supported by DDS, please open a request ticket. "
            f"General Python classes will not be supported since they can carry arbitrary state and "
            f"cannot be easily compared. Consider using a dataclass, a dictionary or a named tuple instead."
        )
        raise DDSException(msg, DDSErrorCode.TYPE_NOT_SUPPORTED)

    return _dds_hash(x, None)


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
    arg_sig = inspect.signature(f)
    num_args = len(args)
    # _logger.debug(f"get_arg_ctx: {f}: arg_sig={arg_sig} args={args}")
    args_hashes = []
    for (idx, (n, p_)) in enumerate(arg_sig.parameters.items()):
        p: inspect.Parameter = p_
        # _logger.debug(f"get_arg_ctx: {f}: idx={idx} n={n} p={p}")
        if p.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.VAR_KEYWORD):
            raise NotImplementedError(
                f"Argument type not understood for function {f}: {p.kind} (see "
                f"exact definition in the module {Parameter}). Suggestion: "
                f"your function is probably using complex argument types. Use "
                f"simpler sorts of arguments (no kargs or kwargs)."
                f" The full signature was: {arg_sig}"
            )
        h: Optional[PyHash]
        if idx < num_args:
            # It is a list argument
            # TODO: should it discard arguments of not-whitelisted types?
            # TODO: raise a warning for non-whitelisted objects
            h = dds_hash(args[idx])
        else:
            # Either positional or default argument
            if n in kwargs:
                # positional argument
                h = dds_hash(kwargs[n])
            elif p.default != Parameter.empty:
                # Argument is not provided but it has a default value
                # Use the default argument as an input
                # This assumes that the user does not mutate the argument, which is
                # a warning/errors in most linters.
                # TODO: should it discard arguments of not-whitelisted types?
                # TODO: raise a warning for non-whitelisted objects
                h = dds_hash(p.default or "__none__")
            elif p.kind == Parameter.VAR_KEYWORD:
                # kwargs: for now, just ignored
                h = None
            elif p.kind == Parameter.POSITIONAL_OR_KEYWORD:
                # We are expecting a positional arguments, but no positional arguments
                # was provided. This is a programming error on the user side.
                raise DDSException(
                    f"Missing argument {n} for function {f}. "
                    f"DDS detected that the function {f} is missing the argument "
                    f"{n} of type {p.kind}. This would trigger an error during the "
                    f"execution of the code, aborting."
                )
            else:
                raise NotImplementedError(
                    f"Cannot deal with argument name {n} of function {f}:"
                    f"The argument kind {p.kind} is not understood (see exact definition in "
                    f"the module {Parameter}). Suggestion: your function is probably "
                    f"using non-standard arguments. Use arguments of a simpler sort "
                    f"(no kargs or kwargs). "
                    f"The full signature was: {arg_sig}"
                )
        args_hashes.append((ArgName(n), h))
    return FunctionArgContext(OrderedDict(args_hashes), None)


def get_arg_ctx_ast(
    f: Callable,  # type: ignore
    args: List[ast.AST],
    kwargs: "OrderedDict[str, ast.AST]",
) -> "OrderedDict[ArgName, Optional[PyHash]]":
    """
    Gets the arg context based on the AST.
    """
    arg_sig = inspect.signature(f)
    num_args = len(args)
    # _logger.debug(f"get_arg_ctx: {f}: arg_sig={arg_sig} args={args}")
    args_hashes: List[Tuple[ArgName, Optional[PyHash]]] = []

    def process_arg(node: ast.AST) -> Optional[PyHash]:
        # NameConstant for python 3.5 - 3.7
        if isinstance(node, (ast.Constant, ast.NameConstant)):
            # We can deal with some constant nodes
            default_ob = node.value if node.value is not None else "__none__"
            return dds_hash(default_ob)
        else:
            # Cannot deal with it for the time being
            return None

    for (idx, (n, p_)) in enumerate(arg_sig.parameters.items()):
        p: inspect.Parameter = p_
        # _logger.debug(f"get_arg_ctx: {f}: idx={idx} n={n} p={p}")
        h: Optional[PyHash]
        if p.kind not in (
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.VAR_KEYWORD,
            Parameter.VAR_POSITIONAL,
        ):
            raise NotImplementedError(
                f"Argument type not understood for function {f}: {p.kind} (see "
                f"exact definition in the module {Parameter}). Suggestion: "
                f"your function is probably using complex argument types. Use "
                f"simpler sorts of arguments (no kargs or kwargs)."
                f" The full signature was: {arg_sig}"
            )
        if idx < num_args:
            # It is a list argument
            h = process_arg(args[idx])
        else:
            if n in kwargs:
                h = process_arg(kwargs[n])
            elif p.default != Parameter.empty:
                # Argument is not provided but it has a default value
                # Use the default argument as an input
                # This assumes that the user does not mutate the argument, which is
                # a warning/errors in most linters.
                # TODO: should it discard arguments of not-whitelisted types?
                # TODO: raise a warning for non-whitelisted objects
                h = dds_hash(p.default or "__none__")
            else:
                # Do not consider this argument for the time being
                h = None
        args_hashes.append((ArgName(n), h))
    return OrderedDict(args_hashes)
