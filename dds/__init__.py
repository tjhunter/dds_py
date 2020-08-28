import logging
from typing import TypeVar, Callable, Any, NewType, NamedTuple, OrderedDict, FrozenSet, Optional
import inspect
import ast

from .structures import Path, PyHash, FunctionInteractions, KSException
from .introspect import introspect
from .store import LocalFileStore, Store

_T = TypeVar("T")
_In = TypeVar("In")
_logger = logging.getLogger(__name__)


__all__ = ["Path", "keep", "load", "cache", "eval"]


_store: Store = LocalFileStore("/tmp", "/tmp/data/")
_context_set: bool = False


def keep(path: Path, fun: Callable[[_In], _T], *args, **kwargs) -> _T:
    if not _context_set:
        raise NotImplementedError("Must call eval for now")
    key = _store.get_path(path)
    assert key is not None, (path, fun)
    if _store.has_blob(key):
        _logger.debug(f"Restoring path {path} -> {key}")
        return _store.fetch_blob(key)
    _logger.info(f"Evaluating (keep:{path}) fun {fun} with args {args} kwargs {kwargs}")
    res = fun(*args, **kwargs)
    _store.store_blob(key, res)
    return res


def load(path: Path) -> Any:
    _logger.info(f"Performing load from {path}")
    return None


def cache(fun: Callable[[_In], _T], *args, **kwargs) -> _T:
    res = fun(*args, **kwargs)
    _logger.info(f"Evaluating (cache) fun {fun} with args {args} kwargs {kwargs} -> {res}")
    return res


def eval(fun: Callable[[_In], _T], *args, **kwargs) -> _T:
    global _context_set
    if _context_set:
        # TODO more info
        raise KSException("Already in eval() context")
    _context_set = True
    try:
        # TODO: pass args too
        inters = introspect(fun)
        for (p, key) in inters.outputs:
            _logger.debug(f"Updating path: {p} -> {key}")
            # The key may not be computed yet
            _store.register(p, key)
        _logger.info(f"fun {fun}: {inters}")
        res = fun(*args, **kwargs)
        _logger.info(f"Evaluating (eval) fun {fun} with args {args} kwargs {kwargs} -> {res}")
        _store.sync_paths()
        return res
    finally:
        _context_set = False

