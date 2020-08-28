import logging
from typing import TypeVar, Callable, Any, Optional
from collections import OrderedDict

from .introspect import introspect
from .store import LocalFileStore, Store
from .structures import Path, PyHash, FunctionInteractions, KSException, EvalContext

_T = TypeVar("T")
_In = TypeVar("In")
_logger = logging.getLogger(__name__)

__all__ = ["Path", "keep", "load", "cache", "eval"]

_store: Store = LocalFileStore("/tmp", "/tmp/data/")
_eval_ctx: Optional[EvalContext] = None


def keep(path: Path, fun: Callable[[_In], _T], *args, **kwargs) -> _T:
    if not _eval_ctx:
        raise NotImplementedError("Must call eval for now")
    key = _eval_ctx.requested_paths[path]
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
    global _eval_ctx
    if _eval_ctx:
        # TODO more info
        raise KSException("Already in eval() context")
    _eval_ctx = EvalContext(requested_paths={})
    try:
        # TODO: pass args too
        inters = introspect(fun)
        _eval_ctx = EvalContext(requested_paths=dict(inters.outputs))
        for (p, key) in inters.outputs:
            _logger.debug(f"Updating path: {p} -> {key}")
        _logger.info(f"fun {fun}: {inters}")
        res = fun(*args, **kwargs)
        _logger.info(f"Evaluating (eval) fun {fun} with args {args} kwargs {kwargs} -> {res}")
        _store.sync_paths(OrderedDict(inters.outputs))
        return res
    finally:
        _eval_ctx = False
