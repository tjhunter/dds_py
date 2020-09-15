import logging
from typing import TypeVar, Callable, Any, Optional, Union
from collections import OrderedDict
import inspect

from .introspect import introspect, whitelist_module
from .store import LocalFileStore, Store
from .structures import DDSPath, PyHash, FunctionInteractions, KSException, EvalContext
from .fun_args import get_arg_ctx

_Out = TypeVar("_Out")
_In = TypeVar("_In")
_logger = logging.getLogger(__name__)

__all__ = ["DDSPath", "keep", "load", "cache", "eval", "whitelist_module", "set_store"]

# TODO: set up in the use temporary space
_store: Store = LocalFileStore("/tmp", "/tmp/data/")
_eval_ctx: Optional[EvalContext] = None


def keep(path: Union[str, DDSPath], fun: Callable[[_In], _Out], *args, **kwargs) -> _Out:
    # TODO: clean and validate the path here
    if not _eval_ctx:
        raise NotImplementedError("Must call eval for now")
    key = _eval_ctx.requested_paths[path]
    assert key is not None, (path, fun)
    if _store.has_blob(key):
        _logger.debug(f"Restoring path {path} from {key}")
        return _store.fetch_blob(key)
    _logger.info(f"Evaluating (keep:{path}) fun {fun} with args {args} kwargs {kwargs}")
    res = fun(*args, **kwargs)
    _store.store_blob(key, res)
    return res


def load(path: DDSPath) -> Any:
    raise NotImplementedError()
    # _logger.info(f"Performing load from {path}")
    # return None


def cache(fun: Callable[[_In], _Out], *args, **kwargs) -> _Out:
    raise NotImplementedError()
    # res = fun(*args, **kwargs)
    # _logger.info(f"Evaluating (cache) fun {fun} with args {args} kwargs {kwargs} -> {res}")
    # return res


def eval(fun: Callable[[_In], _Out], *args, **kwargs) -> _Out:
    """
    Evaluates a function that may cache data, without caching the result
    of the function itself.
    :param fun: the function
    :param args: arguments
    :param kwargs: argument
    :return: the return value of the function
    """
    global _eval_ctx
    if _eval_ctx:
        # TODO more info
        raise KSException("Already in eval() context")
    _eval_ctx = EvalContext(requested_paths={})
    try:
        # TODO: pass args too
        # Fetch the local vars from the call. This is required if running from an old IPython context
        # (like databricks for instance)
        # TODO: detect if we are running in databricks to account for this hack.
        local_vars = inspect.currentframe().f_back.f_locals
        _logger.debug(f"locals: {sorted(local_vars.keys())}")
        # TODO: use the arg context in the call
        arg_ctx = get_arg_ctx(fun, args, kwargs)
        _logger.debug(f"arg_ctx: {arg_ctx}")
        inters = introspect(fun, local_vars)
        _logger.info(f"Interaction tree:")
        FunctionInteractions.pprint_tree(inters, printer=lambda s: _logger.info(s))
        store_paths = FunctionInteractions.all_store_paths(inters)
        _eval_ctx = EvalContext(requested_paths=store_paths)
        for (p, key) in store_paths.items():
            _logger.debug(f"Updating path: {p} -> {key}")
        res = fun(*args, **kwargs)
        _logger.info(f"Evaluating (eval) fun {fun} with args {args} kwargs {kwargs}")
        _store.sync_paths(store_paths)
        return res
    finally:
        _eval_ctx = False


def set_store(
        store: Union[str, Store],
        internal_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        dbutils: Optional[Any] = None):
    """
    Sets the store for the execution of the program.

    store: either a store, or 'local' or 'dbfs'
    """
    global _store
    if isinstance(store, Store):
        # Directly setting the store
        _store = store
        return
    elif store == "local":
        if not internal_dir:
            internal_dir = "/tmp"
        if not data_dir:
            data_dir = "/tmp/data"
        _store = LocalFileStore(internal_dir, data_dir)
        return
    elif store == "dbfs":
        if data_dir is None:
            raise KSException("Missing data_dir argument")
        if internal_dir is None:
            raise KSException("Missing internal_dir argument")
        if dbutils is None:
            raise KSException("Missing dbutils argument")
        from .codecs.databricks import DBFSStore
        _store = DBFSStore(internal_dir, data_dir, dbutils)
    else:
        raise KSException(f"Unknown store {store}")
    _logger.debug(f"Setting the store to {_store}")

