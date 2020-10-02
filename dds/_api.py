"""
The main API functions
"""

import logging
import pathlib
from collections import OrderedDict
from typing import TypeVar, Tuple, Callable, Dict, Any, Optional, Union

from .fun_args import get_arg_ctx
from .introspect import introspect
from .store import LocalFileStore, Store
from .structures import DDSPath, KSException, EvalContext
from .structures_utils import DDSPathUtils, FunctionInteractionsUtils

_Out = TypeVar("_Out")
_In = TypeVar("_In")
_logger = logging.getLogger(__name__)


# TODO: set up in the use temporary space
_store: Store = LocalFileStore("/tmp/dds/internal/", "/tmp/dds/data/")
_eval_ctx: Optional[EvalContext] = None


def keep(
    path: Union[str, DDSPath, pathlib.Path],
    fun: Union[Callable[..., _Out]],
    *args,
    **kwargs,
) -> _Out:
    path = DDSPathUtils.create(path)
    return _eval(fun, path, args, kwargs, None, None)


def eval(
    fun: Callable[[_In], _Out],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dds_export_graph: Union[str, pathlib.Path, None],
    dds_extra_debug: Optional[bool],
) -> _Out:
    return _eval(fun, None, args, kwargs, dds_export_graph, dds_extra_debug)


def set_store(
    store: Union[str, Store],
    internal_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    dbutils: Optional[Any] = None,
    commit_type: Optional[str] = None,
):
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
        from .codecs.databricks import DBFSStore, CommitType

        commit_type = str(commit_type or CommitType.FULL.name).upper()
        commit_type_ = CommitType[commit_type]

        _store = DBFSStore(internal_dir, data_dir, dbutils, commit_type_)
    else:
        raise KSException(f"Unknown store {store}")
    _logger.debug(f"Setting the store to {_store}")


def _eval(
    fun: Callable[[_In], _Out],
    path: Optional[DDSPath],
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    dds_export_graph: Union[str, pathlib.Path, type(None)],
    dds_extra_debug: Optional[bool],
) -> _Out:
    if dds_export_graph is not None:
        export_graph = pathlib.Path(dds_export_graph).absolute()
    else:
        export_graph = None

    extra_debug = dds_extra_debug or False

    if not _eval_ctx:
        # Not in an evaluation context, create one and introspect
        return _eval_new_ctx(fun, path, args, kwargs, export_graph, extra_debug)
    else:
        if not path:
            raise KSException(
                "Already in eval() context. Nested eval contexts are not supported"
            )
        key = None if path is None else _eval_ctx.requested_paths[path]
        if key is not None and _store.has_blob(key):
            _logger.debug(f"_eval:Return cached {path} from {key}")
            return _store.fetch_blob(key)
        arg_repr = [str(type(arg)) for arg in args]
        kwargs_repr = OrderedDict(
            [(key, str(type(arg))) for (key, arg) in kwargs.items()]
        )
        _logger.info(
            f"_eval:Evaluating (keep:{path}) fun {fun} with args {arg_repr} kwargs {kwargs_repr}"
        )
        res = fun(*args, **kwargs)
        _logger.info(f"_eval:Evaluating (keep:{path}) fun {fun}: completed")
        if key is not None:
            _logger.info(f"_eval:Storing blob into key {key}")
            _store.store_blob(key, res)
        return res


def _eval_new_ctx(
    fun: Callable[[_In], _Out],
    path: Optional[DDSPath],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    export_graph: Optional[pathlib.Path],
    extra_debug: bool,
) -> _Out:
    global _eval_ctx
    assert _eval_ctx is None, _eval_ctx
    _eval_ctx = EvalContext(requested_paths={})
    try:
        # Fetch the local vars from the call. This is required if running from an old IPython context
        # (like databricks for instance)
        local_vars = _fetch_ipython_vars()
        _logger.debug(f"_eval_new_ctx: local_vars: {sorted(local_vars.keys())}")
        arg_ctx = get_arg_ctx(fun, args, kwargs)
        _logger.debug(f"arg_ctx: {arg_ctx}")
        inters = introspect(fun, local_vars, arg_ctx)
        _logger.debug(f"_eval_new_ctx: introspect completed")
        # Also add the current path, if requested:
        if path is not None:
            inters = inters._replace(store_path=path)
        store_paths = FunctionInteractionsUtils.all_store_paths(inters)
        _logger.debug(
            f"_eval_new_ctx: assigning {len(store_paths)} store path(s) to context"
        )
        _eval_ctx = EvalContext(requested_paths=store_paths)
        if extra_debug:
            present_blobs = set(
                [key for key in set(store_paths.values()) if _store.has_blob(key)]
            )
            _logger.debug(f"_eval_new_ctx: {len(present_blobs)} present blobs")
        else:
            present_blobs = None

        _logger.info(f"Interaction tree:")
        FunctionInteractionsUtils.pprint_tree(
            inters, present_blobs, printer=lambda s: _logger.info(s)
        )
        if export_graph is not None:
            # Attempt to run the export module:
            from ._plotting import draw_graph

            draw_graph(inters, export_graph, present_blobs)
            _logger.debug(f"_eval_new_ctx: draw_graph_completed")

        for (p, key) in store_paths.items():
            _logger.debug(f"Updating path: {p} -> {key}")

        # If the blob for that node already exists, we have computed the path already.
        # We only need to check if the path is committed to the blob
        current_sig = inters.fun_return_sig
        _logger.debug(f"_eval_new_ctx:current_sig: {current_sig}")
        if _store.has_blob(current_sig):
            _logger.debug(f"_eval_new_ctx:Return cached signature {current_sig}")
            res = _store.fetch_blob(current_sig)
        else:
            arg_repr = [str(type(arg)) for arg in args]
            kwargs_repr = OrderedDict(
                [(key, str(type(arg))) for (key, arg) in kwargs.items()]
            )
            _logger.info(
                f"_eval_new_ctx:Evaluating (eval) fun {fun} with args {arg_repr} kwargs {kwargs_repr}"
            )
            res = fun(*args, **kwargs)
            _logger.info(f"_eval_new_ctx:Evaluating (eval) fun {fun}: completed")
            key = None if path is None else _eval_ctx.requested_paths[path]
            if key is not None:
                _logger.info(f"_eval:Storing blob into key {key}")
                _store.store_blob(key, res)
        _store.sync_paths(store_paths)
        return res
    finally:
        # Cleaning up the context
        _eval_ctx = None


def _fetch_ipython_vars() -> Dict[str, Any]:
    """
    Fetches variables from the ipython / jupyter environment. This is a best effort method.
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return {}
        return dict(ipython.user_ns)
    except ImportError:
        _logger.debug(
            "Failed to import IPython. No jupyter/ipython variables will be logged"
        )
        return {}
