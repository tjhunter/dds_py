"""
The main API functions
"""

import logging
import pathlib
import time
from collections import OrderedDict
from typing import TypeVar, Tuple, Callable, Dict, Any, Optional, Union, Set, List

from ._eval_ctx import EvalMainContext
from ._introspect_indirect import introspect_indirect
from .fun_args import get_arg_ctx
from .introspect import introspect, _accepted_packages
from .store import LocalFileStore, Store
from .structures import (
    DDSPath,
    KSException,
    EvalContext,
    PyHash,
    ProcessingStage,
)
from .structures_utils import (
    DDSPathUtils,
    FunctionInteractionsUtils,
    FunctionIndirectInteractionUtils,
)

_Out = TypeVar("_Out")
_In = TypeVar("_In")
_logger = logging.getLogger(__name__)


# TODO: set up in the use temporary space
_store: Store = LocalFileStore("/tmp/dds/internal/", "/tmp/dds/data/")
_eval_ctx: Optional[EvalContext] = None


def keep(
    path: Union[str, DDSPath, pathlib.Path],
    fun: Union[Callable[..., _Out]],
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> _Out:
    path = DDSPathUtils.create(path)
    res: Optional[_Out] = _eval(fun, path, args, kwargs, None, None, None)
    return res  # type: ignore


def eval(
    fun: Callable[[_In], _Out],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dds_export_graph: Union[str, pathlib.Path, None],
    dds_extra_debug: Optional[bool],
    dds_stages: Optional[List[Union[str, ProcessingStage]]],
) -> Optional[_Out]:
    return _eval(fun, None, args, kwargs, dds_export_graph, dds_extra_debug, dds_stages)


def load(path: Union[str, DDSPath, pathlib.Path]) -> Any:
    path_ = DDSPathUtils.create(path)
    key = _store.fetch_paths([path_]).get(path_)
    if key is None:
        raise KSException(f"The store {_store} did not return path {path_}")
    else:
        return _store.fetch_blob(key)


def set_store(
    store: Union[str, Store],
    internal_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    dbutils: Optional[Any] = None,
    commit_type: Optional[str] = None,
) -> None:
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
        dbutils = dbutils or _fetch_ipython_vars().get("dbutils")
        if dbutils is None:
            raise KSException(
                "Missing dbutils objects from input or from arguments."
                " You must be using a databricks notebook to use the DBFS store"
            )
        from .codecs.databricks import DBFSStore, CommitType

        commit_type = str(commit_type or CommitType.FULL.name).upper()
        commit_type_ = CommitType[commit_type]

        _store = DBFSStore(internal_dir, data_dir, dbutils, commit_type_)
    else:
        raise KSException(f"Unknown store {store}")
    _logger.debug(f"Setting the store to {_store}")


def _parse_stages(
    dds_stages: Optional[List[Union[str, ProcessingStage]]]
) -> List[ProcessingStage]:
    if dds_stages is None:
        return ProcessingStage.all_phases()

    def check(s: Union[str, ProcessingStage], cur: ProcessingStage) -> ProcessingStage:
        if isinstance(s, str):
            s = s.upper()
            if s not in dir(ProcessingStage):
                raise KSException(
                    f"{s} is not a valid stage name. Valid names are {dir(ProcessingStage)}"
                )
            x = ProcessingStage[s]
        elif isinstance(s, ProcessingStage):
            x = s
        else:
            raise KSException(f"Not a valid type: {s} {type(s)}")
        if x != cur:
            raise KSException(
                f"Wrong order for the stage name, expected {cur} but got {x}"
            )
        return cur

    return [check(s, cur) for (s, cur) in zip(dds_stages, ProcessingStage.all_phases())]


def _eval(
    fun: Callable[[_In], _Out],
    path: Optional[DDSPath],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dds_export_graph: Union[str, pathlib.Path, None],
    dds_extra_debug: Optional[bool],
    dds_stages: Optional[List[Union[str, ProcessingStage]]],
) -> Optional[_Out]:
    export_graph: Optional[pathlib.Path]
    if dds_export_graph is not None:
        export_graph = pathlib.Path(dds_export_graph).absolute()
    else:
        export_graph = None

    stages = _parse_stages(dds_stages)

    extra_debug = dds_extra_debug or False

    if not _eval_ctx:
        # Not in an evaluation context, create one and introspect
        return _eval_new_ctx(fun, path, args, kwargs, export_graph, extra_debug, stages)
    else:
        if not path:
            raise KSException(
                "Already in eval() context. Nested eval contexts are not supported"
            )
        key = None if path is None else _eval_ctx.requested_paths[path]
        t = _time()
        if key is not None and _store.has_blob(key):
            _logger.debug(f"_eval:Return cached {path} from {key}")
            blob = _store.fetch_blob(key)
            _add_delta(t, ProcessingStage.STORE_COMMIT)
            return blob
        else:
            _add_delta(t, ProcessingStage.STORE_COMMIT)
        arg_repr = [str(type(arg)) for arg in args]
        kwargs_repr = OrderedDict(
            [(key, str(type(arg))) for (key, arg) in kwargs.items()]
        )
        _logger.info(
            f"_eval:Evaluating (keep:{path}) fun {fun} with args {arg_repr} kwargs {kwargs_repr}"
        )
        t = _time()
        res = fun(*args, **kwargs)  # type: ignore
        _add_delta(t, ProcessingStage.STORE_COMMIT)
        _logger.info(f"_eval:Evaluating (keep:{path}) fun {fun}: completed")
        if key is not None:
            _logger.info(f"_eval:Storing blob into key {key}")
            t = _time()
            _store.store_blob(key, res)
            _add_delta(t, ProcessingStage.STORE_COMMIT)
        return res


def _time() -> float:
    return time.monotonic()


def _add_delta(start_t: float, stage: ProcessingStage) -> None:
    global _eval_ctx
    if _eval_ctx is None:
        return
    _eval_ctx.stats_time[stage] += _time() - start_t


def _eval_new_ctx(
    fun: Callable[[_In], _Out],
    path: Optional[DDSPath],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    export_graph: Optional[pathlib.Path],
    extra_debug: bool,
    stages: List[ProcessingStage],
) -> Optional[_Out]:
    global _eval_ctx
    assert _eval_ctx is None, _eval_ctx
    _eval_ctx = EvalContext(
        requested_paths={},
        stats_time=dict([(stage, 0.0) for stage in ProcessingStage.all_phases()]),
    )
    try:
        t = _time()
        # Fetch the local vars from the call. This is required if running from an old IPython context
        # (like databricks for instance)
        local_vars = _fetch_ipython_vars()
        _logger.debug(f"_eval_new_ctx: local_vars: {sorted(local_vars.keys())}")
        arg_ctx = get_arg_ctx(fun, args, kwargs)
        _logger.debug(f"arg_ctx: {arg_ctx}")
        eval_ctx = EvalMainContext(
            fun.__module__,  # type: ignore
            whitelisted_packages=_accepted_packages,
            start_globals=local_vars,
            resolved_references=OrderedDict(),
        )
        _logger.debug(f"_eval_new_ctx: introspect_indirect completed")
        inters_indirect = introspect_indirect(fun, eval_ctx)
        all_loads = FunctionIndirectInteractionUtils.all_loads(inters_indirect)
        all_stores = FunctionIndirectInteractionUtils.all_stores(inters_indirect)
        loads_to_check = sorted([p for p in all_loads if p not in all_stores])
        # Check that there are no indirect references to resolve:
        if loads_to_check:
            _logger.debug(
                f"_eval_new_ctx: need to resolve indirect references: {loads_to_check}"
            )
            resolved_indirect_refs = _store.fetch_paths(loads_to_check)
            _logger.debug(
                f"_eval_new_ctx: fetched indirect references: {resolved_indirect_refs}"
            )
        else:
            resolved_indirect_refs = OrderedDict()

        # Make a copy of the dictionary: the context will use it to track all the nodes, which means that
        # at plotting stage, it will contain all the nodes, not just the indirect refs.
        eval_ctx.resolved_references = OrderedDict(resolved_indirect_refs)

        inters = introspect(fun, eval_ctx, arg_ctx)
        _logger.debug(f"_eval_new_ctx: introspect completed")
        # Also add the current path, if requested:
        if path is not None:
            inters = inters._replace(store_path=path)
        store_paths = FunctionInteractionsUtils.all_store_paths(inters)
        _logger.debug(
            f"_eval_new_ctx: assigning {len(store_paths)} store path(s) to context"
        )
        _eval_ctx = _eval_ctx._replace(requested_paths=store_paths)
        present_blobs: Optional[Set[PyHash]]
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

            draw_graph(inters, export_graph, present_blobs, resolved_indirect_refs)
            _logger.debug(f"_eval_new_ctx: draw_graph_completed")

        _logger.debug(f"Stage {ProcessingStage.ANALYSIS} completed")
        _add_delta(t, ProcessingStage.ANALYSIS)
        if ProcessingStage.EVAL not in stages:
            _logger.debug("Stopping here")
            return None

        for (p, key) in store_paths.items():
            _logger.debug(f"Updating path: {p} -> {key}")

        # If the blob for that node already exists, we have computed the path already.
        # We only need to check if the path is committed to the blob
        current_sig = inters.fun_return_sig
        _logger.debug(f"_eval_new_ctx:current_sig: {current_sig}")
        t = _time()
        if _store.has_blob(current_sig):
            _logger.debug(f"_eval_new_ctx:Return cached signature {current_sig}")
            res = _store.fetch_blob(current_sig)
            _add_delta(t, ProcessingStage.STORE_COMMIT)
        else:
            arg_repr = [str(type(arg)) for arg in args]
            kwargs_repr = OrderedDict(
                [(key, str(type(arg))) for (key, arg) in kwargs.items()]
            )
            _logger.info(
                f"_eval_new_ctx:Evaluating (eval) fun {fun} with args {arg_repr} kwargs {kwargs_repr}"
            )
            res = fun(*args, **kwargs)  # type: ignore
            _add_delta(t, ProcessingStage.EVAL)
            _logger.info(f"_eval_new_ctx:Evaluating (eval) fun {fun}: completed")
            obj_key: Optional[
                PyHash
            ] = None if path is None else _eval_ctx.requested_paths[path]
            if obj_key is not None:
                # TODO: add a phase for storing the blobs
                _logger.info(f"_eval:Storing blob into key {obj_key}")
                t = _time()
                _store.store_blob(obj_key, res)
                _add_delta(t, ProcessingStage.STORE_COMMIT)

        if ProcessingStage.PATH_COMMIT in stages:
            _logger.debug(f"Starting stage {ProcessingStage.PATH_COMMIT}")
            t = _time()
            _store.sync_paths(store_paths)
            _add_delta(t, ProcessingStage.PATH_COMMIT)
            _logger.debug(f"Stage {ProcessingStage.PATH_COMMIT} done")
        else:
            _logger.info(f"Skipping stage {ProcessingStage.PATH_COMMIT}")
        return res
    finally:
        # Cleaning up the context
        s = (
            sum([_eval_ctx.stats_time[stage] for stage in ProcessingStage.all_phases()])
            + 1e-10
        )
        for stage in ProcessingStage.all_phases():
            x = _eval_ctx.stats_time[stage]
            _logger.info(f"Stage {stage}: {x:.3f} sec {100 * x / s:.2f}%")
        _eval_ctx = None


def _fetch_ipython_vars() -> Dict[str, Any]:
    """
    Fetches variables from the ipython / jupyter environment. This is a best effort method.
    """
    try:
        from IPython import get_ipython  # type: ignore

        ipython = get_ipython()
        if ipython is None:
            return {}
        return dict(ipython.user_ns)
    except ImportError:
        _logger.debug(
            "Failed to import IPython. No jupyter/ipython variables will be logged"
        )
        return {}
