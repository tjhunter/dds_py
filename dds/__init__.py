from typing import TypeVar, Tuple, Callable, Dict, Any, Optional, Union, List

from .structures import DDSPath
from ._api import (
    keep as _keep,
    eval as _eval,
    set_store as _set_store,
)
from .introspect import whitelist_module as _whitelist_module
from .store import Store
from types import ModuleType
from typing import Tuple, Dict


__all__ = ["DDSPath", "keep", "eval", "whitelist_module", "set_store"]


_Out = TypeVar("_Out")
_In = TypeVar("_In")


def keep(
    path: Union[str, DDSPath],
    fun: Union[Callable[[_In], _Out], Callable[[], _Out]],
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any]
) -> _Out:
    return _keep(path, fun, *args, **kwargs)


def eval(
    fun: Callable[[_In], _Out], *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
) -> _Out:
    return _eval(fun, *args, **kwargs)


def set_store(
    store: Union[str, Store],
    internal_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    dbutils: Optional[Any] = None,
) -> None:
    _set_store(store, internal_dir, data_dir, dbutils)


def whitelist_module(module: Union[str, ModuleType]) -> None:
    return _whitelist_module(module)
