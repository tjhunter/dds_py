import functools
import pathlib
from typing import Any, Callable, TypeVar, cast
from typing import Union

from ._api import keep as _keep
from .structures import DDSPath

F = TypeVar("F", bound=Callable[..., Any])
FF = TypeVar("FF", bound=Callable[..., Callable[..., Any]])


def dds_function(path: Union[str, DDSPath, pathlib.Path]) -> FF:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _keep(path, func, *args, **kwargs)

        return cast(F, wrapper)

    return cast(FF, decorator)
