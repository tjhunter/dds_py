import functools
import pathlib
from typing import Any, Callable, TypeVar, cast, overload, Type
from typing import Union

from ._api import keep as _keep
from .structures import DDSPath

F = TypeVar("F", bound=Callable[..., Any])
FF = TypeVar("FF", bound=Callable[..., Callable[..., Any]])

T_CallableOrType = TypeVar("T_CallableOrType", Callable[..., Any], Type[Any])

# @overload
# def dds_function(path: Union[str, DDSPath, pathlib.Path]) -> Callable[[T_CallableOrType], T_CallableOrType]:
#     ...

# @overload
def dds_function(
    path: Union[str, DDSPath, pathlib.Path]
) -> Callable[[T_CallableOrType], T_CallableOrType]:
    def decorator_(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _keep(path, func, *args, **kwargs)

        return cast(F, wrapper)

    return decorator_
    # return cast(FF, decorator_)
