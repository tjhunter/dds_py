import functools
import pathlib
from typing import Any, Callable, TypeVar, cast, Type, Union

from ._api import keep as _keep
from .structures import DDSPath

F = TypeVar("F", bound=Callable[..., Any])
T_CallableOrType = TypeVar("T_CallableOrType", Callable[..., Any], Type[Any])


def dds_function(
    path: Union[str, DDSPath, pathlib.Path]
) -> Callable[[T_CallableOrType], T_CallableOrType]:
    """
    Annotation-style for `dds.keep`.

    This is useful for functions with no arguments that should be cached as DDS functions.

    The following definitions are equivalent:

    ```py
    dds.dds_function("/function")
    def function(): return 1
    ```

    ```py
    def _function(): return 1

    def function():
        return dds.keep("/function", _function)
    ```
    """

    def decorator_(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _keep(path, func, *args, **kwargs)

        return cast(F, wrapper)

    return decorator_
