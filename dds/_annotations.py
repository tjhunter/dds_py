import functools
import pathlib
import warnings
from typing import Any, Callable, TypeVar, cast, Union

from ._api import keep as _keep
from .structures import DDSPath, DDSErrorCode, DDSException

F = TypeVar("F", bound=Callable[..., Any])


def dds_function(path: Union[str, DDSPath, pathlib.Path]) -> Callable[[F], F]:
    """
    Annotation-style for `dds.keep`.

    DEPRECATED. Use data_function instead. 'data_function' provides the
    same functionality but has a more understandable name.
    """
    warnings.warn(
        "The name 'dds_function' is deprecated. Use 'data_function' instead. ",
        DeprecationWarning,
    )

    def decorator_(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _keep(path, func, *args, **kwargs)

        return cast(F, wrapper)

    return decorator_


def data_function(path: Union[str, DDSPath, pathlib.Path]) -> Callable[[F], F]:
    """
    Annotation-style for `dds.keep`.

    This is useful for functions with no arguments that should be cached as DDS functions.

    The following definitions are equivalent:

    ```py
    dds.data_function("/function")
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
            if len(args) > 0 or len(kwargs) > 0:
                raise DDSException(
                    f"@data_function cannot be used with arguments. "
                    f"Arguments were passed to the function {func}, but this function "
                    f"also has a dds.data_function annotation, which is not allowed (see "
                    f"user guide of DDS). "
                    f"Suggestion: write a wrapper function that does not take arguments itself, "
                    f"or use dds.keep to pass arguments",
                    DDSErrorCode.ARG_IN_DATA_FUNCTION,
                )
            return _keep(path, func, *args, **kwargs)

        return cast(F, wrapper)

    return decorator_
