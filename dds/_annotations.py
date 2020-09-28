import functools
import pathlib
from typing import Union

from .structures import DDSPath
from ._api import keep as _keep


def dds_function(path: Union[str, DDSPath, pathlib.Path]):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _keep(path, func, *args, **kwargs)

        return wrapper

    return decorator
