import pathlib
from types import ModuleType
from typing import Tuple, Dict
from typing import TypeVar, Callable, Any, Optional, Union, List

from ._annotations import dds_function
from ._api import (
    keep as _keep,
    eval as _eval,
    set_store as _set_store,
)
from ._version import version
from .introspect import whitelist_module as _whitelist_module
from .store import Store
from .structures import DDSPath, ProcessingStage

__all__ = [
    "DDSPath",
    "keep",
    "eval",
    "whitelist_module",
    "set_store",
    "__version__",
    "dds_function",
]

__version__ = version

_Out = TypeVar("_Out")
_In = TypeVar("_In")


def keep(
    path: Union[str, DDSPath, pathlib.Path],
    fun: Callable[..., _Out],
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any]
) -> _Out:
    """
    Stores the result of calling a function to a specific path. If this particular evaluation has not happened before,
    the function will be evaluated again (with the given arguments).

    For example, instead of writing:

    >>> data = my_function(arg1, arg2)

    you should use:

    >>> data = dds.keep(path, my_function, arg1, arg2)

    Accepted functions.
    ===================

    In general, the functions that should be provided are total, non-recursive, deterministic and referentially
     transparent.

    Functions have currently the following restrictions:
    - no static method, class method
    - not recursive
    - no generators
    - the functions must be in a whitelisted module to be considered, see the whitelist_module() function

    They must return storable objects. The exact list depends on the store that is currently deployed.

    Accepted arguments.
    -------------------

    Only the following classes of arguments are accepted:
    - the basic types of python (int, str, bool, float)
    - lists and tuples of accepted arguments
    - dictionaries. They are evaluated as sorted lists (by their keys)

    Using complex arguments
    -----------------------

    If more complex arguments should be accepted, two strategies are possible:
    - embed them inside the function call
    - wrap them inside a function which is then called through eval()

    Example: the following code will fail:

    >>> df = pd.DataFrame({"x":[1]})
    >>>
    >>> def my_stats(data: pd.DataFrame) -> int: return len(data)
    >>>
    >>> stats = dds.keep("/stats", my_stats, data)

    The workaround is to use a wrapper function that create the dataframe:

    >>> def my_pipeline():
    >>>     df = pd.DataFrame({"x":[1]})
    >>>     return dds.keep("/stats", my_stats, df) # This will work
    >>>
    >>> stats = dds.eval(my_pipeline)

    Another possibility is to move keep the result of the pipeline instead:

    >>> def my_pipeline2():
    >>>     df = pd.DataFrame({"x":[1]})
    >>>     return my_stats(df)
    >>>
    >>> stats = dds.keep("/stats", my_pipeline2)

    The difference is that my_pipeline will be evaluated at each call (but not my_stats), while my_pipeline2 will
    only be evaluated once.

    :param path: a path in the storage system which will store the content of the function, evaluated to the given
    arguments. It is expected to be in absolute form (starting with "/" if a string
    or being an absolute path if a pathlib's Path object).
    If this path exists, it will be overwritten silently.

    :param fun: A function to evaluate. See text above on the limitations over this function
    :param args: the arguments of this function
    :param kwargs: (keyworded arguments are currently unsupported)
    :return: the value that the function would produce for these arguments
    """
    return _keep(path, fun, *args, **kwargs)


def eval(
    fun: Callable[..., _Out],
    *args: Tuple[Any, ...],
    dds_export_graph: Union[str, pathlib.Path, None] = None,
    dds_extra_debug: Optional[bool] = None,
    dds_stages: Optional[List[Union[str, ProcessingStage]]] = None,
    **kwargs: Dict[str, Any]
) -> Optional[_Out]:
    """
    Evaluates a function. The result of the function is not stored in the data store, but the function itself may
    contain multiple calls to keep().

    This function is useful to consider in one evaluation multiple functions that may themselves call keep() or
    refer to each other. eval() allows keep() calls to refer to complex arguments that cannot be evaluated before
    runtime. eval() als ensures some basic rules such as no circular references, and will eventually
    enable automatic parallel execution of internal statements.

    See also the documentation of keep() for an example of eval().

    :param dds_export_graph: if specified, a file with the dependency graph of the function will be exported.
     NOTE: this requires the pydot or pydotplus package to be installed, as well as the graphviz program.
     These packages must be installed separately. If they are not present, a runtime error will be triggered.

    :param dds_extra_debug: (default false). If true, will compute extra statistics to assist with debugging.
      As implemented, it reaches to the store to check which blobs need to be computed.


    Simple example.

    >>> def f1_internal(): return 1
    >>>
    >>> def f1(): return dds.keep("/f1_value", f1_internal)
    >>>
    >>> def f2(): return 2 + f1()
    >>>
    >>> def all_fs():
    >>>     f1()
    >>>     dds.keep("/f2_value", f2)
    In this example, the function is evaluated only once and its result is served to f1 and f2.

    :param fun: a function. See eval() for restrictions.
    :param args: arguments. See eval() for restrictions.
    :param kwargs: (keyworded arguments are currently unsupported)
    :return: the return value of the function.
    """
    return _eval(fun, args, kwargs, dds_export_graph, dds_extra_debug, dds_stages)


def set_store(
    store: Union[str, Store],
    internal_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    dbutils: Optional[Any] = None,
    commit_type: Optional[str] = None,
) -> None:
    """
    Sets a new store.

    By default, a local store pointing to /tmp/ is created.
    TODO: change this to the user directory.

    :param store: a type of store. Currently 'local' (local hard drive) or 'dbfs' (databricks environment) are supported
    :param internal_dir: a path in a filesystem for the internal storage. The internal storage contains evaluated blobs
    and corresponding metadata.
        - for local: a path in the local filesystem
        - for dbfs: a path in DBFS
    :param data_dir: a path in a filesystem for the data storage. All paths provided by the user are guaranteed to map
    to a path in the given data storage.
        - for local: a path in the local filesystem. It contains symbolic links to the internal storage
        - for dbfs: a path in DBFS. Objects are copied from the internal storage to the data storage
    :param dbutils: (required only for the 'dbfs' store) the dbutils object that is in a notebook
    :param commit_type: (DBFS only, 'none', 'links_only', 'full', default 'full'). The type of commit that will be
        executed to update the paths. Committing a path in DBFS involves a full copy, which may be expensive,
        especially if the underlying table uses Databricks Delta. This is why the following options can be used:
        - none: no file will be committed
        - links_only: a metadata link reference to the blob will be updated. This is much faster because it involves
            a 1kb file transfer as opposed to a full copy. This means though that the result will not be readable
            outside of a DDS store, because the file is not fully materialized.

    :return: nothing
    """
    _set_store(store, internal_dir, data_dir, dbutils, commit_type)


def whitelist_module(module: Union[str, ModuleType]) -> None:
    """
    Marks a module as whitelisted for introspection.

    Example to ensure that all the functions in my_lib are considered by DDS.

    >>> import my_lib
    >>> dds.whitelist_module(my_lib)

    alternative approach if you do not want to import the my_lib module:

    >>> dds.whitelist_module("my_lib")

    """
    return _whitelist_module(module)
