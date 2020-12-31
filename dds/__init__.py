import pathlib
import warnings
from types import ModuleType
from typing import Tuple, Dict
from typing import TypeVar, Callable, Any, Optional, Union, List

from ._annotations import dds_function, data_function
from ._api import (
    keep as _keep,
    eval as _eval,
    load as _load,
    set_store as _set_store,
)
from ._version import version
from .introspect import accept_module as _accept_module
from .store import Store
from .structures import DDSPath, ProcessingStage

__all__ = [
    "DDSPath",
    "keep",
    "eval",
    "whitelist_module",
    "accept_module",
    "set_store",
    "__version__",
    "dds_function",
    "data_function",
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

    ```py
    data = my_function(arg1, arg2)
    ```

    you should use:

    ```py
    data = dds.keep(path, my_function, arg1, arg2)
    ```

    Arguments:
        path: a path in the storage system which will store the content of the function, evaluated to the given
            arguments. It is expected to be in absolute form (starting with "/" if a string
            or being an absolute path if a pathlib's `Path` object).

            If this path exists, it will be overwritten silently.

        fun: A function to evaluate. See text above on the limitations over this function

        args: the arguments of this function

        kwargs: *(keyworded arguments are currently unsupported)*

    return: the value that the function would produce for these arguments


    ### Accepted functions.

    In general, the functions that should be provided are *total, non-recursive, deterministic and referentially
     transparent*.

    Functions have currently the following restrictions:

    - no static method, class method
    - not recursive
    - no generators
    - the functions must be in an accepted module to be considered, see the `accept_module()` function

    They must return storable objects. The exact list depends on the store that is currently deployed.

    ### Accepted arguments.

    Only the following classes of arguments are accepted:

    - the basic types of python (int, str, bool, float)
    - lists and tuples of accepted arguments
    - dictionaries. They are evaluated as sorted lists (by their keys)

    ### Using complex arguments

    If more complex arguments should be accepted, two strategies are possible:
    - embed them inside the function call
    - wrap them inside a function which is then called through eval()

    Example: the following code will fail:
    
    ```py
    df = pd.DataFrame({"x":[1]})
    
    def my_stats(data: pd.DataFrame) -> int: return len(data)
    
    stats = dds.keep("/stats", my_stats, data)
    ```

    The workaround is to use a wrapper function that create the dataframe:
    
    ```py
    def my_pipeline():
        df = pd.DataFrame({"x":[1]})
        return dds.keep("/stats", my_stats, df) # This will work
    
    stats = dds.eval(my_pipeline)
    ```

    Another possibility is to move keep the result of the pipeline instead:
    
    ```py
    def my_pipeline2():
        df = pd.DataFrame({"x":[1]})
        return my_stats(df)

    stats = dds.keep("/stats", my_pipeline2)
    ```


    The difference is that my_pipeline will be evaluated at each call (but not my_stats), while my_pipeline2 will
    only be evaluated once.

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

    Arguments:
      fun: the function to call.

      args: the optional arguments for this function.

        NOTE: keyworded arguments are not supported yet.

      dds_extra_debug: If true, will compute extra statistics to assist with debugging.
        As implemented, it reaches to the store to check which blobs need to be computed.

      dds_export_graph: if specified, a file with the dependency graph of the function will be exported.

        NOTE: this requires the pydot or pydotplus package to be installed, as well as the graphviz program.
        These packages must be installed separately. If they are not present, a runtime error will be triggered.

    Simple example.

    In this example, the function is evaluated only once and its result is served to f1 and f2.

    ```py
    def f1_internal(): return 1
    
    def f1(): return dds.keep("/f1_value", f1_internal)
    
    def f2(): return 2 + f1()
    
    def all_fs():
        f1()
        dds.keep("/f2_value", f2)
    ```

    """
    return _eval(fun, args, kwargs, dds_export_graph, dds_extra_debug, dds_stages)


def load(path: Union[str, DDSPath, pathlib.Path]) -> Any:
    """
    Loads the content of an object that has already been stored.

    This command is useful to refer to an object by its *path* instead of using the data function that was used
    to generate it.

    For example, if an object was created with a call to the *keep* function, then *load* can be used
    to retrieve an object later:

    ```py
    _ = dds.keep("/my_path", data_function)
    data = dds.load("/my_path")
    ```

    This function can be used standalone or within an evaluation (i.e. through using *eval()*). In that case,
    some invariants will be checked:
    - loops are not allowed: it is not allowed to both load and output the same path within the same evaluation
    - the signature of the latest version of the loaded data is included in the signature calculation. If the data
    accessed through *load* changes, this will change the signature of the function and may retrigger some evaluations.

    This is useful in the case the data function that generated the data artifact in the first place is not accessible,
    for security reasons for example. The data artifact can still be loaded with the latest version, but it may not
    be up to date with the data function. See the user guide of dds for a more complete presentation on when
    to use *keep*.
    """
    return _load(path)


def set_store(
    store: Union[str, Store],
    internal_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    dbutils: Optional[Any] = None,
    commit_type: Optional[str] = None,
) -> None:
    """
    Sets a new store or replaces the existing store.

    By default, a local store is created, pointing to the
    temporary directory of the current operating system:

      - /tmp/dds/data for the user data
      - /tmp/dds/internal for the blobs

    The exact paths of the default store may change in the future.

    Arguments:

      store: a type of store. Two values are supported currently:

        - `local`: local file system
        - `dbfs`: the Databricks file system (only valid for the Databricks environment)

      internal_dir:  a path in a filesystem for the internal storage. The internal storage contains evaluated blobs
          and corresponding metadata. Accepted values are:

          - local: a path in the local filesystem
          - dbfs: a path in DBFS

      data_dir: a path in a filesystem for the data storage. All paths provided by the user are guaranteed to map
           to a path in the given data storage.

        - for local: a path in the local filesystem. It contains symbolic links to the internal storage
        - for dbfs: a path in DBFS. Objects are copied from the internal storage to the data storage

      dbutils: (optional, valid only for the 'dbfs' store) the `dbutils` object that is in a notebook. If not provided,
        DDS will use reflection facilities from IPython to load it.

      commit_type: (DBFS only, 'none', 'links_only', 'full', default 'full'). The type of commit that will be
        executed to update the paths. Committing a path in DBFS involves a full copy, which may be expensive,
        especially if the underlying table uses Databricks Delta. This is why the following options can be used:

        - none: no file will be committed. Useful for debugging.

        - links_only: a metadata link reference to the blob will be updated. This is much faster because it involves
            a 1kb file transfer of metadata as opposed to a full copy of a dataset.
             However, this means that the final tables are not readable by systems other than DDS, unless they understand
             the DDS file protocol.

        - full: the full dataset and metadata are transfered.

            NOTE: integration with Delta: the full transfer is currently implemented as an overwrite operation. This is
            compatible with the Delta IO protocol which will allow the user to revert to older versions if necessary.

            NOTE: costs: in order for DDS to work, the data must be at least in the internal store, and also copied to
            the final place. In the case of large tables, this may incur extra storage costs.

    :return: nothing
    """
    _set_store(store, internal_dir, data_dir, dbutils, commit_type)


def accept_module(module: Union[str, ModuleType]) -> None:
    """
    Marks a module as accepted for introspection. Only functions in the current scope and in accepted modules
    will be considered for the evaluation.

    Example to ensure that all the functions in my_lib are considered by DDS.

    ```py
    import my_lib
    dds.accept_module(my_lib)
    ```

    The example above causes the `my_lib` module to be imported. If it is not desired, the name of the module can
    be passed instead:

    ```py
    dds.accept_module("my_lib")
    ```

    """
    return _accept_module(module)


def whitelist_module(module: Union[str, ModuleType]) -> None:
    """
    Marks a module as whitelisted for introspection. Only functions in the current scope and in whitelisted modules
    will be considered for the evaluation.

    DEPRECATED: use the `accept_module` function instead.
    """
    warnings.warn(
        "The whitelist_module function has been renamed to 'accept_module', use 'accept_module' instead.",
        DeprecationWarning,
    )
    return _accept_module(module)
