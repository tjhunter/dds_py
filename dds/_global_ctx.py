from typing import (
    Tuple,
    Dict,
    Optional,
    List,
    NewType,
)

from .structures import (
    FunctionInteractions,
    CanonicalPath,
    FunctionArgContextHash,
)

PythonId = NewType("PythonId", int)


class GlobalContext(object):
    """
    The global context holds caches about from past evaluations.
    """

    def __init__(self):
        self.cached_fun_calls: Dict[
            Tuple[CanonicalPath, FunctionArgContextHash], List[CanonicalPath]
        ] = {}
        # The cached interactions
        # TODO: rethink the global cache, it is currently poorly interacting with variable updates
        self.cached_fun_interactions: Dict[
            Tuple[
                CanonicalPath,
                FunctionArgContextHash,
                Tuple[Tuple[CanonicalPath, PythonId], ...],
            ],
            FunctionInteractions,
        ] = {}


# Set this to None to disable the global context.
# TODO: expose as an option
_global_context: Optional[GlobalContext] = GlobalContext()  # type: ignore
