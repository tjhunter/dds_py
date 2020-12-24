"""
The main evaluation context.

All the information stored in this class is only valid for a single run.
"""
import logging
from types import ModuleType
from collections import OrderedDict
from typing import Tuple, Any, Dict, Set, Optional, NewType

from .fun_args import dds_hash as dds_hash
from .structures import (
    PyHash,
    DDSPath,
    FunctionInteractions,
    CanonicalPath,
    LocalDepPath,
    FunctionArgContextHash,
    FunctionIndirectInteractions,
)

_logger = logging.getLogger(__name__)


Package = NewType("Package", str)


class EvalMainContext(object):
    """
    The shared information across a single run.

    TODO: rename RunEvalContext
    """

    def __init__(
        self,
        start_module: ModuleType,
        whitelisted_packages: Set[Package],
        start_globals: Dict[str, Any],
        resolved_references: "OrderedDict[DDSPath, PyHash]",
    ):
        self.whitelisted_packages = whitelisted_packages
        self.start_module = start_module
        self.start_globals = start_globals
        self.resolved_references: "OrderedDict[DDSPath, PyHash]" = resolved_references
        # Hashes of all the static objects
        self._hashes: Dict[CanonicalPath, PyHash] = {}
        self.cached_fun_interactions: Dict[
            Tuple[CanonicalPath, FunctionArgContextHash], FunctionInteractions
        ] = dict()
        self.cached_objects: Dict[
            Tuple[LocalDepPath, CanonicalPath], Optional[Tuple[Any, CanonicalPath]]
        ] = dict()
        self.cached_indirect_interactions: Dict[
            CanonicalPath, FunctionIndirectInteractions
        ] = dict()

    def get_hash(self, path: CanonicalPath, obj: Any) -> PyHash:
        if path not in self._hashes:
            key = dds_hash(obj)
            _logger.debug(f"Cache key: %s: %s %s", path, type(obj), key)
            self._hashes[path] = key
            return key
        return self._hashes[path]

    def is_authorized_path(self, cp: CanonicalPath) -> bool:
        for idx in range(len(self.whitelisted_packages)):
            if ".".join(cp._path[:idx]) in self.whitelisted_packages:
                return True
        return False
