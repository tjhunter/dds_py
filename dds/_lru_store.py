from .store import Store

from functools import lru_cache

from pathlib import PurePath
from typing import Any, Optional, List, Union
from collections import OrderedDict

from .codec import codec_registry, CodecRegistry
from .structures import (
    PyHash,
    DDSPath,
    KSException,
    GenericLocation,
    ProtocolRef,
    FileCodecProtocol,
    CodecProtocol,
)
from .structures_utils import SupportedTypeUtils as STU
import logging

_logger = logging.getLogger(__name__)

default_cache_size = 10


class LRUCacheStore(Store):
    """
    A store that caches the most recent objects.

    This store keeps objects in memory and reserves them if requested.

    TODO: consider adding also the paths.
    It is not necessarily a good idea. For the local case, the speed of fetching a path
    is very high, and in a distributed store, it introduces coherency issues which are
    much more troublesome in practice. This should be added only for very slow stores
    updated by a single process, which is not a common case.
    """

    def __init__(self, store: Store, num_elem: int):
        self._store: Store = store
        self._num_elem = num_elem

        @lru_cache(maxsize=num_elem)
        def my_cache(key: PyHash) -> Optional[Any]:
            _logger.debug(f"Fetching key {key}")
            res = self._store.fetch_blob(key)
            _logger.debug(f"Fetching key {key} completed: {type(res)}")
            return res

        self._fetch_function = my_cache

    def has_blob(self, key: PyHash) -> bool:
        # TODO: consider if this should also be cached.
        # TODO: use a custom cache instead of lru_cache to sync both content and key
        return self._store.has_blob(key)

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        _logger.debug(f"fetch_blob key {key}")
        return self._fetch_function(key)

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef]) -> None:
        """
        Storing the blob is not cached.

        The operation of storing the blob may trigger side effects which are referentially
        transparent but have a big impact on the performance.
        For example, Spark dataframes are converted to datasets, as opposed to just
        lazy query plans.
        """
        _logger.debug(f"store_blob key {key}")
        self._store.store_blob(key, blob, codec)

    def sync_paths(self, paths: "OrderedDict[DDSPath, PyHash]") -> None:
        _logger.debug(f"sync_paths {paths}")
        self._store.sync_paths(paths)

    def fetch_paths(self, paths: List[DDSPath]) -> "OrderedDict[DDSPath, PyHash]":
        res = self._store.fetch_paths(paths)
        _logger.debug(f"fetch_paths {paths} -> {res}")
        return res

    def codec_registry(self) -> CodecRegistry:
        return self._store.codec_registry()

    def __repr__(self):
        return f"LRUCacheStore(num_elem={self._num_elem} store={self._store})"
