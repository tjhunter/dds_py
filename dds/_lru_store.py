import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, List

from .codec import CodecRegistry
from .store import Store
from .structures import (
    PyHash,
    DDSPath,
    ProtocolRef,
)

_logger = logging.getLogger(__name__)

# The default cache is conservatively small to prevent seemingly memory leaks.
default_cache_size = 10


# To ensure that we can store None in the cache
@dataclass(frozen=True)
class Entry:
    obj: Any


class LRUCache(object):
    """
    Very simple LRU cache implementation.

    The reason for not using the default 'lru_cache' implementation of python is that
    the latter does not allow probing into the cache.
    """

    # initialising capacity
    def __init__(self, capacity: int):
        self._cache: OrderedDict[PyHash, Entry] = OrderedDict()
        self._capacity = capacity

    def get(self, key: PyHash) -> Optional[Entry]:
        if key not in self._cache:
            return None
        else:
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: PyHash, value: Any) -> None:
        self._cache[key] = Entry(value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)


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
        self._cache = LRUCache(num_elem)

    def has_blob(self, key: PyHash) -> bool:
        # Check the cache first for the key, and then check the store.
        return (self._cache.get(key) is not None) or self._store.has_blob(key)

    def fetch_blob(self, key: PyHash) -> Optional[Any]:
        cache_obj = self._cache.get(key)
        if cache_obj is not None:
            return cache_obj.obj
        # Not in the cache
        _logger.debug(f"Fetching key {key}")
        res = self._store.fetch_blob(key)
        _logger.debug(f"Fetching key {key} completed: {type(res)}")
        self._cache.put(key, res)
        return res

    def store_blob(self, key: PyHash, blob: Any, codec: Optional[ProtocolRef]) -> None:
        """
        Storing the blob is not cached.

        The operation of storing the blob may trigger side effects which are referentially
        transparent but have a big impact on the performance.
        For example, Spark dataframes are fully materialized and stored as parquet,
         as opposed to just lazy query plans.
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
