import ast
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
from tqdm import tqdm


class RewardCache:
    """
    A general-purpose, memory-based LRU cache for storing reward computation results.
    """

    def __init__(
        self,
        max_size: int = 10000,
        enable_stats: bool = True,
    ):
        self.max_size = max_size
        self.enable_stats = enable_stats
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()

        if self.enable_stats:
            self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                if self.enable_stats:
                    self.stats["misses"] += 1
                return None

            self._cache.move_to_end(key)
            if self.enable_stats:
                self.stats["hits"] += 1
            return self._cache[key]

    def _put_no_lock(self, key: str, value: Any):
        """Internal put method that assumes a lock is already held."""
        if key in self._cache:
            self._cache[key] = value
            self._cache.move_to_end(key)
            return

        if len(self._cache) >= self.max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            if self.enable_stats:
                self.stats["evictions"] += 1

        self._cache[key] = value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._put_no_lock(key, value)

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Retrieve multiple cached values by a list of keys in a single operation.

        Args:
            keys: A list of keys to retrieve.

        Returns:
            A dictionary mapping found keys to their values. Keys not found
            in the cache will be absent from the dictionary.
        """
        found_items = {}
        with self._lock:
            for key in keys:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    found_items[key] = self._cache[key]
                    if self.enable_stats:
                        self.stats["hits"] += 1
                else:
                    if self.enable_stats:
                        self.stats["misses"] += 1
        return found_items

    def batch_put(self, items: Dict[str, Any]) -> None:
        """
        Store multiple key-value pairs in the cache in a single operation.

        Args:
            items: A dictionary of key-value pairs to store.
        """
        with self._lock:
            for key, value in items.items():
                self._put_no_lock(key, value)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            if self.enable_stats:
                self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enable_stats:
            return {"stats_disabled": True}

        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                **self.stats,
                "size": len(self._cache),
                "hit_rate": hit_rate,
                "max_size": self.max_size,
            }


class PersistentRewardCache(RewardCache):
    """
    Extension of RewardCache that supports persistent storage with usage-based persistence.
    """

    def __init__(
        self,
        max_size: int = 10000,
        storage_backend: Optional[Any] = None,
        enable_stats: bool = True,
        persist_threshold: int = 3,
        usage_window_size: int = 1000,
    ):
        super().__init__(max_size=max_size, enable_stats=enable_stats)
        self.storage_backend = storage_backend
        self.persist_threshold = persist_threshold
        self.usage_window_size = usage_window_size

        self._usage_counter = defaultdict(int)
        self._access_history = []
        self._persisted_keys = set()
        self._storage_keys = set()

        if self.enable_stats:
            self.stats.update(
                {
                    "storage_hits": 0,
                    "storage_misses": 0,
                    "storage_writes": 0,
                    "promotions": 0,
                    "persistence_decisions": 0,
                    "keys_qualified_for_persistence": 0,
                }
            )

    def get(self, key: str) -> Optional[Any]:
        """Try memory first, then storage, and track usage."""
        value = super().get(key)
        if value is not None:
            # Track usage for persistence decision
            self._track_usage(key)
            return value

        # If not in memory, try storage
        if self.storage_backend and key in self._storage_keys:
            value = self._load_from_storage(key)
            if value is not None:
                if self.enable_stats:
                    self.stats["storage_hits"] += 1
                    self.stats["promotions"] += 1
                # Promote to memory cache and track usage
                self._promote_to_memory(key, value)
                self._track_usage(key)
                return value
            else:
                # Key was in storage_keys but couldn't be loaded, remove it
                self._storage_keys.discard(key)
                if self.enable_stats:
                    self.stats["storage_misses"] += 1
        elif self.storage_backend and self.enable_stats:
            self.stats["storage_misses"] += 1

        return None

    def put(self, key: str, value: Any) -> None:
        """Store in memory and conditionally in persistent storage based on usage."""
        with self._lock:
            super().put(key, value)

            # Check if this key should be persisted based on usage
            if self.storage_backend and self._should_persist(key):
                if key not in self._persisted_keys:
                    if self._save_to_storage(key, value):
                        self._storage_keys.add(key)
                        self._persisted_keys.add(key)
                        if self.enable_stats:
                            self.stats["storage_writes"] += 1
                            self.stats["keys_qualified_for_persistence"] += 1

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Efficiently get a batch of keys, checking memory first, then storage.
        """
        # Pass 1: Check memory cache. This is fast.
        found_in_memory = super().batch_get(keys)

        # Track usage for all memory hits
        with self._lock:
            for key in found_in_memory.keys():
                self._track_usage(key)

        # Identify keys that were not found in memory
        memory_misses = [k for k in keys if k not in found_in_memory]
        if not memory_misses or not self.storage_backend:
            return found_in_memory

        # Pass 2: Check persistent storage for the misses
        found_in_storage = {}
        for key in memory_misses:
            if key in self._storage_keys:
                value = self._load_from_storage(key)
                if value is not None:
                    found_in_storage[key] = value
                    if self.enable_stats:
                        self.stats["storage_hits"] += 1
                else:
                    self._storage_keys.discard(key)  # Clean up if file was deleted/corrupt
                    if self.enable_stats:
                        self.stats["storage_misses"] += 1
            else:
                if self.enable_stats:
                    self.stats["storage_misses"] += 1

        # Pass 3: Promote items from storage to memory in a single batch
        if found_in_storage:
            if self.enable_stats:
                self.stats["promotions"] += len(found_in_storage)
            # Use super().batch_put to avoid re-triggering persistence logic
            super().batch_put(found_in_storage)

            # Track usage for newly promoted items
            with self._lock:
                for key in found_in_storage.keys():
                    self._track_usage(key)

        # Combine results from memory and storage
        found_in_memory.update(found_in_storage)
        return found_in_memory

    def batch_put(self, items: Dict[str, Any]) -> None:
        """
        Store a batch of items, then check which ones should be persisted.
        """
        # First, put everything into the fast memory cache
        super().batch_put(items)

        if not self.storage_backend:
            return

        # Then, decide which of the newly added items to persist
        with self._lock:
            for key, value in items.items():
                if self._should_persist(key) and key not in self._persisted_keys:
                    if self._save_to_storage(key, value):
                        self._storage_keys.add(key)
                        self._persisted_keys.add(key)
                        if self.enable_stats:
                            self.stats["storage_writes"] += 1
                            self.stats["keys_qualified_for_persistence"] += 1

    def _track_usage(self, key: str) -> None:
        """Track usage of a key for persistence decisions."""
        with self._lock:
            # Increment usage counter
            self._usage_counter[key] += 1

            # Add to access history
            self._access_history.append(key)
            if len(self._access_history) > self.usage_window_size:
                oldest_key = self._access_history.pop(0)
                self._usage_counter[oldest_key] -= 1
                if self._usage_counter[oldest_key] <= 0:
                    del self._usage_counter[oldest_key]

    def _should_persist(self, key: str) -> bool:
        """
        Determine if a key should be persisted based on usage patterns.

        Returns True if the key has been accessed frequently enough in memory.
        """
        with self._lock:
            if self.enable_stats:
                self.stats["persistence_decisions"] += 1
            return self._usage_counter.get(key, 0) >= self.persist_threshold

    def _promote_to_memory(self, key: str, value: Any) -> None:
        with self._lock:
            super().put(key, value)

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._usage_counter.clear()
            self._access_history.clear()
            self._persisted_keys.clear()
            if self.storage_backend:
                for key in list(self._storage_keys):
                    self._remove_from_storage(key)
                self._storage_keys.clear()

    def get_usage_stats(self) -> Dict[str, Any]:
        with self._lock:
            top_keys = sorted(self._usage_counter.items(), key=lambda x: x[1], reverse=True)[:10]
            return {
                "total_tracked_keys": len(self._usage_counter),
                "persisted_keys_count": len(self._persisted_keys),
                "persist_threshold": self.persist_threshold,
                "usage_window_size": self.usage_window_size,
                "current_window_fill": len(self._access_history),
                "top_10_most_accessed": top_keys,
                "keys_above_threshold": sum(
                    1 for count in self._usage_counter.values() if count >= self.persist_threshold
                ),
            }

    def _save_to_storage(self, key: str, value: Any) -> bool:
        try:
            if hasattr(self.storage_backend, 'set'):
                self.storage_backend.set(key, value)
                return True
            elif hasattr(self.storage_backend, 'put'):
                self.storage_backend.put(key, value)
                return True
        except Exception as e:
            print(f"Warning: Failed to save to storage backend: {e}")
        return False

    def _load_from_storage(self, key: str) -> Optional[Any]:
        try:
            if hasattr(self.storage_backend, 'get'):
                return self.storage_backend.get(key)
        except Exception as e:
            print(f"Warning: Failed to load from storage backend: {e}")
        return None

    def _remove_from_storage(self, key: str) -> None:
        try:
            if hasattr(self.storage_backend, 'delete'):
                self.storage_backend.delete(key)
            elif hasattr(self.storage_backend, 'remove'):
                self.storage_backend.remove(key)
        except Exception as e:
            print(f"Warning: Failed to remove from storage backend: {e}")

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        if self.enable_stats:
            stats.update(
                {
                    'storage_keys_count': len(self._storage_keys),
                    'persisted_keys_count': len(self._persisted_keys),
                    'total_keys': len(self._cache) + len(self._storage_keys),
                    'storage_hits': self.stats.get('storage_hits', 0),
                    'storage_misses': self.stats.get('storage_misses', 0),
                    'storage_writes': self.stats.get('storage_writes', 0),
                    'promotions': self.stats.get('promotions', 0),
                    'persistence_decisions': self.stats.get('persistence_decisions', 0),
                    'keys_qualified_for_persistence': self.stats.get('keys_qualified_for_persistence', 0),
                    'persist_threshold': self.persist_threshold,
                    'usage_window_size': self.usage_window_size,
                }
            )
        return stats


class FileStoragePersistentRewardCache(PersistentRewardCache):
    # This class does not need any changes, as the batch logic is handled
    # by its parent, PersistentRewardCache.
    def __init__(
        self,
        max_size: int,
        cache_dir: str,
        enable_stats: bool = True,
        use_json: bool = False,
        persist_threshold: int = 3,
    ):
        super().__init__(
            max_size=max_size,
            storage_backend=self,
            enable_stats=enable_stats,
            persist_threshold=persist_threshold,
            usage_window_size=max_size * persist_threshold,
        )
        self.cache_dir = Path(cache_dir)
        self.use_json = use_json
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._discover_existing_keys()

    def _discover_existing_keys(self) -> None:
        if not self.cache_dir.exists():
            return
        extension = ".json" if self.use_json else ".pkl"
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(extension)]
        for filename in tqdm(cache_files, desc="Discovering existing cache keys"):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if self.use_json:
                    with open(file_path) as f:
                        data = json.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                if isinstance(data, dict) and "key" in data:
                    self._storage_keys.add(data["key"])
                    self._persisted_keys.add(data["key"])
            except Exception:
                continue

    def _get_file_path(self, key: str) -> Path:
        hash_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
        extension = ".json" if self.use_json else ".pkl"
        return self.cache_dir / f"{hash_key}{extension}"

    def _save_to_storage(self, key: str, value: Any) -> bool:
        try:
            file_path = self._get_file_path(key)
            data = {
                "key": key,
                "value": value,
                "timestamp": time.time(),
                "access_count": self._usage_counter.get(key, 0),
            }
            if self.use_json:
                with open(file_path, 'w') as f:
                    json.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Warning: Failed to save cache entry to disk: {e}")
            return False

    def _load_from_storage(self, key: str) -> Optional[Any]:
        try:
            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None
            if self.use_json:
                with open(file_path) as f:
                    data = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            if data.get("key") == key:
                return data.get("value")

            return None

        except Exception as e:
            print(f"Warning: Failed to load cache entry from disk: {e}")
            return None

    def _remove_from_storage(self, key: str) -> None:
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to remove cache file from disk: {e}")

    def get_disk_usage_info(self) -> Dict[str, Any]:
        try:
            total_size, file_count = 0, 0
            extension = ".json" if self.use_json else ".pkl"
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file() and file_path.suffix == extension:
                    total_size += file_path.stat().st_size
                    file_count += 1
            return {
                "cache_directory": str(self.cache_dir),
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "average_file_size_bytes": total_size / file_count if file_count > 0 else 0,
            }
        except Exception as e:
            return {"error": str(e)}


@ray.remote
class DistributedRewardCache:
    """A Ray actor providing distributed cache with new batch operations."""

    def __init__(self, name: str = "verl_reward_cache", max_size: int = 50000, persist_threshold: int = 3):
        # ... (init logic remains the same) ...
        reward_cache_dir = os.getenv('REWARD_CACHE_DIR', None)
        if reward_cache_dir is None:
            hdfs_volumes = os.environ.get('ARNOLD_HDFSFUSE_VOLUMES', '')
            if hdfs_volumes:
                try:
                    volumes = ast.literal_eval(hdfs_volumes)
                    mount_path = volumes[0]['mount_path']
                    cache_dir = os.path.join(mount_path, name)
                except (ValueError, SyntaxError, KeyError, IndexError):
                    cache_dir = "/tmp/" + name
            else:
                cache_dir = "/tmp/" + name
        else:
            cache_dir = reward_cache_dir

        self.cache = FileStoragePersistentRewardCache(
            max_size=max_size,
            cache_dir=cache_dir,
            use_json=True,
            enable_stats=True,
            persist_threshold=persist_threshold,
        )

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def put(self, key: str, value: Any) -> None:
        self.cache.put(key, value)

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Remotely executes a batch get on the cache instance."""
        return self.cache.batch_get(keys)

    def batch_put(self, items: Dict[str, Any]) -> None:
        """Remotely executes a batch put on the cache instance."""
        self.cache.batch_put(items)

    def get_stats(self) -> Dict[str, Any]:
        return self.cache.get_stats()

    def get_usage_stats(self) -> Dict[str, Any]:
        return self.cache.get_usage_stats()

    def get_disk_usage_info(self) -> Dict[str, Any]:
        return self.cache.get_disk_usage_info()

    def clear(self) -> None:
        self.cache.clear()

    def size(self) -> int:
        return self.cache.size()


class DistributedCacheWrapper:
    """Client-side wrapper with new batch methods."""

    def __init__(self, actor):
        self.actor = actor

    def get(self, key: str) -> Optional[Any]:
        return ray.get(self.actor.get.remote(key))

    def put(self, key: str, value: Any) -> None:
        ray.get(self.actor.put.remote(key, value))

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Gets a batch of keys with a single network round-trip.

        Returns:
            A dictionary of found key-value pairs.
        """
        return ray.get(self.actor.batch_get.remote(keys))

    def batch_put(self, items: Dict[str, Any]) -> None:
        """
        Puts a batch of items with a single network round-trip.
        This call is synchronous and will wait for the actor to complete.
        """
        ray.get(self.actor.batch_put.remote(items))

    def get_stats(self) -> Dict[str, Any]:
        return ray.get(self.actor.get_stats.remote())

    def get_usage_stats(self) -> Dict[str, Any]:
        return ray.get(self.actor.get_usage_stats.remote())

    def get_disk_usage_info(self) -> Dict[str, Any]:
        return ray.get(self.actor.get_disk_usage_info.remote())

    def clear(self) -> None:
        ray.get(self.actor.clear.remote())

    def size(self) -> int:
        return ray.get(self.actor.size.remote())
