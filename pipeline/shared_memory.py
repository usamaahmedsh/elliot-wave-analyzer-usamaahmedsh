from multiprocessing import shared_memory
import numpy as np
from typing import Dict, Tuple, Any


def _ensure_np(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr)


def create_shared_arrays(arrays: Dict[str, np.ndarray]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, shared_memory.SharedMemory]]:
    """
    Create shared memory segments for given numpy arrays.

    Returns (meta, shm_objects) where meta maps key -> {name, shape, dtype}
    and shm_objects maps key -> SharedMemory instance (kept by the creator for cleanup).
    """
    meta = {}
    shms = {}

    for k, a in arrays.items():
        arr = _ensure_np(a)
        # For datetime64 arrays, store as int64 view
        if arr.dtype.kind == 'M':
            arr_view = arr.astype('datetime64[ns]').view('int64')
            dtype = np.int64
        else:
            arr_view = arr
            dtype = arr_view.dtype

        nbytes = arr_view.nbytes
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        # create numpy view on the shared buffer
        shm_arr = np.ndarray(arr_view.shape, dtype=dtype, buffer=shm.buf)
        # copy data
        shm_arr[:] = arr_view[:]

        meta[k] = {'name': shm.name, 'shape': arr_view.shape, 'dtype': str(arr_view.dtype)}
        shms[k] = shm

    return meta, shms


def attach_shared_view(shm_meta: Dict[str, Any]) -> np.ndarray:
    """Attach to a single shared array described by shm_meta and return a numpy view."""
    name = shm_meta['name']
    shape = tuple(shm_meta['shape'])
    # dtype string -> numpy dtype
    dtype = np.dtype(shm_meta['dtype'])
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # Note: caller is responsible for keeping shm alive via name or closing when done
    return arr


def cleanup_shared_objects(shm_objs: Dict[str, shared_memory.SharedMemory]):
    """Close and unlink SharedMemory objects created earlier."""
    for name, shm in shm_objs.items():
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
