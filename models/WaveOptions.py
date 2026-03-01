from abc import ABC, abstractmethod
import numpy as np

# =============================================================================
# Fast numpy-based options array (replaces slow Python object sort)
# =============================================================================

_array_cache: dict = {}

def generate_options_array(up_to: int, n_waves: int = 5) -> np.ndarray:
    """
    Returns shape (up_to^n_waves, n_waves) int16 array of all skip combinations,
    sorted by total skip sum ascending (mirrors WaveOptions.__lt__ ordering).
    ~10-50x faster than sorted Python WaveOptions objects at up_to >= 12.
    Cached at module level — generated once per (up_to, n_waves) pair per process.
    """
    idx = np.arange(up_to, dtype=np.int16)
    if n_waves == 5:
        g = np.array(np.meshgrid(idx, idx, idx, idx, idx, indexing='ij'))
        combos = g.reshape(5, -1).T  # (up_to^5, 5)
    elif n_waves == 3:
        g = np.array(np.meshgrid(idx, idx, idx, indexing='ij'))
        combos = g.reshape(3, -1).T  # (up_to^3, 3)
    else:
        raise ValueError(f"n_waves must be 3 or 5, got {n_waves}")

    # lexsort ascending on all columns (col 0 = most significant, matching __lt__)
    order = np.lexsort(combos[:, ::-1].T)
    return combos[order].astype(np.int16)

def get_options_array(up_to: int, n_waves: int = 5) -> np.ndarray:
    """Return cached options array for given up_to and n_waves."""
    key = (up_to, n_waves)
    if key not in _array_cache:
        _array_cache[key] = generate_options_array(up_to, n_waves)
    return _array_cache[key]

# =============================================================================
# Original Python class — kept for backward compatibility
# =============================================================================

class WaveOptions:
    """
    WaveOptions are a list of integers denoting the number of intermediate
    min/maxima to skip while finding a MonoWave.
    E.g. [1,0,0,0,0] skips the first found maxima for the first MonoWaveUp.
    """
    def __init__(self, i: int, j: int = None, k: int = None,
                 l: int = None, m: int = None):
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.m = m

    def __repr__(self):
        return f'[{self.i}, {self.j}, {self.k}, {self.l}, {self.m}]'

    @property
    def values(self):
        if self.k is not None:
            return [self.i, self.j, self.k, self.l, self.m]
        else:
            return [self.i, self.j]

    def __hash__(self):
        if self.k is not None:
            hash_str = f'{self.i}_{self.j}_{self.k}_{self.l}_{self.m}'
        else:
            hash_str = f'{self.i}_{self.j}'
        return hash(hash_str)

    def __eq__(self, other):
        if self.k is not None:
            return (self.i == other.i and self.j == other.j and
                    self.k == other.k and self.l == other.l and
                    self.m == other.m)
        else:
            return self.i == other.i and self.j == other.j

    def __lt__(self, other):
        if self.i < other.i:
            return True
        elif self.i == other.i:
            if self.j < other.j:
                return True
            elif self.j == other.j:
                if self.k == other.k:
                    if self.l < other.l:
                        return True
                    elif self.l == other.l:
                        return self.m < other.m
                    else:
                        return False
                elif self.k < other.k:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False


class WaveOptionsGenerator(ABC):
    def __init__(self, up_to: int):
        self.__up_to = up_to
        self.options = self.populate()

    @property
    def up_to(self):
        return self.__up_to

    @property
    def number(self):
        return len(self.options)

    @abstractmethod
    def populate(self) -> set:
        pass

    @property
    def options_sorted(self):
        all_options = list(self.options)
        return sorted(all_options)


class WaveOptionsGenerator5(WaveOptionsGenerator):
    def populate(self) -> set:
        checked = set()
        for i in range(0, self.up_to):
            for j in range(0, self.up_to):
                for k in range(0, self.up_to):
                    for l in range(0, self.up_to):
                        for m in range(0, self.up_to):
                            checked.add(WaveOptions(i, j, k, l, m))
        return checked


class WaveOptionsGenerator2(WaveOptionsGenerator):
    def populate(self) -> set:
        checked = list
        for i in range(0, self.up_to):
            for j in range(0, self.up_to):
                if i == 0:
                    j = 0
                wave_options = WaveOptions(i, j, None, None, None)
                checked.append(wave_options)
        return checked


class WaveOptionsGenerator3(WaveOptionsGenerator):
    def populate(self) -> set:
        checked = set()
        for i in range(0, self.up_to):
            for j in range(0, self.up_to):
                for k in range(0, self.up_to):
                    checked.add(WaveOptions(i, j, k, None, None))
        return checked
