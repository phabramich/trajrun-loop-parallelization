"""numpy-шим под API mlx.core — тот же batched-VM на CPU.

mx.eval -> no-op (numpy не ленив). dtype-имена совпадают.
"""

import numpy as np


class _NP:
    array = staticmethod(np.array)
    asarray = staticmethod(np.asarray)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    full = staticmethod(np.full)
    full_like = staticmethod(np.full_like)
    zeros_like = staticmethod(np.zeros_like)
    ones_like = staticmethod(np.ones_like)
    arange = staticmethod(np.arange)
    where = staticmethod(np.where)
    clip = staticmethod(np.clip)
    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    concatenate = staticmethod(np.concatenate)
    log2 = staticmethod(np.log2)
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    uint8 = np.uint8
    uint16 = np.uint16
    uint32 = np.uint32
    uint64 = np.uint64
    bool_ = np.bool_

    @staticmethod
    def eval(*a):
        return a


np_shim = _NP()
