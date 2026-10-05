# burst数据处理
# 输入的sizes必须是处理好的无padding的数值
import numpy as np


def fun(sizes, max_bursts):
    """Convert packet directions to signed burst lengths."""
    sizes = np.sign(np.asarray(sizes))

    if sizes.ndim != 1:
        raise ValueError("sizes must be a one-dimensional array")
    if max_bursts <= 0:
        raise ValueError("max_bursts must be positive")
    assert np.all(sizes != 0), "sizes must not contain 0"

    if sizes.size == 0:
        return np.empty(0, dtype=np.int32)

    boundaries = np.flatnonzero(sizes[1:] != sizes[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sizes.size]))
    burst_lengths = ends - starts
    bursts = burst_lengths * sizes[starts]
    return bursts[:max_bursts].astype(np.int32, copy=False)

