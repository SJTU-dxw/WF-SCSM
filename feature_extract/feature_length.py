# Length数据处理
# 输入的sizes必须是处理好的无padding的数值
import numpy as np


def fun(sizes, output_length):
    sizes = np.sign(np.asarray(sizes))

    if sizes.ndim != 1:
        raise ValueError("sizes must be a one-dimensional array")
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    assert np.all(sizes != 0), "sizes must not contain 0"

    output = np.zeros(output_length, dtype=sizes.dtype)
    retained_length = min(len(sizes), output_length)
    output[:retained_length] = sizes[:retained_length]
    return output
