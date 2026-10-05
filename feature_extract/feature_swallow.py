# Swallow数据处理，参考https://github.com/wujinhe0814/Swallow/blob/main/CIF.py#L2
# 输入的times和sizes必须是处理好的无padding的数值
# 内部会自动减去第一个time，处理为sign
# 内部会确认times递增，sizes不为0，times[-1]不为0

# 时间过长为max_slot_duration，会扔掉超出max_slot_duration * slot_len的包
# 时间过短为min_slot_duration
# 实际为3 * load_time / slot_len
import numpy as np


def fun(times, sizes, min_slot_duration, max_slot_duration, time_window_multiplier, slot_len):
    times = np.asarray(times) - times[0]
    sizes = np.sign(sizes)

    assert np.all(np.diff(times) >= 0), "times must be strictly increasing"
    assert times[-1] != 0.0, "times[-1] must not be 0.0"
    assert np.all(sizes != 0), "sizes must not contain 0"

    load_time = times[-1]

    packet_count_matrix = [[0 for i in range(slot_len)], [0 for i in range(slot_len)]]  # 每行 [上行包, 下行包]
    slot_duration =  time_window_multiplier * load_time / slot_len
    min_time_slot = min_slot_duration
    max_time_slot = max_slot_duration
    if slot_duration <= min_time_slot:
        slot_duration = min_time_slot
    if slot_duration >= max_time_slot:
        slot_duration = max_time_slot
    for i, t in enumerate(times):
        slot_index = int(t // slot_duration)
        if slot_index >= slot_len:
            break
        if sizes[i] == 1:
            packet_count_matrix[0][slot_index] += 1  # 上行包计数
        else:
            packet_count_matrix[1][slot_index] += 1  # 下行包计数
    return packet_count_matrix, load_time, slot_duration
