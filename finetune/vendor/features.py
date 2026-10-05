"""Feature routines from WFlib (MIT)."""
import numpy as np

def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs

    return adjusted_lengths

def agg_interval2(packets):
    features = []
    features.append(np.sum(packets>0))
    features.append(np.sum(packets<0))

    pos_packets = packets[packets>0]
    neg_packets = np.abs(packets[packets<0])
    features.append(np.sum(np.diff(pos_packets)))
    features.append(np.sum(np.diff(neg_packets)))

    dirs = np.sign(packets)
    assert not np.any(dirs == 0), "Array contains zero!"
    bursts = fast_count_burst(dirs)
    features.append(np.sum(bursts>0))
    features.append(np.sum(bursts<0))

    pos_bursts = bursts[bursts>0]
    neg_bursts = np.abs(bursts[bursts<0])
    if len(pos_bursts) == 0:
        features.append(0)
    else:
        features.append(np.mean(pos_bursts))
    if len(neg_bursts) == 0:
        features.append(0)
    else:
        features.append(np.mean(neg_bursts))

    return np.array(features, dtype=np.float32)

def process_MTAF(index, sequence, interval, max_len):
    packets = np.trim_zeros(sequence, "fb")
    abs_packets = np.abs(packets)
    st_time = abs_packets[0]
    st_pos = 0
    TAF = np.zeros((8, max_len))

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval
        if interval_idx == max_len - 1:
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time)

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos]
            TAF[..., interval_idx] = agg_interval2(cur_packets)
        st_pos = ed_pos
    
    return index, TAF

def process_TAM(index, sequence, maximum_load_time, max_matrix_len):
    feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix

    for pack in sequence:
        if pack == 0:
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                feature[1, idx] += 1
    return index, feature