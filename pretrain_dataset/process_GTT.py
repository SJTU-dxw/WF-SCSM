import numpy as np
import h5py
import hdf5plugin
import tqdm
import time


source_path = "/nvme/dxw/SCSM/pretrain_dataset/GTT23.hdf5"
target_path = "/nvme/dxw/SCSM/pretrain_dataset/GTT23_train.hdf5"

# 超参数
batch_size = 1024
chunk_rows = 64
sequence_length = 5000
min_relay_cells = 20

def create_target_file(target, total):
    target.create_dataset(
        "times",
        shape=(total, sequence_length),
        maxshape=(None, sequence_length),
        dtype=np.float32,
        chunks=(chunk_rows, sequence_length),
        compression="lzf",
        shuffle=True,
    )

    target.create_dataset(
        "directions",
        shape=(total, sequence_length),
        maxshape=(None, sequence_length),
        dtype=np.int8,
        chunks=(chunk_rows, sequence_length),
        compression="lzf",
        shuffle=True,
    )

    target.create_dataset(
        "lengths",
        shape=(total,),
        maxshape=(None,),
        dtype=np.uint16,
        chunks=(8192,),
        compression="lzf",
        shuffle=True,
    )

    target.create_dataset(
        "labels",
        shape=(total,),
        maxshape=(None,),
        dtype="S44",
        chunks=(8192,),
        compression="lzf",
    )

    target.create_dataset(
        "day",
        shape=(total,),
        maxshape=(None,),
        dtype=np.uint8,
        chunks=(8192,),
        compression="lzf",
    )

    target.attrs["total_rows"] = total
    target.attrs["sequence_length"] = sequence_length
    target.attrs["source_file"] = source_path
    target.attrs["complete"] = False
    target.attrs["processing_protocol"] = "first_relay_begin_relay_only_min20_v2"
    target.attrs["min_relay_cells"] = min_relay_cells


def process_batch(records):
    """Start at the first BEGIN, compact relay cells, preserve real time gaps."""
    cells = records["cells"]
    width = cells.shape[1]
    lengths = records["len"].astype(np.int64)
    if np.any(lengths < 0) or np.any(lengths > width) or width != sequence_length:
        raise ValueError("Unexpected source cell lengths/sequence width")
    columns = np.arange(width)[None, :]
    valid = columns < lengths[:, None]
    relay = valid & ((cells["cell_cmd"] == 3) | (cells["cell_cmd"] == 9))
    begin = relay & (cells["relay_cmd"] == 1)
    retained_rows = np.flatnonzero(begin.any(axis=1))
    dropped_no_begin = len(records) - len(retained_rows)
    if not len(retained_rows):
        return {}, dropped_no_begin, 0
    first = begin[retained_rows].argmax(axis=1)
    selected = relay[retained_rows] & (columns >= first[:, None])
    selected_counts = selected.sum(axis=1)
    keep = selected_counts >= min_relay_cells
    dropped_too_short = np.count_nonzero(~keep)
    retained_rows = retained_rows[keep]
    first = first[keep]
    selected = selected[keep]
    if not len(retained_rows):
        return {}, dropped_no_begin, dropped_too_short
    row, column = np.nonzero(selected)
    destinations = np.cumsum(selected, axis=1, dtype=np.int32)[row, column] - 1
    raw_times = cells["time"][retained_rows]
    # Subtract in float64 before casting, to preserve sub-second precision.
    origin = raw_times[np.arange(len(retained_rows)), first]
    relative_times = raw_times[row, column].astype(np.float64) - origin[row]
    directions = cells["direction"][retained_rows][row, column]
    if not np.all(np.isfinite(relative_times)) or not np.all(np.isin(directions, (-1, 1))):
        raise ValueError("Invalid retained timestamps or directions")
    packed_times = np.zeros((len(retained_rows), width), dtype=np.float64)
    packed_times[row, destinations] = relative_times
    packed_lengths = selected.sum(axis=1)
    consecutive = np.arange(width - 1)[None, :] < (packed_lengths - 1)[:, None]
    if np.any((np.diff(packed_times, axis=1) < 0) & consecutive):
        raise ValueError("Retained timestamps are not monotonic; refusing to reorder")
    packed_directions = np.zeros((len(retained_rows), width), dtype=np.int8)
    packed_directions[row, destinations] = directions
    return dict(
        times=packed_times.astype(np.float32),
        directions=packed_directions,
        lengths=packed_lengths.astype(np.uint16),
        labels=records["domain"][retained_rows],
        day=records["day"][retained_rows].astype(np.uint8),
    ), dropped_no_begin, dropped_too_short


def main():
    with h5py.File(source_path, "r") as source:
        circuits = source["circuits"]
        total = len(circuits)
        print("原始记录数：", total)
        print(
            "处理协议：首个 relay BEGIN 起点，保留后续 RELAY/RELAY_EARLY，"
            f"少于 {min_relay_cells} 个单元的记录丢弃，时间归零"
        )
        # Never delete or overwrite an existing conversion.
        with h5py.File(target_path, "x", libver="latest") as target:
            create_target_file(target, total)
            written = dropped_no_begin = dropped_too_short = 0
            conversion_start = time.time()
            with tqdm.tqdm(total=total, unit="record", desc="Converting GTT23") as progress:
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    result, no_begin, too_short = process_batch(circuits[start:end])
                    dropped_no_begin += no_begin
                    dropped_too_short += too_short
                    count = len(result.get("lengths", ()))
                    if count:
                        for name, values in result.items():
                            target[name][written:written + count] = values
                        written += count
                    progress.update(end - start)
                    progress.set_postfix(
                        kept=written,
                        no_begin=dropped_no_begin,
                        under_20=dropped_too_short,
                        refresh=False,
                    )
            for dataset in target.values():
                dataset.resize(written, axis=0)
            target.attrs["total_rows"] = written
            target.attrs["source_rows"] = total
            target.attrs["dropped_no_begin"] = dropped_no_begin
            target.attrs["dropped_too_short"] = dropped_too_short
            if not written:
                raise ValueError(
                    f"No records contain at least {min_relay_cells} relay cells "
                    "from the first relay BEGIN"
                )
            target.attrs["complete"] = True
            target.flush()
            print(
                f"保留 {written} 条；剔除无 BEGIN 记录 {dropped_no_begin} 条；"
                f"剔除少于 {min_relay_cells} 个单元的记录 {dropped_too_short} 条"
            )
            print(f"转换完成，用时 {(time.time() - conversion_start) / 3600:.2f} 小时")
            print(f"保存位置：{target_path}")


if __name__ == "__main__":
    main()
