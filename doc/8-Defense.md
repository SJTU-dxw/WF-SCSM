# Defense 数据生成

## 下载数据

- 数据来源：Tik-Tok 提供的 [Zenodo 数据](https://zenodo.org/records/11631265/files/Undefended.zip)

```bash
cd pretrain_dataset
wget https://zenodo.org/records/11631265/files/Undefended.zip
unzip Undefended.zip
cd ..
```

## Palette

```bash
cd defense/Palatte
python main.py \
    --traces_path "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000"
cd ../..
```

## WTF-PAD

```bash
cd defense/wtfpad
python main.py \
    --traces_path "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000"
cd ../..
```

## FRONT

```bash
cd defense/front
python main.py \
    --p "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000"
cd ../..
```

## Tamaraw

```bash
cd defense/tamaraw
python tamaraw.py \
    --traces_path "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000"
cd ../..
```

## RegulaTor

```bash
cd defense/regulartor
python regulator_sim.py \
    --source_path "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000/" \
    --output_path "/nvme/dxw/SCSM/pretrain_dataset/defense-output/raw-data-50-1000_regulator/"
cd ../..
```

## TrafficSilver

### Round Robin

```bash
cd defense/trafficsilver
python simulator.py \
    --p "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000/" \
    --o "/nvme/dxw/SCSM/pretrain_dataset/defense-output/raw-data-50-1000_rb/" \
    --s round_robin
cd ../..
```

### By Direction

```bash
cd defense/trafficsilver
python simulator.py \
    --p "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000/" \
    --o "/nvme/dxw/SCSM/pretrain_dataset/defense-output/raw-data-50-1000_bd/" \
    --s in_and_out
cd ../..
```

### Batched Weighted Random

```bash
cd defense/trafficsilver
python simulator.py \
    --p "/nvme/dxw/SCSM/pretrain_dataset/raw-data-50-1000/" \
    --o "/nvme/dxw/SCSM/pretrain_dataset/defense-output/raw-data-50-1000_bwr/" \
    --s batched_weighted_random \
    -r 50,70 \
    -a 1,1,1
cd ../..
```

## 转换为训练 HDF5

每次运行只处理一种 Defense。TrafficSilver 的每个非空 `_split_N` 作为独立样本，
空分路自动跳过。

### Palette

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root raw-data-50-1000_Palette \
    --target DF_Palette_train.hdf5 \
    --defense-name Palette
cd ..
```

### WTF-PAD

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root raw-data-50-1000_wtfpad \
    --target DF_WTF-PAD_train.hdf5 \
    --defense-name WTF-PAD
cd ..
```

### FRONT

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root raw-data-50-1000_front \
    --target DF_FRONT_train.hdf5 \
    --defense-name FRONT
cd ..
```

### Tamaraw

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root raw-data-50-1000_tamaraw \
    --target DF_Tamaraw_train.hdf5 \
    --defense-name Tamaraw
cd ..
```

### RegulaTor

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root defense-output/raw-data-50-1000_regulator \
    --target DF_RegulaTor_train.hdf5 \
    --defense-name RegulaTor
cd ..
```

### TrafficSilver Round Robin

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root defense-output/raw-data-50-1000_rb \
    --target DF_TrafficSilver_RR_train.hdf5 \
    --defense-name TrafficSilver-RR
cd ..
```

### TrafficSilver By Direction

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root defense-output/raw-data-50-1000_bd \
    --target DF_TrafficSilver_BD_train.hdf5 \
    --defense-name TrafficSilver-BD
cd ..
```

### TrafficSilver Batched Weighted Random

```bash
cd pretrain_dataset
python process_defense.py \
    --source-root defense-output/raw-data-50-1000_bwr \
    --target DF_TrafficSilver_BWR_train.hdf5 \
    --defense-name TrafficSilver-BWR
cd ..
```

## 修复重复的首包时间戳

WTF-PAD、FRONT、RegulaTor、TrafficSilver-BD 和 TrafficSilver-BWR 中，部分
trace 的后续包与首包具有相同时间戳。以下命令将这些后续时间戳调整为首包之后
最小的可表示时间，并在校验后原位置替换五个 HDF5。请在 Fine-tune 前运行一次。

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python
"$PYTHON_BIN" pretrain_dataset/fix_defense_initial_timestamps.py
```

## 移除 BWR 单包 trace

TrafficSilver-BWR 中存在长度为 1 的 trace。Fine-tune 前运行以下命令，将这些
单包 trace 从 `DF_TrafficSilver_BWR_train.hdf5` 的全部字段中同步移除；脚本完成
校验后原位置替换 HDF5。

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python
"$PYTHON_BIN" pretrain_dataset/fix_bwr_single_packet_traces.py
```

## Defense Fine-tune

微调实现位于 `finetune_defense/`，采用与 Closed-World 相同的单阶段协议：每类固定
抽取 `k` 条训练 trace，其余 trace 全部用于测试。默认 `k=10`。每种 Defense 包含
24 个任务，覆盖无预训练基线，以及 GTT、SwallowDataset 两套预训练模型组合。

默认自动监控并使用 0–3 号空闲 GPU，也可以通过 `GPU_IDS` 指定其他 GPU。脚本会
跳过已经完整生成结果的任务。

### 全部八种 Defense

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh --k 10 --test-slot-strategy all
```

### 前四种 Defense

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-names palette,wtfpad,front,tamaraw \
    --k 10 --test-slot-strategy all
```

以下命令每次只运行一种 Defense，并继续自动分配 GPU。

### Palette

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name palette --k 10 --test-slot-strategy all
```

### WTF-PAD

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name wtfpad --k 10 --test-slot-strategy all
```

### FRONT

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name front --k 10 --test-slot-strategy all
```

### Tamaraw

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name tamaraw --k 10 --test-slot-strategy all
```

### RegulaTor

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name regulator --k 10 --test-slot-strategy all
```

### TrafficSilver Round Robin

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name trafficsilver-rr --k 10 --test-slot-strategy all
```

### TrafficSilver By Direction

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name trafficsilver-bd --k 10 --test-slot-strategy all
```

### TrafficSilver Batched Weighted Random

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_defense/run_all.sh \
    --defense-name trafficsilver-bwr --k 10 --test-slot-strategy all
```

仅查看某一种 Defense 的 24 项任务、不启动训练：

```bash
bash finetune_defense/run_all.sh --defense-name palette --dry-run
```

结果写入 `finetune_defense/results/<defense>/<pretrain-source>/`，日志写入
`finetune_defense/logs/<defense>/<pretrain-source>/`。
