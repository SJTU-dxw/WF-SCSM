# Multi-tab 实验

## 第一步：准备数据集

### ARES

- 数据来源：[Zenodo 数据](https://zenodo.org/records/14195051)
- 下载并分别解压到独立目录：

  ```bash
  cd pretrain_dataset
  mkdir -p ARES

  for archive in \
      Closed_2tab.zip Closed_3tab.zip Closed_4tab.zip Closed_5tab.zip \
      Open_2tab.zip Open_3tab.zip Open_4tab.zip Open_5tab.zip
  do
      wget "https://zenodo.org/records/14195051/files/${archive}"
      unzip "${archive}" -d "ARES/${archive%.zip}"
  done
  ```

  八个压缩包分别解压到 `pretrain_dataset/ARES/<dataset>/`，不同数据集的文件
  不相互混合。

- 预处理：每个数据集单独生成一个 HDF5。每次只合并对应目录内的
  `train.npz`、`valid.npz` 和 `test.npz`。

  ```bash
  for dataset in \
      Closed_2tab Closed_3tab Closed_4tab Closed_5tab \
      Open_2tab Open_3tab Open_4tab Open_5tab
  do
      python pretrain_dataset/process_multi_data.py \
          --source "pretrain_dataset/ARES/${dataset}" \
          --target "pretrain_dataset/ARES_${dataset}_train.hdf5"
  done
  ```

  `labels` 保留为二维 `uint8` multi-hot 矩阵：Closed 数据集包含 100 个标签，
  Open 数据集包含 101 个标签，其中最后一列为 open-world 类。每条记录恰好有
  与 tab 数相同的正标签。`split` 使用 `0/1/2` 分别记录原始的
  train/valid/test 划分；轨迹按时间戳稳定排序，截断或补零至 10000，所有记录的
  `day` 统一设为 56。

## 第二步：Multi-tab Fine-tune

实现位于 `finetune_multi/`。八个 Closed/Open 2–5 tab 数据集分别作为独立
scenario；每个数据集从全部满足最短长度要求的轨迹中固定随机抽取 `K` 条用于
训练，其余轨迹全部测试。默认 `K=1000`，这里的 `K` 是全局训练样本数，不是
按类别抽取的 k-shot 数。

训练标签直接读取 HDF5 中的 multi-hot 向量，分类头输出维度为 Closed 的 100 或
Open 的 101，损失函数为 `torch.nn.MultiLabelSoftMarginLoss()`。

### 评测指标

按照 ARES 的 multi-tab 协议，`k` 设为当前 scenario 的 tab 数，仅报告 `P@k`
和 `MAP@k`。先分别计算从 `P@1` 到 `P@k` 的全局 precision，`MAP@k` 为这些
precision 的平均值，两个指标均以百分数保存。结果按目标周、scenario、
Closed/Open 设置及 SCSM 测试策略分别保存。

### 单任务运行

无预训练基线示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/main.py \
    --dataset-name closed-2tab \
    --pretrain-source none \
    --model awf \
    --k 1000 \
    --weeks 1 \
    --device cuda:0
```

使用 GTT 预训练权重示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/main.py \
    --dataset-name open-5tab \
    --pretrain-source gtt \
    --model scsm-multi \
    --k 1000 \
    --weeks 1 \
    --device cuda:0 \
    --random-slot \
    --test-slot-strategy all
```

Multi-tab 实验使用对应的 multi 预训练权重：SCSM 从 `scsm_multi/` 加载
`simple_mode=false`、长度 7200 的 checkpoint，Swallow multi 从
`swallow-multi/` 加载并使用 7200-slot CIF。不会使用 `scsm_single/` 或
`swallow-single/`。Swallow origin、NetCLR 和 TraVerse 配置保持不变。

### 批量运行

在 0–3 号 GPU 上运行八个 scenario 的完整模型矩阵：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/run_all.py \
    --gpu-ids 0 1 2 3 \
    --k 1000 \
    --test-slot-strategy all
```

只运行四个 Closed scenario：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/run_all.py \
    --datasets closed-2tab closed-3tab closed-4tab closed-5tab \
    --gpu-ids 0 1 2 3 \
    --k 1000 \
    --test-slot-strategy all
```

只运行四个 Open scenario：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/run_all.py \
    --datasets open-2tab open-3tab open-4tab open-5tab \
    --gpu-ids 0 1 2 3 \
    --k 1000 \
    --test-slot-strategy all
```

只运行单个 scenario，例如 Closed 2-tab：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/run_all.py \
    --datasets closed-2tab \
    --gpu-ids 0 1 2 3 \
    --k 1000 \
    --test-slot-strategy all
```

仅查看任务矩阵，不启动训练：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_multi/run_all.py --dry-run
```

也可在 `--datasets` 后组合任意多个 scenario。结果写入
`finetune_multi/results/<scenario>/<pretrain-source>/<model>/`，日志写入
`finetune_multi/logs/<scenario>/<pretrain-source>/<model>/`。每个结果目录保存
checkpoint、训练历史、仅包含 `P@k`/`MAP@k` 的测试指标，以及逐样本 multi-hot
目标、分数和 top-k 预测。

数据加载器在每个 worker 中复用同一个 HDF5 handle 同时读取轨迹和 multi-hot
标签，并在 200 个训练 epoch 间持久复用 worker，避免反复创建进程和累积文件
描述符。
