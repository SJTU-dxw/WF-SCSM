# Open-World 实验

## 第一步：准备数据集

### DF

- 数据来源：Tik-Tok 提供的 [Zenodo 数据](https://zenodo.org/records/11631265/files/Undefended_OW.zip)
- 下载并解压：

  ```bash
  cd pretrain_dataset
  wget https://zenodo.org/records/11631265/files/Undefended_OW.zip
  unzip Undefended_OW.zip
  ```

  解压后数据位于 `pretrain_dataset/open-world-traces-50-40716/`。

- 预处理：合并原 DF 的 95 个 monitored 类别与 Open-World 轨迹；全部
  Open-World 轨迹统一作为第 96 类（标签 `95`）。

  ```bash
  cd pretrain_dataset
  python process_OW.py
  ```

  输入目录为 `raw-data-50-1000/` 和 `open-world-traces-50-40716/`，输出为
  `OW_train.hdf5`。每条轨迹按时间戳稳定排序，保留固定长度 10000（超长截断、
  不足补零），所有样本的 `day` 统一设为 56。

## 第二步：Open-World Fine-tune

实现位于 `finetune_open/`，复用现有模型、特征和训练循环，结果与日志也全部
保存在 `finetune_open/`。

### 数据划分

- 原 DF 标签 `0–94` 为 95 个 monitored 类别，标签 `95` 为唯一的
  unmonitored 类别。
- 每个 monitored 类别按固定随机种子抽取 `k` 条训练轨迹；unmonitored 训练样本
  数量设置为全部 monitored 训练样本的总数，其余有效轨迹全部用于测试。
- 默认 `k=10`、最短轨迹长度为 1。95 个 monitored 类别共抽取 950 条训练轨迹，
  unmonitored 类同样抽取 950 条，因此训练集共 1900 条。测试集共 144546 条，
  包含 104780 条 monitored 与 39766 条 unmonitored 轨迹。
- 所有模型共享 `finetune_open/splits/` 中的持久化划分。

### Open-World 指标

按照 Holmes 的定义设置 `r=20`，对阈值区间 `[0, 1]` 使用 1001 个等距阈值。
模型以 `1 - P(unmonitored)` 作为 monitored 置信度；置信度不低于阈值时接受为
monitored，并在 95 个 monitored 类别中选择概率最大的类别作为网站预测。

每个阈值分别计算：

- `TPR`（recall）：monitored 测试轨迹被正确识别为对应 monitored 网站的比例；
- `WPR`：monitored 测试轨迹被错误识别为其他 monitored 网站的比例；
- `FPR`：unmonitored 测试轨迹被接受为 monitored 的比例；
- 20-precision：`π20 = TPR / (TPR + WPR + 20 × FPR)`。

每个模型或 SCSM 测试策略均输出完整的 20-precision–recall 曲线。JSON 文件保存
阈值、TPR、WPR、FPR、recall、20-precision 和原始计数；NPZ 文件额外保存逐样本
monitored 置信度及预测。

### 单任务运行

无预训练基线示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_open/main.py \
    --pretrain-source none \
    --model awf \
    --k 10 \
    --weeks 1 \
    --device cuda:0
```

使用 GTT 预训练权重示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_open/main.py \
    --pretrain-source gtt \
    --model scsm-single \
    --k 10 \
    --weeks 1 \
    --device cuda:0 \
    --random-slot \
    --test-slot-strategy all
```

### 批量运行

运行全部 baseline、GTT 和 SwallowDataset 预训练配置：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_open/run_all.py \
    --gpu-ids 0 1 2 3 \
    --k 10 \
    --test-slot-strategy all
```

仅查看任务矩阵，不启动训练：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_open/run_all.py --dry-run
```

结果写入 `finetune_open/results/<pretrain-source>/<model>/`，日志写入
`finetune_open/logs/<pretrain-source>/<model>/`。每个结果目录中的
`open_world_metrics*.json` 与 `open_world_predictions*.npz` 为阈值扫描结果。
批量调度器默认监控 0–3 号 GPU，并跳过已完成任务。完整任务矩阵包含 24 个
配置；正式调度前会统一检查预训练 checkpoint，如果某项尚不存在则直接报错且
不会启动训练。补齐 checkpoint 后可使用相同命令重新运行。
