# Closed-World 实验

## 第一步：准备数据集

### DF

- 数据来源：Tik-Tok 提供的 [Zenodo 数据](https://zenodo.org/records/11631265/files/Undefended.zip)
- 下载并解压：

  ```bash
  cd pretrain_dataset && wget https://zenodo.org/records/11631265/files/Undefended.zip && unzip Undefended.zip
  ```

- 预处理：

  ```bash
  cd pretrain_dataset && python process_DF.py
  ```

  源目录为 `pretrain_dataset/raw-data-50-1000/`，输出为
  `pretrain_dataset/DF_train.hdf5`。每条轨迹按时间戳稳定排序，保留固定长度
  10000（超长截断、不足补零），所有样本的 `day` 统一设为 56。

### R-Precision

- 数据来源：[Zenodo 数据](https://zenodo.org/records/15558538/files/N_Finetune_R-Precision.zip)
- 下载并解压：

  ```bash
  cd pretrain_dataset
  wget https://zenodo.org/records/15558538/files/N_Finetune_R-Precision.zip
  unzip N_Finetune_R-Precision.zip
  ```

- 预处理：将 `N_Finetune_R-Precision/` 目录内的 `finetune.npz` 和
  `val.npz` 合并为训练格式。

  ```bash
  python pretrain_dataset/process_data.py \
      --source pretrain_dataset/N_Finetune_R-Precision \
      --target pretrain_dataset/R-Precision_train.hdf5
  ```

  合并时读取 `X` 中的时间和带符号包长，并读取 `y` 作为标签；有效包按
  时间戳稳定排序，方向与时间同步重排，序列截断或补零至 10000，所有样本的
  `day` 统一设为 56。

### W-T

- 数据来源：[Zenodo 数据](https://zenodo.org/records/14195051/files/W_T.zip)
- 下载并解压：

  ```bash
  cd pretrain_dataset
  wget https://zenodo.org/records/14195051/files/W_T.zip
  mkdir -p W_T
  unzip W_T.zip -d W_T
  ```

- 预处理：合并 `W_T/` 目录内的三个 `.npz` 文件并生成训练格式。

  ```bash
  python pretrain_dataset/process_data.py \
      --source pretrain_dataset/W_T \
      --target pretrain_dataset/W-T_train.hdf5
  ```

  合并时读取 `X` 中的时间和带符号包长，并读取 `y` 作为标签；有效包按
  时间戳稳定排序，方向与时间同步重排，序列截断或补零至 10000，所有样本的
  `day` 统一设为 56。

### k-NN

- 数据来源：[Zenodo 数据](https://zenodo.org/records/14195051/files/k-NN.zip)
- 下载并解压：

  ```bash
  cd pretrain_dataset
  wget https://zenodo.org/records/14195051/files/k-NN.zip
  mkdir -p k-NN
  unzip k-NN.zip -d k-NN
  ```

- 预处理：合并 `k-NN/` 目录内的三个 `.npz` 文件并生成训练格式。

  ```bash
  python pretrain_dataset/process_data.py \
      --source pretrain_dataset/k-NN \
      --target pretrain_dataset/k-NN_train.hdf5
  ```

  合并时读取 `X` 中的时间和带符号包长，并读取 `y` 作为标签；有效包按
  时间戳稳定排序，方向与时间同步重排，序列截断或补零至 10000，所有样本的
  `day` 统一设为 56。

## 第二步：Closed-World Fine-tune

实现位于 `finetune_closed/`，复用 `finetune/` 的模型、特征和训练循环，但不修改
已有的 `finetune/` 与 `finetune_swallow/` 代码。

### 数据划分

- 四个数据集均使用单阶段 Closed-World 协议。
- 每个类别按固定随机种子抽取 `k` 条训练轨迹，其余有效轨迹全部用于测试。
- 所有模型共享 `finetune_closed/splits/<dataset>/` 中的持久化划分。
- 多 GPU 首次并发启动时使用阻塞式文件锁，只由一个进程创建划分，其余进程
  等待完成后复用；进程异常退出时锁会由系统自动释放。
- 默认 `k=10`、最短轨迹长度为 1，不筛选或减少网站类别。

### 模型与预训练权重

- 无预训练基线 AWF、TMWF、ARES、DF、TikTok、VarCNN、RF 和 CountMamba
  每个数据集运行一次，结果放在预训练来源 `none` 下。
- SCSM single、NetCLR、Swallow origin、Swallow single 和 TraVerse 分别使用
  GTT 与 SwallowDataset 两套预训练权重运行。
- SCSM single 保留 random-slot 与 no-random-slot 两个变体，并支持 fixed、
  adaptive、ensemble 或 all 测试策略。
- Swallow origin 与 Swallow single 均运行冻结和不冻结编码器两个变体；
  TraVerse 冻结编码器。
- 当前缺少 `output/pretrain/swallow_dataset/traverse/config.json`；运行完整矩阵前
  需要先生成 SwallowDataset TraVerse 预训练权重。

### 单任务运行

无预训练基线示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_closed/main.py \
    --dataset-name df \
    --pretrain-source none \
    --model awf \
    --k 10 \
    --weeks 1 \
    --device cuda:0
```

使用 GTT 预训练权重示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_closed/main.py \
    --dataset-name r-precision \
    --pretrain-source gtt \
    --model netclr \
    --k 10 \
    --weeks 1 \
    --device cuda:0
```

使用 SwallowDataset 预训练权重示例：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_closed/main.py \
    --dataset-name w-t \
    --pretrain-source swallow \
    --model scsm-single \
    --k 10 \
    --weeks 1 \
    --device cuda:0 \
    --random-slot \
    --test-slot-strategy all
```

### 批量运行

运行四个数据集的完整矩阵：

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_closed/run_all.sh --k 10 --test-slot-strategy all
```

只运行一个数据集：

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
    bash finetune_closed/run_all.sh --dataset-name k-nn --k 10 \
    --test-slot-strategy all
```

仅查看任务矩阵，不启动训练：

```bash
bash finetune_closed/run_all.sh --dry-run
```

结果写入 `finetune_closed/results/<dataset>/<pretrain-source>/`，日志写入
`finetune_closed/logs/<dataset>/<pretrain-source>/`。批量脚本默认动态监控 0–3 号
GPU，并支持通过 `GPU_IDS`、`GPU_MAX_USED_MIB` 和 `GPU_POLL_SECONDS` 调整调度。
