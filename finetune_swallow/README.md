# SwallowDataset 微调评估

独立入口，不修改 `finetune/`。引用原来的训练循环、数据划分、特征提取、模型和评估函数；适配数据路径、类别对齐、权重路径和 Swallow single checkpoint 校验。

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python bash finetune_swallow/run_all.sh
```

默认使用 cuda:0、cuda:1、cuda:2、cuda:3，k=10、固定 100 个网站。D6 类别 100–103 已在生成 `Swallow_train.hdf5` 时排除；finetune 保留 HDF5 中的全部 trace，不再筛选网站或轨迹长度。每阶段每类随机抽 10 条训练，其余全部测试，所有模型共享同一划分。D2–D6 每阶段训练 1,000 条、测试 9,000 条；D7 训练 1,000 条、测试 14,677 条。

批量脚本中 NetCLR、SCSM 运行不冻结 encoder 的微调，Swallow 同时运行冻结和不冻结 encoder 两种实验，TraVerse 只运行冻结 encoder、训练分类头的实验。保留 SCSM random-slot 开启/关闭两种训练，以及 fixed/adaptive/ensemble 三种测试。6 个阶段共 84 个训练任务，按任务编号轮流分配到 GPU 0–3，每张 GPU 同时运行一个任务；已完成结果会跳过，未完成目录拒绝覆盖。Swallow 冻结实验保存到带 `_freeze` 的结果和日志目录，不冻结实验保存到不带 `_freeze` 的目录。

单个实验：

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_swallow/main.py \
  --model scsm-single --weeks 1 --device cuda:0
```

训练参数沿用 `finetune/main.py`，但默认 k=10，不支持 top-N 网站或长度筛选。单次入口仅接受兼容值 `--website-count 100`、`--min-trace-length 1`，内部关闭 top-N；批量脚本不再提供这两个参数。k 超过实际可用数量、导致没有测试样本时会报错，不丢弃类别。划分缓存包含 k 和随机种子，改变 k 不会沿用旧划分。

- 数据：`pretrain_dataset/Swallow_train.hdf5`，仅 Undefence。
- 权重：`output/pretrain/swallow_dataset/` 下对应模型的目录。
- D1（day=49）用于预训练。评估固定类别 0–99，仅排除 D6 的 100–103；出现其他类别不一致或样本不足时报错，不额外筛选。
- `--weeks 1..6` 对应 D2..D7；每个阶段分别 k-shot 微调，并测试同阶段预留样本，不是跨阶段迁移测试。
- 结果、日志分别在 `finetune_swallow/results/all_remaining/`、`logs/all_remaining/`；划分在 `splits/` 下独立协议缓存中。不与之前固定 5 条测试的实验混用，旧结果和划分保留。

## TraVerse

按 `pretrain/configs/swallow_dataset_traverse.yaml` 训练，权重放在 `output/pretrain/swallow_dataset/traverse/`。需要 `config.json` 和 `encoder.pt`（或 `checkpoint-final.pt`），配置必须对应 `Swallow-TraVerse` 且预训练 `max_day <= 49`。

批量脚本中的 TraVerse 只运行 freeze 实验，冻结 encoder、只训练分类头；缺少配置或权重时正常报错，不自动跳过，也不会回退使用 GTT23 权重。

也可单次运行 `python finetune_swallow/main.py --model traverse --freeze --weeks 1 --device cuda:0`。接口复用原来的 tokenizer、LoRA、特征和分类头逻辑；需先安装对应依赖并准备基础 Llama 模型。

适配器只在当前调用期间临时替换导入的训练模块入口，退出后恢复，不修改源文件或 checkpoint 配置。
