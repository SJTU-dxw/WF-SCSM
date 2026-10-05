# Ablation

对比基线：

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_scsm_single.yaml`

## 去除 Segmentation

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/swallow_dataset_scsm_single_no_segmentation.yaml`

## 去除 Combination

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/swallow_dataset_scsm_single_no_combination.yaml`

## 去除 Scaling

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/swallow_dataset_scsm_single_no_scaling.yaml`

## 去除 Masking

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/swallow_dataset_scsm_single_no_masking.yaml`


## Mask 单独有效性对照

以下两个实验采用相同的数据、模型、优化器、训练轮数和随机种子，并都关闭 Segmentation、Combination 与 Scaling。唯一变量是是否启用 Mask，因此应直接比较 `Mask only` 与 `无任何增强`，用来判断 Mask 本身是否优于无增强基线。`log_transform` 属于固定输入预处理，在两组中保持启用。

### Mask only

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/swallow_dataset_scsm_single_mask_only.yaml`

### 无任何增强

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/swallow_dataset_scsm_single_no_augmentation.yaml`

该结论应以两组使用同一评测协议后的结果为准；若 `Mask only` 优于 `无任何增强`，可说明 Mask 单独具有正向作用。完整模型中去除 Mask 后的轻微提升则表明 Mask 与其他增强组合时可能存在负交互，而不是 Mask 本身无效。

## GTT 消融

GTT 对比基线：

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_scsm_single.yaml`

### 去除 Segmentation

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/GTT_scsm_single_no_segmentation.yaml`

### 去除 Combination

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/GTT_scsm_single_no_combination.yaml`

### 去除 Scaling

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/GTT_scsm_single_no_scaling.yaml`

### 去除 Masking

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/GTT_scsm_single_no_masking.yaml`

### Mask 单独有效性对照

以下两个实验均关闭 Segmentation、Combination 与 Scaling，唯一变量是是否启用 Mask；`log_transform` 在两组中保持启用。

#### Mask only

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/GTT_scsm_single_mask_only.yaml`

#### 无任何增强

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config ablation_pretrain_strategy/GTT_scsm_single_no_augmentation.yaml`

## SwallowDataset 预训练权重跨周评测

使用 `3-pretrain.md` 中的 SwallowDataset 预训练权重，在 `Swallow_train.hdf5` 上对完整 SCSM、四个单模块消融模型、`Mask only`、`无任何增强` 及 Swallow-origin、NetCLR、TraVerse 三种预训练基线进行跨周评测。编码器全程冻结，不进行微调。

时间划分为：Day 49 作为 reference；Day 50–56、57–63、64–70、71–77、78–84 和 85–91 分别作为 Week 1–6。每个 week 分别报告：

- `与参考时期质心余弦相似度`
- `同时期样本平均余弦相似度`
- `质心分类准确率 (macro)`
- `与参考时期样本的 KNN-5 macro accuracy`
- `同时期样本的 KNN-5 macro accuracy`

完整 SCSM、四个单模块消融模型和两个 Mask 对照模型都分别评测 `fixed`、`adaptive` 和 `ensemble` 三种推理策略。NetCLR、Swallow-origin 和 TraVerse 使用各自原生输入协议。

```bash
CUDA_VISIBLE_DEVICES=0 /nvme/dxw/conda/envs/SCSM/bin/python feature_similarity/evaluate.py \
  --models swallow_dataset_scsm_single \
    swallow_dataset_scsm_no_segmentation \
    swallow_dataset_scsm_no_combination \
    swallow_dataset_scsm_no_scaling \
    swallow_dataset_scsm_no_masking \
    swallow_dataset_scsm_mask_only \
    swallow_dataset_scsm_no_augmentation \
    swallow_dataset_swallow_origin \
    swallow_dataset_netclr \
    swallow_dataset_traverse \
  --min-trace-length 80 \
  --scsm-slot-strategy all \
  --output-dir feature_similarity/results/ablation_swallow_weekly
```

输出保存在 `feature_similarity/results/ablation_swallow_weekly/` 下各模型对应的子目录中。

## GTT 预训练权重跨周评测

使用 GTT23 预训练权重，在 `GTT23_train.hdf5` 上对完整 SCSM、四个单模块消融模型、`Mask only`、`无任何增强` 及 Swallow-origin、NetCLR、TraVerse 三种预训练基线进行跨周评测。编码器全程冻结，不进行微调。

时间划分和评测指标与 SwallowDataset 跨周评测一致。首先只保留长度不少于 80 的 trace；随后筛选 reference 样本数不少于 1000、每个预测周样本数不少于 25 的网站，再按全部有效样本总数选取前 200 个网站。每个网站在每个时期最多抽取 1000 条样本，默认使用随机种子 0，所选网站写入输出目录的 `selected_websites_gtt.json`，便于复现实验。

完整 SCSM、四个单模块消融模型和两个 Mask 对照模型都分别评测 `fixed`、`adaptive` 和 `ensemble` 三种推理策略。NetCLR、Swallow-origin 和 TraVerse 使用各自原生输入协议。

```bash
CUDA_VISIBLE_DEVICES=0 /nvme/dxw/conda/envs/SCSM/bin/python feature_similarity/evaluate_GTT.py \
  --models GTT_scsm_single \
    GTT_scsm_no_segmentation \
    GTT_scsm_no_combination \
    GTT_scsm_no_scaling \
    GTT_scsm_no_masking \
    GTT_scsm_mask_only \
    GTT_scsm_no_augmentation \
    GTT_swallow_origin \
    GTT_netclr \
    GTT_traverse \
  --min-trace-length 80 \
  --website-count 200 \
  --scsm-slot-strategy all \
  --output-dir feature_similarity/results/ablation_GTT_weekly
```

输出保存在 `feature_similarity/results/ablation_GTT_weekly/` 下各模型对应的子目录中。

## SwallowDataset 微调策略消融

实验对象为 SCSM-single，沿用 `finetune_swallow` 的数据划分、优化参数和 6 个 week 的 k-shot 协议。微调阶段的 Scaling 对应 `--random-slot`：开启时，每个训练样本从预训练配置的 `[min_slot_time, max_slot_time]` 中随机采样 slot width；关闭时使用固定 slot width。无预训练实验保持相同 SCSM-single 架构和输入协议，但仅随机初始化模型，不读取 `encoder.pt` 或其他 checkpoint 权重。

### 无预训练 × 有/无微调 Scaling（需要新增运行）

以下命令一次运行随机初始化的两种微调策略，并在推理时同时输出 fixed、adaptive 和 ensemble 三套结果：

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
  bash ablation_finetune_strategy/run_all.sh \
  --k 10 \
  --initialization scratch \
  --test-slot-strategy all
```

结果和日志分别写入 `ablation_finetune_strategy/results/scratch/` 与 `ablation_finetune_strategy/logs/scratch/`。带 `_randomslot_alltest` 的结果是“微调采用 Scaling”，带 `_alltest` 的结果是“微调不采用 Scaling”。

### 已有的有预训练结果与推理策略消融

现有结果已经覆盖“有预训练 × 微调采用/不采用 Scaling”，且两个训练设置的 6 个 week 均已有 fixed、adaptive、ensemble 三种推理结果：

- 有 Scaling：`finetune_swallow/results/all_remaining/scsm-single/k10_seed0_minlen1_randomslot_alltest/`
- 无 Scaling：`finetune_swallow/results/all_remaining/scsm-single/k10_seed0_minlen1_alltest/`

各 week 的 `test_metrics_fixed.json` 是 fixed 推理，`test_metrics_ensemble.json` 是多尺度集成推理，可直接用这两份文件完成“多尺度还是 fixed”的消融比较；`test_metrics_adaptive.json` 还提供了逐 trace 自适应选择单一尺度的附加对照。因此这一项无需重新训练。

## GTT 微调策略消融

以下命令使用随机初始化的 SCSM-single，按 GTT 的 200 网站、最短 trace 长度 80、6 个 week 和 k=10 协议，同时运行微调有/无 Scaling，并输出 fixed、adaptive 和 ensemble 三套推理结果：

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python \
  bash ablation_finetune_strategy2/run_all.sh \
  --k 10 \
  --initialization scratch \
  --test-slot-strategy all
```

结果和日志分别写入 `ablation_finetune_strategy2/results/scratch/` 与 `ablation_finetune_strategy2/logs/scratch/`。
