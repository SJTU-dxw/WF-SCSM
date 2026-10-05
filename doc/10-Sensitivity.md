# Parameter Sensitivity

对比基线：

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_scsm_single.yaml`

每个实验只修改标题所示参数，其余参数与基线一致。

## segment_aug_num

### segment_aug_num = 1

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_segment_aug_num_1.yaml`

### segment_aug_num = 3

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_segment_aug_num_3.yaml`

## combine_max

### combine_max = 2

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_combine_max_2.yaml`

### combine_max = 3

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_combine_max_3.yaml`

### combine_max = 4

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_combine_max_4.yaml`

## max_overlap_ratio

### max_overlap_ratio = 0.0

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_max_overlap_ratio_0_0.yaml`

### max_overlap_ratio = 0.1

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_max_overlap_ratio_0_1.yaml`

### max_overlap_ratio = 0.3

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_max_overlap_ratio_0_3.yaml`

## mask_ratio

### mask_ratio = 0.1

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_mask_ratio_0_1.yaml`

### mask_ratio = 0.9

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config parameter_sensitive_pretrain_strategy/swallow_dataset_scsm_single_mask_ratio_0_9.yaml`

## SwallowDataset 预训练权重跨周评测

使用上述 10 个参数敏感性实验和完整 SCSM 基线的 SwallowDataset 预训练权重，在 `Swallow_train.hdf5` 上进行跨周冻结编码器评测。评测协议与 `9-Abaltion.md` 中的 SwallowDataset 跨周评测一致；所有模型均分别评测 `fixed`、`adaptive` 和 `ensemble` 三种推理策略。

基线参数为 `segment_aug_num=2`、`combine_max=5`、`max_overlap_ratio=0.5` 和 `mask_ratio=0.5`，统一由 `swallow_dataset_scsm_single` 表示，因此不为每组重复运行基线。

```bash
CUDA_VISIBLE_DEVICES=0 /nvme/dxw/conda/envs/SCSM/bin/python feature_similarity/evaluate_sensitivity.py \
  --models swallow_dataset_scsm_single \
    swallow_dataset_scsm_segment_aug_num_1 \
    swallow_dataset_scsm_segment_aug_num_3 \
    swallow_dataset_scsm_combine_max_2 \
    swallow_dataset_scsm_combine_max_3 \
    swallow_dataset_scsm_combine_max_4 \
    swallow_dataset_scsm_max_overlap_ratio_0_0 \
    swallow_dataset_scsm_max_overlap_ratio_0_1 \
    swallow_dataset_scsm_max_overlap_ratio_0_3 \
    swallow_dataset_scsm_mask_ratio_0_1 \
    swallow_dataset_scsm_mask_ratio_0_9 \
  --min-trace-length 80 \
  --scsm-slot-strategy all \
  --output-dir feature_similarity/results/sensitivity_swallow_weekly
```

输出保存在 `feature_similarity/results/sensitivity_swallow_weekly/` 下各模型对应的子目录中。

## 推理阶段 ensemble 数量 R

固定使用完整模型 `swallow_dataset_scsm_single`，只改变 ensemble 推理时在
`[min_slot_time, max_slot_time]` 区间内均匀选取的 slot width 数量 R。每个
slot width 分别产生一个编码器特征，最终特征为 R 个特征的算术平均。

默认评测 `R = 2, 3, 5, 7, 9`，其中 `R = 5` 是现有跨周 ensemble 评测的默认值。
不纳入 `R = 1`，因为 `numpy.linspace(min_slot_time, max_slot_time, 1)` 只会选择
区间下界，会同时改变单一 slot 的位置，不能只反映 R 数量的影响。模型权重、
数据划分、随机种子、slot 范围和 kNN 参数均保持一致，不需要重新预训练。

该实验使用独立入口，不修改 `feature_similarity/evaluate.py` 或
`feature_similarity/evaluate_sensitivity.py` 的参数及默认行为：

```bash
CUDA_VISIBLE_DEVICES=0 /nvme/dxw/conda/envs/SCSM/bin/python \
  feature_similarity/evaluate_inference_r_sensitivity.py \
  --r-values 2 3 5 7 9 \
  --min-trace-length 80 \
  --output-dir feature_similarity/results/sensitivity_inference_r
```

各个 R 的原始结果分别保存在 `r_02/`、`r_03/`、`r_05/`、`r_07/` 和
`r_09/` 子目录；根目录的 `weekly_metrics.csv` 汇总所有结果，并包含
`inference_strategy` 和 `inference_r` 字段。`protocol.json` 记录每个 R
实际使用的 slot width。
