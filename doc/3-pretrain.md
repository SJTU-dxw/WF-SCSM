# 预训练

GTT 权重保存到 `output/pretrain/GTT_dataset/`；SwallowDataset 权重保存到 `output/pretrain/swallow_dataset/`。

- netclr

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_netclr.yaml`

- swallow

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_swallow_origin.yaml`

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_swallow_single.yaml`

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_swallow_multi.yaml`

- traverse

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_traverse.yaml`

- scsm

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_scsm_single.yaml`

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/GTT_scsm_multi.yaml`

## Swallow 数据集

- netclr

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_netclr.yaml`

- swallow

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_swallow_origin.yaml`

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_swallow_single.yaml`

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_swallow_multi.yaml`

- traverse

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_traverse.yaml`

- scsm

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_scsm_single.yaml`

`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 pretrain/main.py --config pretrain/configs/swallow_dataset_scsm_multi.yaml`
