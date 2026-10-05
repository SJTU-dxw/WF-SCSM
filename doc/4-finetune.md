# Fine-tune

```bash
cd /nvme/dxw/SCSM
```

## GTT23 微调

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python bash finetune/run_all.sh
```

## SwallowDataset 微调

```bash
PYTHON_BIN=/nvme/dxw/conda/envs/SCSM/bin/python bash finetune_swallow/run_all.sh --k 10 --test-slot-strategy all
```

## 所有 Week 1 模型跨周测试（GTT23）

自动发现 `finetune/results` 下全部 Week 1 checkpoint（包括所有 k、模型及 freeze/random-slot 变体），分别在 Week 1–6 固定测试集上评测。结果和日志分别写入 `finetune_GTT/results` 与 `finetune_GTT/logs`；默认使用 GPU 0–3，已完成的分周结果会自动跳过。

```bash
/nvme/dxw/conda/envs/SCSM/bin/python finetune_GTT/run_all.py --gpu-ids 0 1 2 3
```
