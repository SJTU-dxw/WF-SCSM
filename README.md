# SCSM: A Traffic-Native Foundation Model for Transferable Website Fingerprinting

[Paper (arXiv)](https://arxiv.org/abs/2610.07776)

This repository contains the implementation of **SCSM**, a traffic-native foundation model for transferable website fingerprinting.

SCSM constructs paired traffic views through **Segmentation, Combination, Scaling, and Masking**, and learns representations from unlabeled traces using a Mamba2-based encoder. The pretrained encoder supports few-shot adaptation across collection times, network conditions, browsers, defenses, and multi-tab browsing, with multi-scale inference.

## Quick evaluation with Hugging Face

[Datasets](https://huggingface.co/collections/2594306528-dxw/scsm-dataset) · [Pretrained models](https://huggingface.co/collections/2594306528-dxw/scsm-model)

After [environment setup](doc/1-environment.md), install the additional dependencies and run the example below from the repository root:

```bash
pip install -U huggingface_hub safetensors scikit-learn
```

This example downloads a pretrained encoder and both datasets. It uses five-scale averaged features and cosine 5-NN, reports Swallow D1-to-D2–D7 reference kNN and reference-centroid accuracy, and computes DF leave-one-out kNN accuracy. All results are macro-averaged across websites; traces shorter than 80 packets are excluded.

```python
import json
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from model.scsm import SCSM
from feature_similarity.evaluate import TraceDataset, extract_embeddings

model_id = "2594306528-dxw/WF-SCSM-Swallow-Single"
folder = Path(snapshot_download(model_id, allow_patterns=["config.json", "model.safetensors"]))
config = json.loads((folder / "config.json").read_text())
model = SCSM(config)
model.load_state_dict(load_file(str(folder / "model.safetensors")))
model = model.cuda().eval()
config.update(slot_strategy="ensemble", min_slot_time=0.005, max_slot_time=0.05,
              test_slot_count=5, time_interval_threshold=0.1, log_transform=True)

datasets = {
    "swallow": ("2594306528-dxw/wf_scsm_swallow", "data/Swallow_train.hdf5"),
    "df": ("2594306528-dxw/wf_scsm_df_closed", "data/DF_train.hdf5"),
}
for name, (repo, filename) in datasets.items():
    path = Path(hf_hub_download(repo, filename, repo_type="dataset"))
    with h5py.File(path, "r") as f:
        days, labels = f["day"][:], f["labels"][:]
        periods = (days.astype(int) - 49) // 7 if name == "swallow" else np.zeros(len(days), int)
        records = [(int(periods[i]), int(i), labels[i])
                   for i in np.flatnonzero(f["lengths"][:] >= 80)]
    loader = DataLoader(TraceDataset(path, records, "scsm", config), batch_size=32)
    embeddings = extract_embeddings(model, loader, torch.device("cuda"), "scsm")
    sites = sorted(site for period, site in embeddings if period == 0)

    def pack(period):
        x = np.concatenate([embeddings[(period, site)] for site in sites])
        y = np.concatenate([np.full(len(embeddings[(period, site)]), i)
                            for i, site in enumerate(sites)])
        return x, y

    x_ref, y_ref = pack(0)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1).fit(x_ref, y_ref)
    if name == "df":
        # X=None excludes each indexed point from its own neighbors.
        print(f"DF kNN: {balanced_accuracy_score(y_ref, knn.predict(None)):.2%}")
        continue

    centroids = normalize(np.stack([embeddings[(0, site)].mean(0) for site in sites]))
    centroid_clf = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(centroids, np.arange(len(sites)))
    for period in range(1, 7):
        x, y = pack(period)
        knn_acc = balanced_accuracy_score(y, knn.predict(x))
        centroid_acc = balanced_accuracy_score(y, centroid_clf.predict(x))
        print(f"D1 -> D{period + 1}: kNN={knn_acc:.2%}, centroid={centroid_acc:.2%}")
```

Change `model_id` to evaluate another released encoder. This compact example uses [scikit-learn's default uniform-vote kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html); its tie handling differs from the similarity-sum tie breaker in the original evaluator. Feature extraction runs on CUDA, while scikit-learn kNN runs on the CPU.

## Documentation

Please refer to the following guides for setup, data preparation, and experiments:

1. [Environment setup](doc/1-environment.md)
2. [Pretraining datasets and preprocessing](doc/2-pretrain_dataset.md)
3. [Pretraining](doc/3-pretrain.md)
4. [Fine-tuning and temporal-drift evaluation](doc/4-finetune.md)
5. [Closed-world experiments](doc/5-Closed-World.md)
6. [Open-world experiments](doc/6-Open-World.md)
7. [Multi-tab experiments](doc/7-Multi-tab.md)
8. [Defended-traffic experiments](doc/8-Defense.md)
9. [Ablation studies](doc/9-Abaltion.md)
10. [Parameter sensitivity](doc/10-Sensitivity.md)

The detailed guides are written in Chinese. Replace machine-specific paths in the guides with your local paths.
