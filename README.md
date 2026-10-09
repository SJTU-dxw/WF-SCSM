# SCSM: A Traffic-Native Foundation Model for Transferable Website Fingerprinting

[Paper (arXiv)](https://arxiv.org/abs/2610.07776)

This repository contains the implementation of **SCSM**, a traffic-native foundation model for transferable website fingerprinting.

SCSM constructs paired traffic views through **Segmentation, Combination, Scaling, and Masking**, and learns representations from unlabeled traces using a Mamba2-based encoder. The pretrained encoder supports few-shot adaptation across collection times, network conditions, browsers, defenses, and multi-tab browsing, with multi-scale inference.

## Quick evaluation with Hugging Face

[Datasets](https://huggingface.co/collections/2594306528-dxw/scsm-dataset) · [Pretrained models](https://huggingface.co/collections/2594306528-dxw/scsm-model)

The example below uses only the Hugging Face releases and standard Python libraries; cloning this repository is not required. A CUDA environment with PyTorch, Mamba, causal-conv1d, and timm is required; see [environment setup](doc/1-environment.md).

```bash
pip install -U "datasets==3.6.0"
```

`datasets==3.6.0` supports the custom dataset loaders used by these releases. Run the following in a fresh Python session. It evaluates all traces with at least 80 packets using five-scale averaged features and cosine 5-NN. Swallow uses D1 as the reference for D2–D7; DF uses leave-one-out kNN. Accuracies are macro-averaged across websites.

```python
import sys
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.metrics import balanced_accuracy_score as accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Model and processor implementations come from the HF model snapshot.
folder = snapshot_download("2594306528-dxw/WF-SCSM-Swallow-Single")
sys.path.insert(0, folder)
from scsm_hub import HubSCSM
from processing_scsm import SCSMProcessor

model = HubSCSM.from_pretrained(folder, map_location="cpu").cuda().eval()
processor = SCSMProcessor.from_pretrained(folder)

@torch.inference_mode()
def encode(dataset):
    features, labels, days = [], [], []
    for sample in tqdm(dataset, desc="Extracting features"):
        if sample["length"] < 80:
            continue
        inputs = processor(sample["times"], sample["directions"],
                           slot_times=np.linspace(0.005, 0.05, 5),
                           time_interval_threshold=0.1, log_transform=True)
        feature = model(inputs["matrix"].cuda(), inputs["last_index"].cuda()).mean(0)
        features.append(feature.cpu().numpy())
        labels.append(sample["label"])
        days.append(sample["day"])
    return normalize(np.stack(features)), np.array(labels), np.array(days)

swallow = load_dataset("2594306528-dxw/wf_scsm_swallow", "all", trust_remote_code=True)["train"]
df = load_dataset("2594306528-dxw/wf_scsm_df_closed", trust_remote_code=True)["train"]

for name, dataset in [("Swallow", swallow), ("DF", df)]:
    x, y, days = encode(dataset)
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    if name == "DF":
        knn.fit(x, y)
        # None excludes each sample from its own neighbors.
        print(f"DF kNN: {accuracy(y, knn.predict(None)):.2%}")
        continue

    reference = days == 49  # D1
    knn.fit(x[reference], y[reference])
    sites = np.unique(y[reference])
    centroids = normalize(np.stack([x[reference & (y == site)].mean(0) for site in sites]))
    centroid_clf = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(centroids, sites)
    for period, day in enumerate([56, 63, 70, 77, 84, 91], start=2):
        test = days == day
        knn_acc = accuracy(y[test], knn.predict(x[test]))
        centroid_acc = accuracy(y[test], centroid_clf.predict(x[test]))
        print(f"D1 -> D{period}: kNN={knn_acc:.2%}, centroid={centroid_acc:.2%}")
```

Change the model ID passed to `snapshot_download` to evaluate another released encoder. This example uses [scikit-learn's default uniform-vote kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html); its tie handling differs from the similarity-sum tie breaker in the original evaluator. Feature extraction runs on CUDA, while scikit-learn kNN runs on the CPU.

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
