# SCSM: A Traffic-Native Foundation Model for Transferable Website Fingerprinting

[Paper (arXiv)](https://arxiv.org/abs/2610.07776)

This repository contains the implementation of **SCSM**, a traffic-native foundation model for transferable website fingerprinting.

SCSM constructs paired traffic views through **Segmentation, Combination, Scaling, and Masking**, and learns representations from unlabeled traces using a Mamba2-based encoder. The pretrained encoder supports few-shot adaptation across collection times, network conditions, browsers, defenses, and multi-tab browsing, with multi-scale inference.

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
