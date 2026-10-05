# 预训练数据集

- GTT23
    - 官方提供的zenodo数据，需申请
    ```bash
    aria2c \
    --all-proxy='' \
    --continue=true \
    --max-tries=25 \
    --retry-wait=10 \
    --connect-timeout=30 \
    --timeout=600 \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=64M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --summary-interval=30 \
    --header="Authorization: Bearer ${ZENODO_PAT}" \
    --dir=/nvme/dxw/SCSM/pretrain_dataset \
    --out=GTT23.hdf5 \
    'https://zenodo.org/api/records/10869889/files/GTT23.hdf5/content'
    ```
    - 预处理: `cd pretrain_dataset && python process_GTT.py`
    - 处理为训练所需格式:
        - Swallow: 
            - dataset (特征), label (字符串), day, load_time, slot_duration
            - origin (1000, 20-80), single (2700, 20-80), multi (7200, 40-320)
            - `python pretrain_process_dataset/process_swallow.py --config pretrain_process_dataset/config_pretrain_dataset/GTT_swallow_origin.yaml`
            - `python pretrain_process_dataset/process_swallow.py --config pretrain_process_dataset/config_pretrain_dataset/GTT_swallow_single.yaml`
            - `python pretrain_process_dataset/process_swallow.py --config pretrain_process_dataset/config_pretrain_dataset/GTT_swallow_multi.yaml`
        - NetCLR:
            - dataset (10000 长度的方向序列), label (字符串), day
            - 长序列截取前 10000 个方向，短序列在末尾补 0
            - `python pretrain_process_dataset/process_netclr.py --config pretrain_process_dataset/config_pretrain_dataset/GTT_netclr.yaml`
        - TraVerse:
            - dataset (burst prompt 字符串), label (字符串), day
            - burst 使用带符号的连续同方向包数量，最多保留前 300 个
            - `python pretrain_process_dataset/process_traverse.py --config pretrain_process_dataset/config_pretrain_dataset/GTT_traverse.yaml`
    
- Swallow
    - 作者提供的数据来源:
        - wget https://zenodo.org/records/16607834/files/Swallow-dataset.zip
        - wget https://zenodo.org/records/17861912/files/D6D7-Undefence.rar
    - 预处理: `python pretrain_dataset/process_swallow.py`
    - 以 D1 的 100 个网站为基准筛选各时期；D6 额外的 4 个网站（标签 100-103，共 400 条轨迹）不会写入训练数据。
    - NetCLR:
        - `python pretrain_process_dataset/process_netclr.py --config pretrain_process_dataset/config_pretrain_dataset/swallow_dataset_netclr.yaml`
    - Swallow origin (1000, 20-80):
        - `python pretrain_process_dataset/process_swallow.py --config pretrain_process_dataset/config_pretrain_dataset/swallow_dataset_swallow_origin.yaml`
    - Swallow single (2700, 20-80):
        - `python pretrain_process_dataset/process_swallow.py --config pretrain_process_dataset/config_pretrain_dataset/swallow_dataset_swallow_single.yaml`
    - Swallow multi (7200, 40-320):
        - `python pretrain_process_dataset/process_swallow.py --config pretrain_process_dataset/config_pretrain_dataset/swallow_dataset_swallow_multi.yaml`
    - TraVerse:
        - `python pretrain_process_dataset/process_traverse.py --config pretrain_process_dataset/config_pretrain_dataset/swallow_dataset_traverse.yaml`
    
