# 环境准备

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_24.1.2-0-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p /nvme/dxw/conda && rm miniconda.sh

/nvme/dxw/conda/bin/conda init bash
source ~/.bashrc

conda create -n SCSM python=3.12.1
conda activate SCSM

# pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# mamba
wget https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install mamba_ssm-2.3.0+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install causal_conv1d-1.6.0+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# gdown
pip install gdown==6.1.0
# zenodo-downloader
pip install zenodo-downloader==0.1.0
# zenodo-get
pip install zenodo-get==3.1.0
# hdf5plugin
pip install hdf5plugin==7.0.0
# tensorboard
pip install tensorboard==2.21.0
# peft
pip install peft==0.20.0
# modelscope
pip install modelscope==1.39.1
# timm
pip install timm==1.0.29
# matplotlib
pip install matplotlib==3.11.1
# pandas
pip install pandas==3.0.6
# scipy
pip install scipy==1.18.1
# natsort
pip install natsort==8.4.0
# noise
pip install noise==1.2.2

# 下载模型权重
modelscope download \
  --model LLM-Research/Meta-Llama-3-8B \
  --local_dir /nvme/dxw/SCSM/pretrain/Meta-Llama-3-8B
```
