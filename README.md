# Structure-Preserving Patch Decoding for Efficient Neural Video Representation

[![MMSP 2025](https://img.shields.io/badge/MMSP-2025-lightgray)](https://attend.ieee.org/mmsp-2025/)
[![IEEE Xplore](https://img.shields.io/badge/IEEE_Xplore-17445A)](https://ieeexplore.ieee.org/document/11324200)
[![arXiv](https://img.shields.io/badge/arXiv-2506.12896-b31b1b)](https://arxiv.org/abs/2506.12896)

This repository provides training code for the paper "Structure-Preserving Patch Decoding for Efficient Neural Video Representation".

## 1. Setup

```bash
git clone https://github.com/th359/SPP-NeRV.git
cd SPP-NeRV
conda create -n spp-nerv python=3.8
conda activate spp-nerv
pip install -r requirements.txt
```

## 2. Project structure

```text
.
|-- train_nerv_all.py
|-- scripts/
|   |-- run_Bunny.sh
|   |-- run_DAVIS.sh
|   |-- run_UVG.sh
|   `-- run_MCL-JCV.sh
`-- dataset/   # prepare by yourself
```

## 3. Dataset layout expected by scripts

```text
dataset/
  bunny/
    00000.png
    00001.png
    ...
  UVG/
    Beauty/video/
      00000.png
      ...
    Bosphorus/video/
      ...
  DAVIS-data/
    DAVIS/JPEGImages/1080p/
      bear/
      blackswan/
      ...
  MCL-JCV/
    images/
      1080PAVCFQPvideoSRC01/
      1080PAVCFQPvideoSRC02/
      ...
```

Each `--data_path` directory should contain frame images.

## 4. Run training scripts

```bash
bash scripts/run_Bunny.sh
bash scripts/run_UVG.sh
bash scripts/run_DAVIS.sh
bash scripts/run_MCL-JCV.sh
```

## 5. How to customize

Current scripts are intentionally simple and do not use CLI options.
To change settings, edit variables at the top of each script:

- `data_path`
- `out_path`
- `epoch` or `epoch_list`
- `model_size` or `model_size_list`
- `video_list`

Common training arguments are grouped in `common_args` in each script.

## 6. Default presets in scripts

- `run_Bunny.sh`
  - epochs: `50 100 150 200 250 300`
  - model size: `1.5`
  - data path: `./dataset/bunny`
- `run_DAVIS.sh`
  - epoch: `300`
  - model size: `1.5`
  - videos: DAVIS 1080p list defined in script
  - data path: `./dataset/DAVIS-data/DAVIS/JPEGImages/1080p`
- `run_UVG.sh`
  - epoch: `150`
  - model sizes: `3.0 5.0 8.0 10.0`
  - videos: `Beauty Bosphorus HoneyBee Jockey ReadySetGo YachtRide ShakeNDry`
  - data path: `./dataset/UVG`
- `run_MCL-JCV.sh`
  - epoch: `300`
  - model size: `1.5`
  - videos: `01` to `30`
  - data path: `./dataset/MCL-JCV/images`

## Acknowledgement

Our code is based on [Boosting-NeRV](https://github.com/Xinjie-Q/Boosting-NeRV).

## Contact

hayatai17@fuji.waseda.jp
