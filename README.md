<div align="center">

# CVFusion

**Camera–4D Radar Fusion for 3D Object Detection on View-of-Delft**

[Best Stage1 result](#stage1-result) · [Evaluation](#evaluation) · [Training](#training) · [Technical report](docs/CVFUSION_STAGE1_BEST.md)

</div>

This repository contains an OpenPCDet-based reproduction and engineering
implementation of CVFusion on the View-of-Delft (VoD) dataset. The checked-in
snapshot is the highest independently verified **Stage1** state produced by
this project. It fuses five-frame 4D radar point clouds with camera features
and includes the corresponding checkpoint through Git LFS.

## Stage1 result

The checkpoint was evaluated on all 1,296 samples of the VoD validation split.
Metrics below are 3D AP_R11.

| Class | Evaluation IoU | 3D AP_R11 |
|---|---:|---:|
| Car | 0.50 | 42.3078 |
| Pedestrian | 0.25 | 45.5489 |
| Cyclist | 0.25 | 69.8962 |
| **mAP** | — | **52.5843** |

Final-box recall is 58.4019% at IoU 0.25. The verified optimum is epoch 78,
iteration 50154.

| Artifact | Path |
|---|---|
| Active configuration | `tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml` |
| Resolved experiment configuration | `tools/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid.yaml` |
| Stage1 checkpoint | `output/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid/paper_stage1_zvalid_v1/ckpt/checkpoint_epoch_78.pth` |
| Detailed report | [`docs/CVFUSION_STAGE1_BEST.md`](docs/CVFUSION_STAGE1_BEST.md) |

Checkpoint SHA-256:

```text
a245ad1364f898bfa99a1df5bdbc11cdbf5fd6f2c9543c1e3aec13f28edf45ad
```

## Architecture

```text
Five-frame radar [x,y,z,rcs,v_r,v_r_comp,time]
  -> 0.05 x 0.05 x 0.10 m voxelization + MeanVFE
  -> VoxelBackBone16xRGIter sparse 3D backbone
  -> three radar BEV scales with max-height compression
                                                    \
RGB 384 x 608 -> frozen Swin-T stage 0              +-> RGIter fusion
             -> 64-bin depth distribution -> BEV lift /
  -> aligned 128-channel fused BEV
  -> AnchorHeadSingle
  -> Car / Pedestrian / Cyclist 3D boxes
```

The key correction in this snapshot is **Z-valid camera lifting**. A projected
depth hypothesis is accumulated into BEV only if its full 3D location is
inside the configured detector volume. This prevents image-ray samples above
or below the radar volume from contaminating valid XY cells.

The active Stage1 configuration uses:

- frozen ImageNet-pretrained Swin-T image features;
- three radar scales at approximately 0.2, 0.4 and 0.8 m resolution;
- radar-guided iterative image/radar fusion;
- max pooling along sparse height cells;
- no depth or gate auxiliary loss;
- no experimental height-attention or BEV RPN block;
- no Stage2/RoI head.

## Installation

The verified environment is:

- Python 3.10.14
- PyTorch 2.4.1 with CUDA 12.1
- torchvision 0.19.1
- spconv 2.3.6
- NVIDIA driver 550.67

Install Git LFS before cloning or pull the model afterward:

```bash
git clone https://github.com/tang125810/CVFusion.git
cd CVFusion
git lfs pull

python setup.py develop
```

The pretrained Swin-T file `swin_t-704ceda3.pth` is not redistributed here.
The validated configuration references `../../weights/swin_t-704ceda3.pth`
relative to the repository working directory. Download the torchvision
Swin-T weight and place it there, or update `PRETRAINED` in a copied config.

## Dataset

Prepare the five-frame View-of-Delft radar data in this layout:

```text
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ImageSets/{train,val,test}.txt
  training/{velodyne,label_2,calib,image_2}/
  testing/{velodyne,calib,image_2}/
```

The verified split contains 5,139 training frames and 1,296 validation frames.
Dataset files and generated `.pkl` indexes are intentionally not stored in
Git.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml \
  --batch_size 2 \
  --workers 2 \
  --ckpt output/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid/paper_stage1_zvalid_v1/ckpt/checkpoint_epoch_78.pth \
  --extra_tag stage1_best_verify \
  --eval_tag manual_verify
```

Expected class AP values are `42.3078 / 45.5489 / 69.8962`, giving 52.5843
mAP. Verify the checkpoint hash and dependency versions before investigating
small numerical differences.

## Training

The verified run used four RTX 3090 GPUs, total batch size 8, 80 epochs, a
fixed random seed, and Adam OneCycle with learning rate 0.01.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  tools/train.py \
  --launcher pytorch \
  --cfg_file tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml \
  --batch_size 8 \
  --epochs 80 \
  --workers 2 \
  --ckpt_save_interval 1 \
  --max_ckpt_save_num 30 \
  --extra_tag stage1_best_reproduction
```

Evaluate late-epoch checkpoints on the complete validation set. Training loss
alone is not a reliable model-selection metric for this experiment.

## Repository lineage

This implementation is built on the
[`fthbng77/RadarPillar`](https://github.com/fthbng77/RadarPillar) reproduction,
which in turn is based on
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). The upstream RadarPillar
history is intentionally preserved for attribution and traceability; the
current project entry point and included checkpoint are the CVFusion Stage1
artifacts listed above.

## License

The inherited code is distributed under the Apache 2.0 license. See
[`LICENSE`](LICENSE) and retain the notices and citations required by the
upstream projects.
