# CVFusion Stage1 best model

This snapshot contains the highest verified CVFusion Stage1 model produced in
this repository as of 2026-09-26.

## Result

The checkpoint was evaluated on all 1,296 samples in the VoD validation split.
The reported metric is VoD 3D AP_R11.

| Class | IoU | 3D AP_R11 |
|---|---:|---:|
| Car | 0.50 | 42.3078 |
| Pedestrian | 0.25 | 45.5489 |
| Cyclist | 0.25 | 69.8962 |
| mAP | — | **52.5843** |

The final-box recalls are 0.584019, 0.554993, 0.389593 and 0.127401 at IoU
thresholds 0.25, 0.30, 0.50 and 0.70 respectively.

## Canonical artifacts

- Entry configuration:
  `tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml`
- Resolved experiment configuration:
  `tools/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid.yaml`
- Checkpoint:
  `output/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid/paper_stage1_zvalid_v1/ckpt/checkpoint_epoch_78.pth`
- Checkpoint SHA-256:
  `a245ad1364f898bfa99a1df5bdbc11cdbf5fd6f2c9543c1e3aec13f28edf45ad`
- Checkpoint metadata: epoch 78, iteration 50154,
  `pcdet+0.3.0+eb52eb0`

The checkpoint is stored with Git LFS. Run `git lfs pull` after cloning.

## Active architecture

The active model is a one-stage `SECONDNet` detector with:

- seven-dimensional five-frame radar input
  `[x, y, z, rcs, v_r, v_r_comp, time]`;
- 0.05 x 0.05 x 0.10 m sparse voxels and `MeanVFE`;
- the three-scale `VoxelBackBone16xRGIter` radar backbone;
- channel-wise max height compression at 0.2, 0.4 and 0.8 m BEV scales;
- a frozen pretrained Swin-T stage-0 image backbone;
- 64-bin camera depth lifting with full x/y/z volume validation;
- radar-guided iterative camera/radar BEV fusion;
- a direct 128-channel `AnchorHeadSingle` proposal path.

Depth and gate auxiliary losses are disabled. Height attention, the
experimental BEV RPN and every Stage2/RoI module are inactive. In particular,
the z-valid camera lift rejects hypotheses outside `[-3, 2)` metres in height;
this prevents invalid parts of an image ray from being collapsed into valid XY
BEV cells.

## Environment used for verification

- Python 3.10.14
- PyTorch 2.4.1 with CUDA 12.1
- torchvision 0.19.1
- spconv 2.3.6
- 4 x NVIDIA RTX 3090 for training; one RTX 3090 for verification

The frozen image backbone expects `../../weights/swin_t-704ceda3.pth` relative
to the RadarPillar working directory configuration. Dataset files and the
Swin-T pretrained weights are not included in this snapshot.

## Evaluation

From the repository root:

```bash
conda activate cvfusion-paper

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml \
  --batch_size 2 \
  --workers 2 \
  --ckpt output/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid/paper_stage1_zvalid_v1/ckpt/checkpoint_epoch_78.pth \
  --extra_tag stage1_best_verify \
  --eval_tag manual_verify
```

The expected class AP values are `42.3078 / 45.5489 / 69.8962`. Check the
checkpoint hash, dependency versions and configuration before investigating
small numerical differences.

## Training

The verified run used four GPUs, batch size 2 per GPU, 80 epochs, fixed random
seed, Adam OneCycle with learning rate 0.01 and weight decay 0.01.

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

Evaluate every late-epoch checkpoint on the complete validation set. The
lowest training loss or an automatically named `checkpoint_best.pth` is not a
reliable substitute for the measured AP_R11; epoch 78 is the verified optimum
for this run.
