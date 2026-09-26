# Stage1 最佳模型：多层图像特征（imgmulti）

本分支（`stage1-imgmulti-best`）在已验证的 Stage1 快照之上，只增加一个**通过验收的单变量改动**，
并保留完整可复现的评测链路。原有 `master` 上的文件与检查点未被替换。

## 1. 结果（官方 VoD devkit 协议 + 精确 float64 旋转 IoU，1296 帧验证集）

| 指标（Entire 全标注区） | 旧最佳 `zvalid` ep78 | **本分支 `imgmulti` ep72** | 差 |
|---|---:|---:|---:|
| **3D AP_R11 mAP** | 52.268 | **56.164** | **+3.895** |
| 3D AP_R40 mAP | 50.807 | 55.538 | +4.731 |
| BEV AP_R11 mAP | 57.236 | 60.878 | +3.642 |
| Car 3D AP_R11 | 42.160 | 42.815 | +0.655 |
| Pedestrian 3D AP_R11 | 45.670 | **52.255** | +6.585 |
| Cyclist 3D AP_R11 | 68.974 | 73.421 | +4.447 |
| **ROI（Driving Corridor）3D AP_R11 mAP** | 69.471 | **79.061** | **+9.590** |
| ROI BEV AP_R11 mAP | 76.786 | 82.968 | +6.182 |
| Recall@0.25 | 58.40% | **62.91%** | +4.51 pt |

与论文 CVFusion Table 1 的 Stage1-only（Entire 59.70 / Corridor 76.66）：

| | 论文 | 旧最佳 | 本分支 | 差距变化 |
|---|---:|---:|---:|---:|
| Entire mAP | 59.70 | 52.27 | **56.16** | -7.43 → **-3.54** |
| Corridor mAP | 76.66 | 69.47 | **79.06** | -7.19 → **+2.40（反超）** |

稳健性：epoch 70–80 的中位数 旧最佳 52.19 / 本分支 **55.06**（+2.87），增益不是单个幸运 checkpoint。

## 2. 唯一改动：图像特征层级

论文只说明图像特征 `F_I` 的分辨率是 `H/4 × W/4`，**未说明 Swin-T 从哪一层截取**。
旧实现取 `features[:2]`（patch embedding + stage 0，96 通道 H/4），本分支改为：完整冻结 Swin-T，
stage 0/1/2/3 各接 1×1 侧向卷积到 96 通道，双线性上采样到 H/4 后拼接，再用 3×3 卷积融回 96 通道。

深度头（96→128→64）、图像投影（96→64）、视角变换、RGIter 融合、anchor、NMS、损失、调度、
随机种子、4 卡 × batch 2 全部保持不变；输出张量形状与旧实现一致 `(B, 96, 96, 152)`。
可训练参数 1.09M → 1.56M，单 epoch 时间 +4%（Swin 仍冻结）。

## 3. 核心文件

| 文件 | 说明 |
|---|---|
| `pcdet/models/backbones_2d/paper_rgiter_fusion.py` | 图像骨干变体（`IMAGE_BACKBONE: stage0/multi`）、散射/采样/关闭三种视角变换、RGIter |
| `pcdet/models/backbones_2d/map_to_bev/multiscale_height_compression.py` | 高度注意力参数改为按需创建（`POOL_METHOD=max` 时不再产生无梯度参数，否则 DDP 报 unused parameter） |
| `tools/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti.yaml` | 本模型配置 |
| `tools/cfgs/vod_models/vod_cvfusion_paper_stage1.yaml` | 其继承的基础配置（未改动） |
| `experiments/stage1_audit_20260926/*.py` | 评测与验证链路：精确 IoU、修正后的预测导出、官方协议评分、几何契约检查、误差分解 |
| `output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1/ckpt/checkpoint_epoch_72.pth` | 最佳权重（Git LFS） |

权重 SHA-256：

```text
76d84177b76b446667a87eccd96f052409f62d7b6cab91ecc3ec5e90d03b8dd2
```

## 4. 复现

```bash
conda activate cvfusion-paper

# 训练（4 卡，80 epochs，与基线同调度）
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  tools/train.py --launcher pytorch \
  --cfg_file tools/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti.yaml \
  --batch_size 8 --epochs 80 --workers 2 \
  --ckpt_save_interval 1 --max_ckpt_save_num 30 \
  --extra_tag paper_stage1_imgmulti_v1

# 官方口径评分（精确 IoU；先导出再用 devkit 协议打分）
python experiments/stage1_audit_20260926/export_predictions_to_kitti.py \
  --det output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1/eval/eval_with_train/epoch_72/val/result.pkl \
  --out experiments/stage1_audit_20260926/pred_imgmulti_ep72
python experiments/stage1_audit_20260926/score_official_protocol.py \
  --pred experiments/stage1_audit_20260926/pred_imgmulti_ep72 --tag imgmulti_ep72 --iou exact

# 逐 epoch 复评（70–80）与自检
python experiments/stage1_audit_20260926/score_epochs_official.py \
  --run output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1 --epochs 70-80
python experiments/stage1_audit_20260926/exact_rotated_iou.py       # IoU 原语自检
python experiments/stage1_audit_20260926/verify_geometry_contract.py --frames 400
```

## 5. 评测口径说明（与旧数字的关系）

本分支所有数字都走**官方 devkit 协议 + 精确 float64 旋转 IoU**。项目自带的 CUDA 旋转 IoU
（RRPN 同源）在近重合框上会高估，同一权重会高出约 **+0.3 mAP**（例如基线官方 52.268 vs
项目口径 52.584）。因此 `master` README 上的 52.5843 应在同口径下读作 52.27。

评测链路自带验收：把原始标签当作预测输入，三个区域、三个类别、R11/R40 的 3D/BEV AP 全为 100。

## 6. 边界

- 本分支证明“把冻结 Swin-T 的多层特征聚合回 H/4”显著优于“只用 stage 0”，即骨干截取位置是真实缺口；
  **不证明**作者使用的就是 1×1 侧向 + 3×3 融合这一具体形式。
- 剩余 Entire 缺口（-3.54）集中在 Car（42.82 vs 论文 52.53）；Corridor 已反超论文。
- 本分支不包含 Stage2（PGF/GGF）改动；两阶段结果与其它候选实验记录在项目技术报告中。
