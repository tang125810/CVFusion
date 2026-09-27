# Stage1 评估对拍与旋转 IoU 核验报告

日期：2026-09-26
范围：仅评估侧。使用冻结的 epoch-78 Stage1 预测与冻结的 Stage2 预测，不重新训练、不改动权重、不改动活动配置。
执行依据：《Stage1_四个参考项目的适用性与排查顺序_20260926.md》第 5 节第 1 步“评估对齐”，以及《SGDet3D_数据与评估对照审计_20260926.md》第 3 节遗留的“完整 evaluator 对拍”。

## 1. 结论

1. 官方 VoD evaluator 对拍**已完成并自检通过**：把原始标签当作预测输入，Entire/ROI/not-ROI 三个区域、三个类别、R11 与 R40 的 3D/BEV AP **全部为 100.0000**。在此之前的版本该自检只有 0.19/27.98/26.66，说明当时的对拍结果不可用。
2. 修复了两个真实缺陷，两者都会污染分数：
   - **导出行高宽写反**：`result.pkl` 的 `dimensions` 是 `(l, h, w)`，旧导出脚本按 `(l, w, h)` 解包，导致每个框的高度与宽度互换。Car（h≈1.9、w≈2.0）几乎无损，Pedestrian/Cyclist（h≈1.7、w≈0.6）被压成“躺倒”的薄板，官方口径 3D AP 直接归零。
   - **旋转 IoU 丢失顶点**：devkit 自带的 CPU `rotate_iou_cpu`（与 OpenPCDet 的 CUDA/RRPN 核同源）在角点落到对方边界附近时会漏掉该顶点。同一个框与自身比较得到 **0.3333** 而不是 1.0；对 300 组“近似重合”的车辆框，94 组与精确值的差超过 0.01，最大 0.057。
3. 用**精确 float64 IoU**（Sutherland–Hodgman 裁剪，自带解析解自检）重跑后，官方口径 Stage1 epoch-78 结果为：Entire 3D AP_R11 = Car 42.1602 / Ped 45.6701 / Cyc 68.9743，**mAP 52.2682**；Driving Corridor（roi）**mAP 69.4713**。
4. 历史数字 52.5843 可以精确复现，但它比正确 IoU 下高 **+0.3164 mAP**（Cyclist +0.92、Car +0.15、Pedestrian -0.12）。也就是说既有结论“52.58”应改口径为 **52.27**。项目自身协议（`vod_eval=True`）与官方协议在正确 IoU 下只差 0.0003 mAP，两者差异不来自 ignore 规则。
5. 与论文 Stage1 的差距按官方口径重述：Entire **-7.43 mAP**，Corridor **-7.19 mAP**。论文 Stage1 为 Entire 59.70 / Corridor 76.66。

## 2. 官方协议入口与验证方式

入口固定为 SGDet3D 随附的 View-of-Delft devkit：`参考项目/SGDet3D-main/tools_det3d/view-of-delft-dataset/vod/evaluation`。它不是本地副本，而是被直接 `import` 使用，因此 `clean_data`、11 点插值、ROI 规则不会被改写：

| 规则 | 取值 |
|---|---|
| IoU | 2D 0.7/0.5/0.5，BEV 0.5/0.25/0.25，3D 0.5/0.25/0.25（Car/Ped/Cyc） |
| GT ignore | `occluded > 4` 或 2D 框高 `<= 40 px` |
| DT ignore | 2D 框高 `< 40 px` |
| ROI（Driving Corridor） | `x ∈ [-4, 4]` 且 `z <= 25`，区域外标 ignore，不删除 |
| AP | R11（`*_3d_all`）与 R40 同时输出 |

唯一被替换的是旋转 IoU 原语（见第 4 节）。为保证替换合法，先跑“完美预测”验收：

| 输入 | Entire 3D mAP | ROI 3D mAP | not-ROI 3D mAP |
|---|---:|---:|---:|
| 原始标签作为预测（精确 IoU） | **100.0000** | **100.0000** | **100.0000** |
| 同上（devkit 自带 IoU，修复导出后） | 见 `official_ep78_sub300_shipped.json`，类别级偏差可达 ±4 AP | | |

实测另证：40 px 高度规则在本地数据上几乎不触发——1296 帧中 GT 被忽略 0 个，DT 被忽略 1 个（行人）；所有 2D 框高中位数为 Car 111.7 px、Pedestrian 138.8 px、Cyclist 140.2 px。因此官方协议与项目协议在高规则上等价，分数差异只可能来自 IoU。

## 3. 缺陷一：预测导出的高宽互换

`pcdet/utils/box_utils.py::boxes3d_lidar_to_kitti_camera` 的返回布局是 `[x, y, z, l, h, w, r]`，因此 `result.pkl` 的 `dimensions` 是 `(l, h, w)`。旧导出脚本按 `l, w, h = dimensions` 解包再写 `h w l` 三列，等于把 h 与 w 对调。

| 类别 | 真实 (l,h,w) | 旧导出被解释为 | 3D IoU 后果 |
|---|---|---|---|
| Car | 4.2, 1.9, 2.0 | 4.2, 2.0, 1.9 | 影响小（h≈w） |
| Pedestrian | 0.8, 1.75, 0.6 | 0.8, 0.6, 1.75 | 竖直范围差 3 倍，3D AP → 0 |
| Cyclist | 1.8, 1.7, 0.6 | 1.8, 0.6, 1.7 | 同上 |

这正是此前“官方口径 Pedestrian 0.11、Cyclist 5.29，而 Car 40.5”这一反常现象的全部原因；BEV 几乎不受影响也与该错位一致。

新导出脚本 `experiments/stage1_audit_20260926/export_predictions_to_kitti.py` 在写文件前先用 devkit 自己的解析器做一次往返自检（写入 → 读回 → 比对 dims/loc/ry/score），自检不通过就不产出结果。

## 4. 缺陷二：旋转 IoU 的顶点丢失

`rotate_iou_cpu.quadrilateral_intersection` 用 float32 的平行四边形投影判据判断角点是否在对方框内，判据在角点恰好落在边界时会因 1 ulp 舍入返回 False，而重合边又不会被“线段相交”补回，于是多边形少一个顶点：

```text
[0, 0, 2, 1, 0.5236] 与自身比较：devkit CPU = 0.3333，精确值 = 1.0000
```

对真实预测（不是自身比较）同样有影响。`exact_rotated_iou.py` 的自检第 [5] 项：300 组车辆尺寸、中心扰动 5 cm、角度扰动 0.02 rad 的框对中，**94 组**与精确值相差 > 0.01，最大 0.057。

新的 `exact_rotated_iou.py` 用 Sutherland–Hodgman 凸多边形裁剪在 float64 下计算交集，并用解析解自检：

- 5 个角度下“框与自身”IoU 恒为 1.0；
- 半重叠 0.33333、四分之一重叠 0.14286、不相交 0、2×2 方块与其 45° 旋转 0.70711（=1/√2）；
- 400 组随机框对与独立纯 Python 参考实现最大偏差 < 1e-9；
- 对称性与“交集/(A+B-交集)”一致性通过。

## 5. 对拍结果

全部为 3D AP，官方协议，精确 IoU。命令与报告见第 7 节。

### 5.1 Stage1（epoch 78，`paper_stage1_zvalid_v1`）

| 区域 | 指标 | Car | Pedestrian | Cyclist | mAP |
|---|---|---:|---:|---:|---:|
| Entire | 3D R11 | 42.1602 | 45.6701 | 68.9743 | **52.2682** |
| Entire | 3D R40 | 40.2220 | 42.5809 | 69.6174 | 50.8068 |
| Entire | BEV R11 | 51.2398 | 49.6830 | 70.7852 | 57.2360 |
| ROI | 3D R11 | 71.1727 | 51.5472 | 85.6940 | **69.4713** |
| ROI | 3D R40 | 74.5407 | 50.8019 | 90.3625 | 71.9017 |
| ROI | BEV R11 | 79.8720 | 59.0130 | 91.4728 | 76.7859 |
| not-ROI | 3D R11 | 30.8468 | 33.8426 | 48.6493 | 37.7796 |

### 5.2 完整两阶段（`paper_stage2_pgfg3d_v3c_r11`，RoI NMS 0.90，epoch 16）

| 区域 | 指标 | Car | Pedestrian | Cyclist | mAP |
|---|---|---:|---:|---:|---:|
| Entire | 3D R11 | 49.2105 | 49.7869 | 76.7043 | **58.5672** |
| Entire | 3D R40 | 46.5714 | 50.3108 | 77.2545 | 58.0456 |
| Entire | BEV R11 | 51.8410 | 55.5183 | 77.1114 | 61.4902 |
| ROI | 3D R11 | 72.1690 | 59.7570 | 88.3040 | **73.4100** |
| ROI | 3D R40 | 75.9454 | 61.5185 | 91.5182 | 76.3274 |
| ROI | BEV R11 | 79.5862 | 65.2948 | 88.3197 | 77.7336 |

### 5.3 与论文及历史数字的对照

| 对象 | 口径 | Car | Ped. | Cyc. | mAP |
|---|---|---:|---:|---:|---:|
| 论文 Stage1 | Entire | 52.53 | 50.76 | 75.80 | 59.70 |
| 本报告 Stage1 ep78 | Entire | 42.16 | 45.67 | 68.97 | 52.27 |
| 差值 | Entire | -10.37 | -5.09 | -6.83 | **-7.43** |
| 论文 Stage1 | Corridor | — | — | — | 76.66 |
| 本报告 Stage1 ep78 | ROI | — | — | — | 69.47 |
| 差值 | ROI | — | — | — | **-7.19** |
| 论文完整模型 | Entire | 60.87 | 57.89 | 77.46 | 65.41 |
| 本报告 Stage2 | Entire | 49.21 | 49.79 | 76.70 | 58.57 |
| 差值 | Entire | -11.66 | -8.10 | -0.76 | **-6.84** |
| 论文完整模型 | Corridor | — | — | — | 82.42 |
| 本报告 Stage2 | ROI | — | — | — | 73.41 |
| 差值 | ROI | — | — | — | **-9.01** |

历史 52.5843 与本次 52.2682 的差**不是**协议差，而是 IoU 差。同一份 `result.pkl`：

| 协议 | IoU | Car | Ped. | Cyc. | mAP |
|---|---|---:|---:|---:|---:|
| 项目（`vod_eval=True`） | 项目 CUDA 核（历史） | 42.3078 | 45.5489 | 69.8962 | **52.5843** |
| 项目（`vod_eval=True`） | 精确 float64 | 42.1602 | 45.6692 | 68.9743 | 52.2679 |
| 官方 devkit | 精确 float64 | 42.1602 | 45.6701 | 68.9743 | 52.2682 |

两套协议在正确 IoU 下相差 0.0003 mAP，可视为同一口径；差别集中在 IoU 原语，量级 +0.32 mAP，且类别间可差约 1 AP。此前所有以 0.2 mAP 为“可靠提升”门槛的结论，其分辨率实际上受 IoU 实现影响，建议后续一律以本报告口径复评关键权重。

## 6. 对排查顺序的影响

第 1 步（评估对齐）已闭环，后续实验应统一使用本报告的口径与脚本：

- 报分必须写清“官方 devkit 协议 + 精确 IoU + Entire/ROI + R11/R40”；
- 由于历史数字整体偏高约 0.3 mAP，比较旧结论时要按同口径复评，例如 Step1 目前真实基线是 **52.27**，不是 52.58；
- 完美预测自检（GT-as-prediction = 100）应作为以后每次换 evaluator 或换导出脚本的固定验收项。

## 7. 复现命令

```bash
cd third_party/RadarPillar
conda activate cvfusion-paper

# 0) IoU 原语自检（含 devkit 缺陷的定量证据）
python experiments/stage1_audit_20260926/exact_rotated_iou.py

# 1) 重新导出 epoch-78 预测（含 devkit 解析往返自检）
python experiments/stage1_audit_20260926/export_predictions_to_kitti.py \
  --det output/cfgs/vod_models/vod_cvfusion_active_stage1_best/stage1_best_rollback_verify/eval/epoch_78/val/rollback_20260926/result.pkl \
  --out experiments/stage1_audit_20260926/pred_ep78_kitti

# 2) 官方协议评分（含 GT-as-prediction 自检）
python experiments/stage1_audit_20260926/score_official_protocol.py \
  --pred experiments/stage1_audit_20260926/pred_ep78_kitti \
  --tag ep78 --iou exact --self-test \
  --out experiments/stage1_audit_20260926/official_ep78_exact.json

# 3) 历史口径复现（用于对照，不用于报分）
python experiments/stage1_audit_20260926/score_inhouse_protocol.py \
  --det output/cfgs/vod_models/vod_cvfusion_active_stage1_best/stage1_best_rollback_verify/eval/epoch_78/val/rollback_20260926/result.pkl \
  --iou shipped --tag ep78
```

产出文件（`third_party/RadarPillar/experiments/stage1_audit_20260926/`）：

| 文件 | 内容 |
|---|---|
| `exact_rotated_iou.py` | 精确 float64 旋转 IoU + 3D IoU，模块内自带 5 组自检 |
| `export_predictions_to_kitti.py` | 修正后的 16 列导出，含布局往返自检 |
| `score_official_protocol.py` | 官方 devkit 协议评分器（R11+R40，三区域，可切 IoU） |
| `score_inhouse_protocol.py` | 项目 `vod_eval=True` 协议评分器（用于历史对照） |
| `official_ep78_exact.json` | Stage1 官方口径结果 + GT-as-prediction 自检 |
| `official_stage2_v3c_exact.json` | Stage2 官方口径结果 |
| `inhouse_ep78_shipped.json` / `inhouse_ep78_exact.json` | 历史口径与精确 IoU 对照 |

## 8. 未做与边界

- 未重新推理、未训练、未改动任何权重或活动配置。
- 未重跑 3090 上 1296 帧以外的数据；本报告全部结论限于 VoD validation 1296 帧。
- devkit 自带的 IoU 仍按其原样保留在参考项目目录中，本报告只是不再使用它评分，未修改参考仓库源码。
- not-ROI 区域论文未给对照数字，本报告仅作记录。
