# Stage1 雷达骨干、RPN 与 Anchor 核对报告

日期：2026-09-26
执行依据：《Stage1_四个参考项目的适用性与排查顺序_20260926.md》第 5 节第 5 步“RPN/雷达骨干核对”。
范围：源码级核对 + 数据侧实测统计，不改活动配置、不训练。结论中的“保留/不动”部分与报告第 5 步“不能把其他参考的 CenterHead、circle NMS 直接当成论文 RPN”一致。

## 1. 结论

1. **稀疏骨干与三尺度输出和论文一致。** 体素 `0.05×0.05×0.1`（论文 §4.1.2 明示），SECOND 式稀疏卷积给出 `x_conv3/4/5` 三个尺度，BEV 分辨率为 0.2/0.4/0.8 m（论文 j=2,3,4）。最后一层只在下采样 BEV 平面（z 步长保持 1），避免把已经稀疏的高度信号再砍一半。
2. **高度压缩的垂直量化很粗。** z 方向 50 个体素（0.1 m）经过 stride 2/2/2/1 后变成 25/13/7/7 格，即三个尺度的垂直分辨率是 **0.4 / 0.8 / 0.8 m**，再做 max 压缩。这解释了为什么“高度注意力”实验会被立项，也说明当前融合丢掉了 0.4 m 以下的垂直结构。
3. **检测头是 1×1 卷积，感受野几乎全部来自骨干与融合。** `AnchorHeadSingle` 的 cls/box/dir 都是 1×1；按跳步公式估算，稀疏骨干在 BEV 平面的感受野约 **155 体素 ≈ 7.8 m**，融合再叠一个 0.4 m 分辨率的 3×3（≈ +0.8 m）。因此“RPN 感受野太小”不是当前的主要嫌疑，且项目已实测过加装大型 PV-RCNN 式 BEV RPN 会掉分（`structfix_v1` 52.58→49.75），本次不再重复该方向。
4. **Anchor 正样本覆盖对小目标明显不足（本次最有价值的发现）。**
   - Car：82.8% 的 GT 存在 IoU ≥ 0.6 的 anchor，中位最佳 IoU 0.759；
   - Pedestrian：只有 **60.0%** 的 GT 存在 IoU ≥ 0.5 的 anchor，中位最佳 IoU 仅 **0.535**；
   - Cyclist：只有 **70.9%** 的 GT 存在 IoU ≥ 0.5 的 anchor，中位最佳 IoU 0.570。
   注：`AxisAlignedTargetAssigner` 有 force-matching，每个 GT 仍有 1 个“最佳 anchor”被强制置正；但其余 40%（行人）/29%（骑行者）GT 没有第二个正样本，且被强制置正的那个 anchor 的 IoU 低于匹配阈值，回归目标更远、正样本质量更差。
5. **Anchor 几何是 KITTI 尺寸，与 VoD 数据不匹配。** 实测验证集 GT 尺寸中位数：

   | 类别 | Anchor (l,w,h) | Anchor 底面 z | GT 中位 (l,w,h) | GT 中位底面 z | 偏差 |
   |---|---|---|---|---|---|
   | Car | 3.9, 1.6, 1.56 | -1.78 | 3.98, 1.83, 1.52 | -0.79 | 宽偏小 0.23 m；底面对齐差 1.0 m |
   | Pedestrian | 0.8, 0.6, 1.73 | -0.60 | 0.59, 0.61, 1.68 | -0.73 | **长度偏大 0.21 m（36%）** |
   | Cyclist | 1.76, 0.6, 1.73 | -0.60 | 1.95, 0.73, 1.75 | -0.71 | 长/宽均偏小 |

   行人长度偏大是 BEV IoU 上不去的直接原因：0.8 m 的 anchor 与 0.59 m 的真值在 IoU 意义下最佳只能到 0.59/0.8≈0.74（还要再乘朝向/宽度折扣）。
6. **类别间匹配阈值也偏低。** 行人/骑行者用 0.5，而数据分布下最佳 IoU 中位数只有 0.535/0.570——一半左右的 GT 正好卡在阈值附近，训练信号对 anchor 尺寸非常敏感。
7. **门控饱和（与几何报告同一发现，这里给出机制）。** `W_k = Sigmoid(Conv2d(F_B,k^Rad))` 三尺度全部 ≈1（logit 均值 15.9），论文式(2) 的 `F'_B,k = F_B,k^Cam ⊙ W_k` 退化成恒等。当前活动配置把 `GATE_LOSS_WEIGHT` 设为 0，没有任何项约束 W_k；本次不新增该监督（报告明确禁止用论文外模块补分），但把它记为“已知差异”。
8. **体素上限不会截断本地数据。** 测试期 `MAX_NUMBER_OF_VOXELS: 40000`，而五帧雷达单帧点数在 592～1680 之间（既有审计实测），远低于上限，`mask_points_and_boxes_outside_range` 也不会因上限丢点。

## 2. 逐项核对

### 2.1 体素与稀疏下采样

| 项目 | 值 | 来源 |
|---|---|---|
| 体素 | 0.05 × 0.05 × 0.1 | 配置 `DATA_PROCESSOR.transform_points_to_voxels` |
| 点特征 | 7 列全用（x,y,z,rcs,v_r,v_r_comp,time） | `MeanVFE`，`input_channels=7` |
| 稀疏形状 | z 50 × y 1024 × x 1024（+padding） | `grid_size=(range/voxel)`，`sparse_shape = grid_size[::-1] + [1,0,0]` |
| 通道 | 16 → 32 → 64 → 64 → 64 | `VoxelBackBone16xRGIter` |
| 步长 | 1, 2, 2, 2, (1,2,2) | 同上；最后一层 z 步长为 1 |
| BEV 尺度 | 0.2 / 0.4 / 0.8 m | 0.05 × 4/8/16，对应 `OCCUPANCY_STRIDES` |

### 2.2 高度压缩

`MultiScaleHeightCompression` 对每个尺度的稀疏张量沿 z 做 max（`scatter_reduce(reduce='amax')`，越界填 0 后再 reshape 到 BEV），不做任何垂直卷积。

垂直格数（由跳步推算，配置与代码一致）：50 → 25（conv2）→ 13（conv3）→ 7（conv4）→ 7（conv5）。对应物理分辨率 0.1 → 0.2 → 0.4 → 0.8 → 0.8 m。也就是说送进融合的三个 BEV 特征，其“高度信息”已经被压成 0.4/0.8/0.8 m 的粗粒度，且以 max 而非加权方式聚合。

### 2.3 检测头与感受野

`AnchorHeadSingle` 的 `conv_cls`/`conv_box`/`conv_dir_cls` 都是 kernel=1，因此头的自身感受野就是 1 格。有效感受野来自：

- 稀疏骨干：按 `RF_out = RF_in + (k-1)·jump_in`、`jump_out = jump_in·stride` 逐层推算（k=3），到 `x_conv5` 为约 155 体素 ≈ 7.75 m；
- 融合模块：`image_fuse`/`fuse_conv`/`out_conv` 均为 0.4 m 分辨率上的 3×3，合计约 +1.2 m；
- 相机分支：Swin-T stage 0 在 H/4 上窗口 7（约 7 像素），但视角变换把每个像素沿射线铺开，等效空间支持由深度分布决定。

结论：不认为当前瓶颈是“RPN 感受野不足”，与项目历史实验（加大型 BEV RPN 掉 2.83 mAP）一致。

### 2.4 Anchor 正样本覆盖实测

脚本 `verify_anchor_coverage.py`（400 帧等间隔抽样，1410/1092/475 个 Car/Ped/Cyc GT），复刻 `AnchorGenerator`（`align_center=False`：128×128 网格、间距 `51.2/127 = 0.4031 m`，每类 2 个旋转 × 1 尺寸 × 1 底面高度 = 32768 个 anchor），用与评测同一套精确旋转 IoU 计算 GT 与邻近 anchor 的 BEV IoU。

| 类别 | GT 数 | 最佳 IoU 中位 | ≥ matched_threshold | < unmatched_threshold |
|---|---:|---:|---:|---:|
| Car | 1410 | 0.759 | 0.828（阈值 0.6） | 0.022（0.45） |
| Pedestrian | 1092 | 0.535 | 0.600（阈值 0.5） | 0.033（0.35） |
| Cyclist | 475 | 0.570 | 0.709（阈值 0.5） | 0.017（0.35） |

两点补充：

- anchor 网格间距 0.4031 m 与 BEV 特征网格 0.4 m 不完全重合（OpenPCDet `align_center=False` 的既有约定），存在最多约 0.2 m 的亚格错位；这与论文的 BEV 0.4 m 描述不完全等价，属于已知实现差异。
- 底面高度偏差只影响回归目标（`MATCH_HEIGHT: False`，匹配只用 BEV IoU），不改变正负样本集合。

### 2.5 与论文式(1)-(5)的逐条对照

| 论文 | 实现 | 是否一致 |
|---|---|---|
| (1) `W_k = Sigmoid(Conv2d(F_B,k^Rad))` | `weight_conv[k](radar_bev)` → sigmoid | 一致（但训练后饱和到 1） |
| (2) `F'_B,k = F_B,k^Cam ⊙ W_k` | `camera_scale * weight` | 一致 |
| (3) `F_B,k+1^Cam = Conv2d(F'_B,k, stride=2)` | `iter_conv[k]`（conv stride2+BN+ReLU） | 一致 |
| (4) `F*_B,k = Conv2d([F_B,k^Rad, F'_B,k])` | `fuse_conv[k](cat(...))` | 一致 |
| (5) `F_B = Conv2d([Down(F*_B,0), F*_B,1, Up(F*_B,2)])` | 统一插值到 k=1 后 `out_conv` | 一致 |

## 3. 对后续步骤的输入

- **anchor 覆盖是当前可动的单变量候选**：把三类 anchor 改成 VoD 数据拟合值（尺寸取上表中位数附近，底面高度对齐实测中位），检测头张量形状不变（仍是每类 1 尺寸 × 2 旋转），因此可以单独训练、单独归因。历史上“直接替换 VoD anchors 导致 Car 崩溃”是**复用旧权重**造成（`vodanchors_v1` Car 9.09），本次若做必须从头训练。
- 不建议动的：完整 BEV RPN、CenterPoint 式 head、circle NMS（报告明确排除，项目也已实测掉分）。
- 需要记录的已知差异：门控饱和、anchor 网格 0.4031 vs 0.4 m、底面高度偏差、垂直量化 0.4/0.8 m。

## 4. 复现命令

```bash
cd third_party/RadarPillar
conda activate cvfusion-paper
python experiments/stage1_audit_20260926/verify_anchor_coverage.py --frames 400
# 输出 experiments/stage1_audit_20260926/anchor_coverage.json
```

## 5. 未做与边界

- 未训练、未修改任何配置或权重。
- 感受野为按跳步公式的推算值（代码与配置一致），未做逐层梯度反传实测。
- anchor 覆盖用 400 帧等间隔抽样（约 1/3 验证集），不是全部 1296 帧；三个类别的样本量分别为 1410/1092/475。
- 未评估“anchor 改为 VO D 拟合值”后的训练结果；该项属候选实验，需完整 80 epoch 才能与基线比较。
