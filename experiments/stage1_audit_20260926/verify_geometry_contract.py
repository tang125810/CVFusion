"""Geometry contract checks for the Stage1 camera/radar pipeline.

The technical report (section 5, step 2) asks for the geometry to be verified
with fixed calibration, fixed features and fixed depth *before* any structural
change is attempted.  This script performs four checks whose failure modes are
concrete and falsifiable:

``A. box -> image``
    Project the 8 corners of every GT 3D box with the original ``P2`` and
    compare with the 2D box in the label file.  A wrong axis order, a wrong
    extrinsic or a wrong principal point collapses this IoU immediately.

``B. radar point -> image``
    Take the raw five-frame radar points that fall inside a GT 3D box and check
    whether they project inside the same object's 2D box.  This validates the
    full chain point -> ``R0 @ V2C`` -> ``P2`` against the labels, independently
    of the detector.

``C. augmentation closed loop``
    In training mode the boxes/points live in the augmented radar frame and the
    camera branch lifts its rays through ``camera_to_lidar @ lidar_aug_matrix``.
    Undoing the augmentation must return each box to a place whose projection
    still matches the (scaled) label box; this is the exact inverse chain the
    view transformation relies on.

``D. view-transformation cell geometry``
    Calling the real ``PaperRGIterFusion._lift_to_bev`` with a one-hot image
    feature and a delta depth distribution must deposit mass in exactly the
    cell predicted by an independent numpy re-derivation; out-of-volume depth
    hypotheses must contribute nothing (the "z-valid" rule).

Usage::

    python experiments/stage1_audit_20260926/verify_geometry_contract.py \
        --frames 200 --out experiments/stage1_audit_20260926/geometry_contract.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from pcdet.utils import box_utils  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402

DATA_ROOT = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames')
CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=int, default=200, help='number of val frames to check')
    p.add_argument('--out', default=os.path.join(HERE, 'geometry_contract.json'))
    p.add_argument('--skip-vt', action='store_true')
    return p.parse_args()


def box_iou_2d(a, b):
    """IoU of two [x1, y1, x2, y2] boxes (single box each)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def read_label(path):
    """KITTI/VoD label rows -> list of dicts (dimensions kept as h, w, l as written)."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 15:
                continue
            rows.append({
                'name': p[0],
                'truncated': float(p[1]),
                'occluded': int(p[2]),
                'bbox': np.array([float(v) for v in p[4:8]]),
                'dims_hwl': np.array([float(v) for v in p[8:11]]),
                'loc': np.array([float(v) for v in p[11:14]]),
                'ry': float(p[14]),
            })
    return rows


def check_a_box_projection(frames, report):
    """GT 3D box corners -> P2 -> compare with the GT 2D box."""
    per_class = {c: [] for c in CLASSES}
    for fid in frames:
        calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', fid + '.txt'))
        for row in read_label(os.path.join(DATA_ROOT, 'training/label_2', fid + '.txt')):
            if row['name'] not in per_class or row['truncated'] > 0.2:
                continue
            l, h, w = row['dims_hwl'][2], row['dims_hwl'][0], row['dims_hwl'][1]
            box_cam = np.concatenate([row['loc'], [l, h, w, row['ry']]])[None, :].astype(np.float32)
            corners = box_utils.boxes3d_to_corners3d_kitti_camera(box_cam)  # (1, 8, 3)
            boxes2d, _ = calib.corners3d_to_img_boxes(corners)
            proj = boxes2d[0]
            if proj[2] - proj[0] <= 1.0 or proj[3] - proj[1] <= 1.0:
                continue  # degenerate / behind camera
            per_class[row['name']].append(box_iou_2d(proj, row['bbox']))
    out = {}
    for cls, values in per_class.items():
        v = np.array(values) if values else np.zeros(1)
        out[cls] = {'n': int(len(values)), 'median_iou': float(np.median(v)),
                    'mean_iou': float(v.mean()), 'frac_gt_0.5': float((v > 0.5).mean())}
    report['A_gt_box_to_image'] = out
    return out


def check_b_points_in_box(frames, report):
    """Raw radar points inside a GT 3D box must project inside its 2D box."""
    stats = {c: {'points': 0, 'inside': 0, 'objects': 0} for c in CLASSES}
    for fid in frames:
        calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', fid + '.txt'))
        points = np.fromfile(os.path.join(DATA_ROOT, 'training/velodyne', fid + '.bin'),
                             dtype=np.float32).reshape(-1, 7)[:, :3]
        pts_cam = calib.lidar_to_rect(points)
        rows = [r for r in read_label(os.path.join(DATA_ROOT, 'training/label_2', fid + '.txt'))
                if r['name'] in CLASSES and r['truncated'] == 0
                and r['bbox'][0] > 2 and r['bbox'][1] > 2
                and r['bbox'][2] < 1934 and r['bbox'][3] < 1214]
        if not rows:
            continue
        boxes_cam = np.stack([
            np.concatenate([r['loc'], [r['dims_hwl'][2], r['dims_hwl'][0], r['dims_hwl'][1], r['ry']]])
            for r in rows]).astype(np.float32)
        boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(boxes_cam, calib)
        pts_img, depth = calib.rect_to_img(pts_cam)
        for i, r in enumerate(rows):
            cx, cy, cz, l, w, h, ry = boxes_lidar[i]
            dx, dy, dz = points[:, 0] - cx, points[:, 1] - cy, points[:, 2] - cz
            c, s = np.cos(-ry), np.sin(-ry)
            lx = c * dx - s * dy
            ly = s * dx + c * dy
            inside_box = ((np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)
                          & (np.abs(dz) <= h / 2) & (pts_cam[:, 2] > 0.5))
            n = int(inside_box.sum())
            if n == 0:
                continue
            u, v = pts_img[inside_box, 0], pts_img[inside_box, 1]
            inside_2d = ((u >= r['bbox'][0]) & (u <= r['bbox'][2])
                         & (v >= r['bbox'][1]) & (v <= r['bbox'][3]))
            stats[r['name']]['points'] += n
            stats[r['name']]['inside'] += int(inside_2d.sum())
            stats[r['name']]['objects'] += 1
    out = {}
    for cls, s in stats.items():
        frac = (s['inside'] / s['points']) if s['points'] else float('nan')
        out[cls] = {'objects': s['objects'], 'points_in_box': s['points'],
                    'points_also_in_2d_box': s['inside'], 'fraction': frac}
    report['B_points_in_box_project_inside_2d'] = out
    return out


def check_c_augmentation_loop(frames, report, seed=0):
    """Training-mode samples: undo the BEV augmentation, then project.

    ``gt_boxes`` in a training sample are ``(N, 8)``: seven box parameters in the
    *augmented* radar frame plus the class id in column 7 (1/2/3 =
    Car/Pedestrian/Cyclist).  Projecting the box centre back through
    ``inv(lidar_aug_matrix)`` then ``inv(camera_to_lidar)`` and the scaled
    network intrinsics must land inside that object's (scaled) label box - the
    same inverse chain ``_lift_to_bev`` relies on.
    """
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils

    cfg_from_yaml_file(os.path.join(REPO, 'tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml'), cfg)
    cfg.LOCAL_RANK = 0
    logger = common_utils.create_logger()
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                          batch_size=1, dist=False, workers=0,
                                          logger=logger, training=True)
    ious = {c: [] for c in CLASSES}
    raw = []
    n_samples = 0
    for batch in loader:
        if n_samples >= frames:
            break
        n_samples += 1
        fid = str(batch['frame_id'][0])
        aug = np.asarray(batch['lidar_aug_matrix'][0], dtype=np.float64)
        cam_to_lidar = np.asarray(batch['camera_to_lidar'][0], dtype=np.float64)
        intrinsics = np.asarray(batch['camera_intrinsics'][0], dtype=np.float64)
        gt_boxes = np.asarray(batch['gt_boxes'][0], dtype=np.float64)
        img_h, img_w = [int(v) for v in batch['image_shape'][0]]
        rows = read_label(os.path.join(DATA_ROOT, 'training/label_2', fid + '.txt'))
        label_by_name = {}
        for r in rows:
            label_by_name.setdefault(r['name'], []).append(r)
        sx, sy = 608.0 / img_w, 384.0 / img_h
        for box in gt_boxes:
            cls_id = int(round(box[7])) if box.shape[0] > 7 else 0
            if not 1 <= cls_id <= len(CLASSES):
                continue
            name = CLASSES[cls_id - 1]
            centre = np.array([box[0], box[1], box[2], 1.0])
            orig = np.linalg.inv(aug) @ centre
            cam = np.linalg.inv(cam_to_lidar) @ orig
            if cam[2] <= 1.0:
                continue
            uv = intrinsics @ cam[:3]
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            best = 0.0
            for r in label_by_name.get(name, []):
                cand = np.array([r['bbox'][0] * sx, r['bbox'][1] * sy,
                                 r['bbox'][2] * sx, r['bbox'][3] * sy])
                if (cand[0] <= u <= cand[2]) and (cand[1] <= v <= cand[3]):
                    best = 1.0
                    break
                best = max(best, box_iou_2d(np.array([u - 4, v - 4, u + 4, v + 4]), cand))
            ious[name].append(best)
            raw.append((name, u, v))
    out = {}
    for cls, values in ious.items():
        v = np.array(values) if values else np.zeros(1)
        out[cls] = {'n': int(len(values)), 'frac_centre_in_label_box': float((v >= 1.0).mean()),
                    'median_proxy_iou': float(np.median(v))}
    report['C_augmentation_closed_loop'] = out
    return out


def check_d_vt_geometry(report, sample_frame='00000'):
    """One-hot feature + delta depth through the real _lift_to_bev."""
    import torch
    from easydict import EasyDict
    from pcdet.models.backbones_2d.paper_rgiter_fusion import PaperRGIterFusion

    model_cfg = EasyDict({
        'IMAGE_SIZE': [384, 608], 'DEPTH_BINS': 64, 'DEPTH_MIN': 1.0, 'DEPTH_MAX': 51.2,
        'IMAGE_CHANNELS': 64, 'OUTPUT_CHANNELS': 128,
        'RADAR_SCALE_CHANNELS': [64, 64, 64],
        'RADAR_SCALE_KEYS': ['radar_bev_0', 'radar_bev_1', 'radar_bev_2'],
        'DEPTH_LOSS_WEIGHT': 0.0, 'GATE_LOSS_WEIGHT': 0.0, 'RPN_NUM_BLOCKS': 0,
        'PRETRAINED': '../../weights/swin_t-704ceda3.pth',
    })
    pc_range = [0.0, -25.6, -3.0, 51.2, 25.6, 2.0]
    model = PaperRGIterFusion(model_cfg, input_channels=64, point_cloud_range=pc_range)
    model.eval()

    calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', sample_frame + '.txt'))
    old_h, old_w = 1216, 1936
    intrinsic = calib.P2[:, :3].astype(np.float64).copy()
    intrinsic[0] *= 608.0 / old_w
    intrinsic[1] *= 384.0 / old_h
    rect = np.eye(4)
    rect[:3, :3] = calib.R0
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :4] = calib.V2C
    cam_to_lidar = np.linalg.inv(rect @ velo_to_cam)

    feat_h, feat_w = 96, 152
    bev_h, bev_w = 256, 256
    results = {}
    failures = []

    # --- geometry used by the independent re-derivation ---
    depths = np.linspace(1.0, 51.2, 64)
    us = (np.arange(feat_w) + 0.5) * (608.0 / feat_w) - 0.5
    vs = (np.arange(feat_h) + 0.5) * (384.0 / feat_h) - 0.5
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 0)
    rays = np.linalg.solve(intrinsic, pix)                        # (3, N)
    x_res = (pc_range[3] - pc_range[0]) / bev_w
    y_res = (pc_range[4] - pc_range[1]) / bev_h
    xyz_cam_all = rays[None, :, :] * depths[:, None, None]        # (D, 3, N)
    xyz_all = np.einsum('ij,djn->din', cam_to_lidar,
                        np.concatenate([xyz_cam_all, np.ones((64, 1, uu.size))], 1))
    gx_all = np.floor((xyz_all[:, 0] - pc_range[0]) / x_res)
    gy_all = np.floor((xyz_all[:, 1] - pc_range[1]) / y_res)
    valid_all = ((gx_all >= 0) & (gx_all < bev_w) & (gy_all >= 0) & (gy_all < bev_h)
                 & (xyz_all[:, 2] >= pc_range[2]) & (xyz_all[:, 2] < pc_range[5]))
    report['D_coverage'] = {
        'in_volume_pairs': int(valid_all.sum()),
        'total_pairs': int(valid_all.size),
        'fraction': float(valid_all.mean()),
        'valid_bins_per_pixel_row': [float(v) for v in
                                     valid_all.reshape(64, feat_h, feat_w).sum(0).mean(1)],
    }

    # pick one in-volume sample per class of behaviour, plus one rejection case
    flat = np.argwhere(valid_all.reshape(64, feat_h * feat_w))
    picks = []
    for want_bin in (5, 20, 60):
        same = flat[flat[:, 0] == want_bin]
        if len(same):
            picks.append((int(same[len(same) // 2, 1]), want_bin, True))
    rejected = np.argwhere(~valid_all.reshape(64, feat_h * feat_w))
    if len(rejected):
        rb = rejected[len(rejected) // 2]
        picks.append((int(rb[1]), int(rb[0]), False))

    for (flat_idx, bin_idx, expect_valid) in picks:
        px, py = flat_idx % feat_w, flat_idx // feat_w
        image_feat = torch.zeros(1, 64, feat_h, feat_w)
        image_feat[0, 0, py, px] = 1.0     # single channel -> mass == deposited weight
        depth_prob = torch.zeros(1, 64, feat_h, feat_w)
        depth_prob[0, bin_idx, py, px] = 1.0
        out = model._lift_to_bev(
            image_feat, depth_prob, torch.tensor(intrinsic[None], dtype=torch.float32),
            torch.tensor(cam_to_lidar[None], dtype=torch.float32),
            torch.eye(4, dtype=torch.float32)[None], bev_h, bev_w)
        nz = (out[0].abs().sum(0) > 0).nonzero().tolist()
        gx, gy = int(gx_all[bin_idx, flat_idx]), int(gy_all[bin_idx, flat_idx])
        xyz_lidar = xyz_all[bin_idx, :, flat_idx]
        ok = (nz == [[gy, gx]]) if expect_valid else (nz == [])
        key = 'pixel_%d_%d_bin_%d' % (px, py, bin_idx)
        results[key] = {
            'expected_cell': [gy, gx], 'expected_in_volume': bool(expect_valid),
            'got_nonzero_cells': nz, 'ok': bool(ok),
            'lidar_xyz': [float(x) for x in xyz_lidar], 'depth': float(depths[bin_idx])}
        if not ok:
            failures.append('D1 ' + key)

    # --- D2: mass conservation over the whole pixel x depth grid ---
    image_feat = torch.zeros(1, 64, feat_h, feat_w)
    image_feat[0, 0] = 1.0                       # one channel only
    depth_prob = torch.zeros(1, 64, feat_h, feat_w)
    depth_prob[0] = 1.0 / 64.0
    out = model._lift_to_bev(
        image_feat, depth_prob, torch.tensor(intrinsic[None], dtype=torch.float32),
        torch.tensor(cam_to_lidar[None], dtype=torch.float32),
        torch.eye(4, dtype=torch.float32)[None], bev_h, bev_w)
    mass = float(out.sum())
    expected_mass = float(valid_all.sum()) / 64.0
    results['z_valid_mass'] = {'expected': expected_mass, 'got': mass,
                               'relative_error': float(abs(mass - expected_mass) / max(expected_mass, 1e-9))}
    if abs(mass - expected_mass) / max(expected_mass, 1e-9) > 1e-4:
        failures.append('D2_z_valid_mass')

    report['D_vt_geometry'] = results
    report['D_failures'] = failures
    return results


def main():
    args = parse_args()
    with open(os.path.join(DATA_ROOT, 'ImageSets/val.txt')) as f:
        all_frames = [line.strip() for line in f if line.strip()]
    # spread the sample over the whole split: the first N frames of val.txt
    # contain almost no cyclists
    stride = max(1, len(all_frames) // max(args.frames, 1))
    frames = all_frames[::stride][:args.frames]
    report = {'frames_checked': len(frames), 'frame_stride': stride, 'data_root': DATA_ROOT}

    print('check A: GT 3D box -> image vs label 2D box')
    a = check_a_box_projection(frames, report)
    for cls, s in a.items():
        print('  %-11s n=%5d median IoU %.3f  frac>0.5 %.3f' % (cls, s['n'], s['median_iou'], s['frac_gt_0.5']))

    print('check B: radar points inside a GT 3D box project inside its 2D box')
    b = check_b_points_in_box(frames, report)
    for cls, s in b.items():
        print('  %-11s objects=%5d points=%6d fraction inside 2D box %.4f'
              % (cls, s['objects'], s['points_in_box'], s['fraction']))

    print('check C: augmentation closed loop (training-mode samples)')
    c = check_c_augmentation_loop(min(args.frames, 100), report)
    for cls, s in c.items():
        print('  %-11s n=%5d centre inside label box %.4f' % (cls, s['n'], s['frac_centre_in_label_box']))

    if not args.skip_vt:
        print('check D: view-transformation geometry (one-hot / delta)')
        d = check_d_vt_geometry(report)
        for key in sorted(k for k in d if k != 'z_valid_mass'):
            print('  %-26s expected cell %s got %s -> %s'
                  % (key, d[key]['expected_cell'], d[key]['got_nonzero_cells'],
                     'ok' if d[key]['ok'] else 'FAIL'))
        z = d['z_valid_mass']
        print('  z-valid mass: expected %.1f got %.1f (rel err %.2e)'
              % (z['expected'], z['got'], z['relative_error']))

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print('\nreport written to %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
