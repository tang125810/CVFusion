"""Anchor positive coverage check for the active Stage1 config.

The technical report's step 5 asks to verify the region-proposal side: height
compression, sparse downsampling, RPN receptive field and anchor positive
coverage.  This script covers the last item with numbers:

* rebuild the exact anchor grid the model uses (``align_center: False``, two
  rotations, one size and one bottom height per class, 128 x 128 at
  ``feature_map_stride`` 8);
* convert every validation GT box to the radar frame exactly the way the
  dataset does (``boxes3d_kitti_camera_to_lidar``: geometric centre, dims
  ``l w h``, heading ``-(ry + pi/2)``);
* compute the rotated BEV IoU between each GT and the anchors near it, then
  report how many GT have at least one anchor above the class's
  ``matched_threshold`` (a GT without one can never produce a positive anchor),
  and how the anchor geometry compares with the data.

Usage::

    python experiments/stage1_audit_20260926/verify_anchor_coverage.py --frames 400
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from exact_rotated_iou import rotated_iou  # noqa: E402
from pcdet.utils import box_utils  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from verify_geometry_contract import DATA_ROOT, read_label  # noqa: E402

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
# mirrors MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG in
# tools/cfgs/vod_models/vod_cvfusion_paper_stage1.yaml
ANCHOR_CFG = {
    'Car': {'sizes': [[3.9, 1.6, 1.56]], 'rotations': [0, 1.57], 'bottom': [-1.78],
            'stride': 8, 'matched': 0.6, 'unmatched': 0.45},
    'Pedestrian': {'sizes': [[0.8, 0.6, 1.73]], 'rotations': [0, 1.57], 'bottom': [-0.6],
                   'stride': 8, 'matched': 0.5, 'unmatched': 0.35},
    'Cyclist': {'sizes': [[1.76, 0.6, 1.73]], 'rotations': [0, 1.57], 'bottom': [-0.6],
                'stride': 8, 'matched': 0.5, 'unmatched': 0.35},
}
PC_RANGE = [0.0, -25.6, -3.0, 51.2, 25.6, 2.0]
VOXEL = [0.05, 0.05, 0.1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frames', type=int, default=400)
    p.add_argument('--out', default=os.path.join(HERE, 'anchor_coverage.json'))
    return p.parse_args()


def build_anchors(cfg):
    """Same maths as AnchorGenerator.generate_anchors with align_center=False."""
    grid = np.round((np.array(PC_RANGE[3:6]) - np.array(PC_RANGE[0:3])) / np.array(VOXEL)).astype(np.int64)
    gx = grid[0] // cfg['stride']
    gy = grid[1] // cfg['stride']
    x_stride = (PC_RANGE[3] - PC_RANGE[0]) / (gx - 1)
    y_stride = (PC_RANGE[4] - PC_RANGE[1]) / (gy - 1)
    xs = PC_RANGE[0] + np.arange(gx) * x_stride
    ys = PC_RANGE[1] + np.arange(gy) * y_stride
    anchors = []
    for size in cfg['sizes']:
        for rot in cfg['rotations']:
            for bottom in cfg['bottom']:
                l, w, h = size
                xx, yy = np.meshgrid(xs, ys, indexing='ij')
                rows = np.stack([xx.ravel(), yy.ravel(),
                                 np.full(xx.size, bottom + h / 2.0),
                                 np.full(xx.size, l), np.full(xx.size, w),
                                 np.full(xx.size, h), np.full(xx.size, rot)], axis=1)
                anchors.append(rows)
    return np.concatenate(anchors, 0), (gx, gy, x_stride, y_stride)


def main():
    args = parse_args()
    with open(os.path.join(DATA_ROOT, 'ImageSets/val.txt')) as f:
        all_frames = [l.strip() for l in f if l.strip()]
    stride = max(1, len(all_frames) // max(args.frames, 1))
    frames = all_frames[::stride][:args.frames]

    anchors = {cls: build_anchors(ANCHOR_CFG[cls]) for cls in CLASSES}
    for cls in CLASSES:
        a, grid = anchors[cls]
        print('%s: %d anchors on a %dx%d grid, spacing %.4f x %.4f m'
              % (cls, a.shape[0], grid[0], grid[1], grid[2], grid[3]))

    best_iou = {cls: [] for cls in CLASSES}
    gt_dims = {cls: [] for cls in CLASSES}
    gt_bottom = {cls: [] for cls in CLASSES}
    radius = 4.0

    for fid in frames:
        calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', fid + '.txt'))
        for row in read_label(os.path.join(DATA_ROOT, 'training/label_2', fid + '.txt')):
            if row['name'] not in CLASSES:
                continue
            h, w, l = row['dims_hwl']
            box_cam = np.concatenate([row['loc'], [l, h, w, row['ry']]])[None, :].astype(np.float32)
            box_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box_cam, calib)[0]
            cx, cy, cz, dl, dw, dh, heading = box_lidar
            if not (PC_RANGE[0] <= cx <= PC_RANGE[3] and PC_RANGE[1] <= cy <= PC_RANGE[4]
                    and PC_RANGE[2] <= cz <= PC_RANGE[5]):
                continue
            gt_dims[row['name']].append([dl, dw, dh])
            gt_bottom[row['name']].append(cz - dh / 2.0)
            a = anchors[row['name']][0]
            near = (np.abs(a[:, 0] - cx) < radius) & (np.abs(a[:, 1] - cy) < radius)
            if not near.any():
                best_iou[row['name']].append(0.0)
                continue
            boxes = a[near][:, [0, 1, 3, 4, 6]].astype(np.float64)   # x, y, l, w, ry
            gt = np.array([[cx, cy, dl, dw, heading]], dtype=np.float64)
            iou = rotated_iou(gt, boxes, -1)[0]
            best_iou[row['name']].append(float(iou.max()))

    summary = {}
    for cls in CLASSES:
        v = np.array(best_iou[cls]) if best_iou[cls] else np.zeros(1)
        dims = np.array(gt_dims[cls]) if gt_dims[cls] else np.zeros((1, 3))
        bottom = np.array(gt_bottom[cls]) if gt_bottom[cls] else np.zeros(1)
        cfg = ANCHOR_CFG[cls]
        summary[cls] = {
            'n_gt': int(len(best_iou[cls])),
            'median_best_iou': float(np.median(v)),
            'mean_best_iou': float(v.mean()),
            'frac_above_matched': float((v >= cfg['matched']).mean()),
            'frac_below_unmatched': float((v < cfg['unmatched']).mean()),
            'anchor_size_lwh': cfg['sizes'][0],
            'anchor_bottom': cfg['bottom'][0],
            'gt_median_lwh': [float(x) for x in np.median(dims, axis=0)],
            'gt_median_bottom_z': float(np.median(bottom)),
            'gt_bottom_p5_p95': [float(np.percentile(bottom, 5)), float(np.percentile(bottom, 95))],
        }
        s = summary[cls]
        print('\n%-11s GT %5d | best anchor IoU median %.3f mean %.3f' %
              (cls, s['n_gt'], s['median_best_iou'], s['mean_best_iou']))
        print('   above matched_threshold %.3f: %.3f | below unmatched %.3f: %.3f'
              % (cfg['matched'], s['frac_above_matched'], cfg['unmatched'], s['frac_below_unmatched']))
        print('   anchor lwh %s bottom %.2f  vs  GT median lwh [%.2f %.2f %.2f] bottom %.2f (p5 %.2f, p95 %.2f)'
              % (cfg['sizes'][0], cfg['bottom'][0], s['gt_median_lwh'][0], s['gt_median_lwh'][1],
                 s['gt_median_lwh'][2], s['gt_median_bottom_z'],
                 s['gt_bottom_p5_p95'][0], s['gt_bottom_p5_p95'][1]))

    with open(args.out, 'w') as f:
        json.dump({'frames': len(frames), 'frame_stride': stride, 'summary': summary}, f, indent=2)
    print('\nreport written to %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
