"""Score a frozen VoD prediction set with the official devkit AP bookkeeping.

This runs the *official* View-of-Delft protocol (the devkit shipped inside the
SGDet3D reference project, identical in structure to the LXL/R4Det copies):

* IoU thresholds: image 0.7/0.5/0.5, BEV 0.5/0.25/0.25, 3D 0.5/0.25/0.25
  for Car/Pedestrian/Cyclist (the ``overlap_0_5`` row that feeds ``*_3d_all``);
* ``clean_data`` GT rules: ``occluded > 4`` or ``bbox height <= 40 px`` -> ignore
  (DT: ``height < 40 px`` -> ignore);
* regions: ``entire_area`` (custom_method 0), ``roi`` (3, the driving corridor
  x in [-4, 4] and z <= 25), ``not_roi`` (4);
* AP: 11-point interpolated (R11) *and* 40-point (R40).

Everything above comes from the devkit module, imported - not copied - so the
bookkeeping cannot drift.  Only the rotated-IoU primitive is selectable:

``--iou exact``    : :mod:`exact_rotated_iou` (float64 Sutherland-Hodgman).
                     Required for a valid number: with the shipped routine a box
                     compared against itself scores 0.333, so the mandatory
                     "score the GT against itself" acceptance test reports ~0 AP.
``--iou shipped``  : the devkit's own ``rotate_iou_cpu`` (kept for comparison).

Usage::

    python experiments/stage1_audit_20260926/score_official_protocol.py \
        --pred experiments/stage1_audit_20260926/pred_ep78_kitti \
        --tag ep78 --iou exact --self-test --out experiments/stage1_audit_20260926/official_ep78.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROJECT = os.path.dirname(os.path.dirname(REPO))
DEVKIT = os.path.join(PROJECT, '参考项目', 'SGDet3D-main', 'tools_det3d', 'view-of-delft-dataset')

FRAME_LIST = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames/ImageSets/val.txt')
LABEL_DIR = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames/training/label_2')

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
REGION_METHOD = {'entire_area': 0, 'roi': 3, 'not_roi': 4}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True, help='folder of 16-column KITTI txt predictions')
    p.add_argument('--tag', default='run')
    p.add_argument('--iou', choices=['exact', 'shipped'], default='exact')
    p.add_argument('--regions', default='entire_area,roi,not_roi')
    p.add_argument('--limit', type=int, default=0, help='debug: use only the first N frames')
    p.add_argument('--self-test', action='store_true',
                   help='also score the GT label files as if they were predictions')
    p.add_argument('--out', default=None)
    return p.parse_args()


def import_devkit():
    sys.path.insert(0, os.path.join(HERE, '_devkit_stubs'))
    sys.path.insert(0, DEVKIT)
    from vod.evaluation import kitti_official_evaluate as k
    return k


def set_iou_source(kitti_eval, source):
    """Point the devkit's IoU hooks at the exact implementation (or leave them)."""
    if source == 'shipped':
        print('[iou] using the devkit as published (rotate_iou_cpu)')
        return
    import exact_rotated_iou as ex
    kitti_eval.bev_box_overlap = lambda boxes, qboxes, criterion=-1: ex.rotated_iou(
        np.asarray(boxes, dtype=np.float64), np.asarray(qboxes, dtype=np.float64), criterion)
    kitti_eval.d3_box_overlap = lambda boxes, qboxes, criterion=-1: ex.d3_box_overlap(
        np.asarray(boxes, dtype=np.float64), np.asarray(qboxes, dtype=np.float64), criterion)
    print('[iou] using exact float64 rotated IoU (Sutherland-Hodgman)')


def min_overlaps_for(kitti_eval, classes):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.50, 0.50, 0.7, 0.50, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    return np.stack([overlap_0_7, overlap_0_5], axis=0)[:, :, classes]


def read_annos(folder, frames, with_score):
    """Same parsing rules as ``vod.evaluation.evaluation_common.get_label_annotation``."""
    annos = []
    for fid in frames:
        path = os.path.join(folder, fid + '.txt')
        with open(path) as f:
            content = [line.strip().split(' ') for line in f if line.strip()]
        a = {
            'name': np.array([x[0] for x in content]),
            'truncated': np.array([float(x[1]) for x in content]),
            'occluded': np.array([int(x[2]) for x in content]),
            'alpha': np.array([float(x[3]) for x in content]),
            'bbox': np.array([[float(v) for v in x[4:8]] for x in content]).reshape(-1, 4),
            'dimensions': np.array([[float(v) for v in x[8:11]] for x in content]).reshape(-1, 3)[:, [2, 0, 1]],
            'location': np.array([[float(v) for v in x[11:14]] for x in content]).reshape(-1, 3),
            'rotation_y': np.array([float(x[14]) for x in content]).reshape(-1),
        }
        if with_score and len(content) and len(content[0]) == 16:
            a['score'] = np.array([float(x[15]) for x in content])
        else:
            a['score'] = np.zeros([len(a['bbox'])])
        annos.append(a)
    return annos


def score_region(kitti_eval, gt_annos, dt_annos, region, classes=(0, 1, 2)):
    min_overlaps = min_overlaps_for(kitti_eval, list(classes))
    out = kitti_eval.do_eval([dict(a) for a in gt_annos], [dict(a) for a in dt_annos],
                             list(classes), min_overlaps, compute_aos=False,
                             custom_method=REGION_METHOD[region])
    mAPbbox, mAPbev, mAP3d, _, mAPbbox40, mAPbev40, mAP3d40, _ = out
    row = {}
    for j, cls in enumerate(CLASSES):
        row[cls] = {
            '3d_r11': float(mAP3d[j, 0, 1]), '3d_r40': float(mAP3d40[j, 0, 1]),
            'bev_r11': float(mAPbev[j, 0, 1]), 'bev_r40': float(mAPbev40[j, 0, 1]),
            'bbox_r11': float(mAPbbox[j, 0, 1]),
        }
    row['mAP'] = {k: float(np.mean([row[c][k] for c in CLASSES]))
                  for k in ('3d_r11', '3d_r40', 'bev_r11', 'bev_r40', 'bbox_r11')}
    return row


def print_region_row(tag, region, row):
    print('\n=== %s :: %s ===' % (tag, region))
    print('  %-11s %9s %9s %9s %9s' % ('class', '3D R11', '3D R40', 'BEV R11', 'BEV R40'))
    for cls in CLASSES + ['mAP']:
        r = row[cls]
        print('  %-11s %9.4f %9.4f %9.4f %9.4f'
              % (cls, r['3d_r11'], r['3d_r40'], r['bev_r11'], r['bev_r40']))


def build_gt_subset(workdir, frames):
    gt_dir = os.path.join(workdir, 'gt_labels')
    os.makedirs(gt_dir, exist_ok=True)
    for fid in frames:
        shutil.copyfile(os.path.join(LABEL_DIR, fid + '.txt'), os.path.join(gt_dir, fid + '.txt'))
    return gt_dir


def gt_as_prediction(workdir, frames):
    out = os.path.join(workdir, 'perfect')
    os.makedirs(out, exist_ok=True)
    for fid in frames:
        lines = []
        with open(os.path.join(LABEL_DIR, fid + '.txt')) as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 15:
                    lines.append(' '.join(parts[:15]) + ' 1.000000')
        with open(os.path.join(out, fid + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    return out


def main():
    args = parse_args()
    sys.path.insert(0, HERE)
    pred_dir = os.path.abspath(args.pred)
    regions = [r for r in args.regions.split(',') if r]

    with open(FRAME_LIST) as f:
        frames = [line.strip() for line in f if line.strip()]
    if args.limit:
        frames = frames[:args.limit]

    found = sorted(f[:-4] for f in os.listdir(pred_dir) if f.endswith('.txt'))
    if not args.limit and found != sorted(frames):
        raise RuntimeError('prediction folder does not match val.txt (%d vs %d files)'
                           % (len(found), len(frames)))

    kitti_eval = import_devkit()
    set_iou_source(kitti_eval, args.iou)

    report = {'tag': args.tag, 'pred_dir': pred_dir, 'iou': args.iou,
              'frames': len(frames), 'rows': []}
    workdir = tempfile.mkdtemp(prefix='vod_official_')
    try:
        gt_dir = build_gt_subset(workdir, frames)
        gt_annos = read_annos(gt_dir, frames, with_score=False)
        dt_annos = read_annos(pred_dir, frames, with_score=True)
        n_gt = sum(len(a['name']) for a in gt_annos)
        n_dt = sum(len(a['name']) for a in dt_annos)
        print('frames %d | GT objects %d | DT objects %d | IoU=%s' % (len(frames), n_gt, n_dt, args.iou))

        for region in regions:
            t0 = time.time()
            row = score_region(kitti_eval, gt_annos, dt_annos, region)
            print('  [%s] %.1f s' % (region, time.time() - t0))
            print_region_row(args.tag, region, row)
            report['rows'].append({'tag': args.tag, 'region': region, 'iou': args.iou, 'result': row})

        if args.self_test:
            perfect_dir = gt_as_prediction(workdir, frames)
            perfect = read_annos(perfect_dir, frames, with_score=True)
            for region in regions:
                row = score_region(kitti_eval, gt_annos, perfect, region)
                print_region_row('GT-as-prediction', region, row)
                report['rows'].append({'tag': 'GT-as-prediction', 'region': region,
                                       'iou': args.iou, 'result': row})
        out = args.out or os.path.join(HERE, 'official_%s_%s.json' % (args.tag, args.iou))
        with open(out, 'w') as f:
            json.dump(report, f, indent=2)
        print('\nreport written to %s' % out)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
