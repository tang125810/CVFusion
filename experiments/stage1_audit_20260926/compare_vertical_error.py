"""Why did the HEIGHT_STATS run lose its early lead?  Compare the vertical error.

Both runs are at the same training stage (epoch 30), so comparing the Car
bottom-height error distribution of their predictions separates two stories:

  * the extra vertical channels improved the vertical regression but the score
    dropped anyway -> the loss sits in ranking/classification, not in z;
  * the vertical error is unchanged -> the cue was never used, i.e. the BEV
    features already carried enough vertical information and the bottleneck is
    elsewhere.

Usage::
    python experiments/stage1_audit_20260926/compare_vertical_error.py \
        --runs a=/path/result.pkl b=/path/result.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '_devkit_stubs'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))),
                                '参考项目', 'SGDet3D-main', 'tools_det3d', 'view-of-delft-dataset'))

from exact_rotated_iou import d3_box_overlap, rotated_iou  # noqa: E402
from score_official_protocol import FRAME_LIST, LABEL_DIR, read_annos  # noqa: E402


def boxes_of(anno, cls):
    m = np.zeros(len(anno['name']), dtype=bool)
    if len(anno['name']):
        m = np.asarray(anno['name']) == cls
    if m.sum() == 0:
        return np.zeros((0, 7))
    return np.concatenate([anno['location'][m], anno['dimensions'][m],
                           anno['rotation_y'][m][:, None]], axis=1).astype(np.float64)


def stats(det_path, gt_annos, frames, cls='Car', radius=4.0):
    with open(det_path, 'rb') as f:
        det = pickle.load(f)
    by = {str(a['frame_id']): a for a in det}
    errs, best3, bestbev, dists = [], [], [], []
    for i, fid in enumerate(frames):
        g = boxes_of(gt_annos[i], cls)
        d = boxes_of(by[fid], cls)
        if len(g) == 0 or len(d) == 0:
            continue
        i3 = d3_box_overlap(g, d, -1)
        ib = rotated_iou(g[:, [0, 2, 3, 5, 6]], d[:, [0, 2, 3, 5, 6]], -1)
        gc = g[:, [0, 2]]
        for k in range(len(g)):
            dist = np.linalg.norm(gc[k][None, :] - d[:, [0, 2]], axis=1)
            j = int(np.argmin(dist))
            if dist[j] > radius:
                continue
            errs.append(d[j][1] - g[k][1])            # camera-frame bottom y
            best3.append(i3[k, j]); bestbev.append(ib[k, j]); dists.append(g[k][2])
    errs = np.array(errs, dtype=float)
    best3 = np.array(best3, dtype=float)
    bestbev = np.array(bestbev, dtype=float)
    dists = np.array(dists, dtype=float)
    if errs.size:
        print_bins(errs, bestbev, dists)
    return {
        'n': len(errs),
        'bias_median': float(np.median(errs)),
        'spread_p10_p90': [float(np.percentile(errs, 10)), float(np.percentile(errs, 90))],
        'mad': float(np.median(np.abs(errs - np.median(errs)))),
        'abs_gt_0.3_frac': float(np.mean(np.abs(errs) > 0.3)),
        'car3d_ge_0.5_frac': float(np.mean(best3 >= 0.5)),
        'carbev_ge_0.5_frac': float(np.mean(bestbev >= 0.5)),
        'far_frac_25m': float(np.mean(dists > 25)),
    }


def print_bins(errs, bestbev, dists):
    print('  %-10s %6s %12s %10s %10s %11s' % ('distance', 'n', 'bias', 'MAD', '|e|>0.3', 'BEV>=0.5'))
    for lo, hi in ((0, 15), (15, 25), (25, 35), (35, 60)):
        m = (dists >= lo) & (dists < hi)
        if m.sum() == 0:
            continue
        e = errs[m]
        print('  %-10s %6d %+12.3f %10.3f %9.1f%% %10.1f%%'
              % ('%d-%dm' % (lo, hi), int(m.sum()), np.median(e),
                 np.median(np.abs(e - np.median(e))), 100 * np.mean(np.abs(e) > 0.3),
                 100 * np.mean(bestbev[m] >= 0.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='label=result.pkl')
    args = ap.parse_args()
    with open(FRAME_LIST) as f:
        frames = [l.strip() for l in f if l.strip()]
    gt_annos = read_annos(LABEL_DIR, frames, with_score=False)
    print('%-14s %5s %9s %16s %8s %9s %10s %10s' % (
        'run', 'n', 'bias', 'p10..p90', 'MAD', '|err|>0.3', '3D>=0.5', 'BEV>=0.5'))
    for item in args.runs:
        label, path = item.split('=', 1)
        s = stats(path, gt_annos, frames)
        print('%-14s %5d %+9.3f  %+7.3f..%+6.3f %8.3f %8.1f%% %9.1f%% %9.1f%%' % (
            label, s['n'], s['bias_median'], s['spread_p10_p90'][0], s['spread_p10_p90'][1],
            s['mad'], 100 * s['abs_gt_0.3_frac'], 100 * s['car3d_ge_0.5_frac'],
            100 * s['carbev_ge_0.5_frac']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
