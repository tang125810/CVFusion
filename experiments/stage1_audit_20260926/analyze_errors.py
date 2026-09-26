"""Error decomposition for a frozen Stage1 prediction set.

The reproduction is short of the paper mostly on Car (-10.4 AP on the entire
area), so a single mAP number is not enough to decide what to change.  This
script matches predictions to ground truth greedily by score at the official
per-class 3D IoU thresholds and reports:

* AP by forward-distance bucket (0-15, 15-25, 25-35, 35-50 m) using the same
  R11 protocol as the rest of the audit;
* for each class: TP / FP / FN counts, and how the FNs split into
  "no prediction nearby" (best IoU < 0.1), "localisation failure"
  (0.1 <= best IoU < threshold) and "matched but outranked";
* for true positives: the distribution of centre error (lateral / vertical /
  forward), size error and yaw error;
* for false positives: how many overlap a GT of a *different* class above that
  class's threshold (classification confusion) rather than empty space.

Usage::

    python experiments/stage1_audit_20260926/analyze_errors.py \
        --pred experiments/stage1_audit_20260926/pred_ep78_kitti \
        --tag ep78 --out experiments/stage1_audit_20260926/errors_ep78.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '_devkit_stubs'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(REPO)), '参考项目', 'SGDet3D-main',
                                'tools_det3d', 'view-of-delft-dataset'))

from exact_rotated_iou import d3_box_overlap  # noqa: E402
from score_official_protocol import FRAME_LIST, LABEL_DIR, read_annos  # noqa: E402

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
IOU_THRESH = {'Car': 0.5, 'Pedestrian': 0.25, 'Cyclist': 0.25}
DIST_BUCKETS = [(0, 15), (15, 25), (25, 35), (35, 50)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True)
    p.add_argument('--tag', default='run')
    p.add_argument('--out', default=None)
    return p.parse_args()


def boxes_of(anno, cls):
    mask = np.zeros(len(anno['name']), dtype=bool)
    if len(anno['name']):
        mask = np.asarray(anno['name']) == cls
    if mask.sum() == 0:
        return np.zeros((0, 7)), np.zeros(0)
    boxes = np.concatenate([anno['location'][mask], anno['dimensions'][mask],
                            anno['rotation_y'][mask][:, None]], axis=1).astype(np.float64)
    return boxes, np.asarray(anno['score'])[mask].astype(np.float64)


def r11_ap(rows):
    """rows: list of (score, is_tp); returns 11-point AP and the valid-GT count."""
    if not rows:
        return 0.0, 0.0
    rows = sorted(rows, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows])
    fp = np.cumsum([1 - r[1] for r in rows])
    recall = tp / max(rows[0][2], 1e-9) if len(rows[0]) > 2 else None
    return None


def main():
    args = parse_args()
    with open(FRAME_LIST) as f:
        frames = [line.strip() for line in f if line.strip()]
    gt_annos = read_annos(LABEL_DIR, frames, with_score=False)
    dt_annos = read_annos(args.pred, frames, with_score=True)
    if sorted(os.listdir(args.pred)) != sorted(f + '.txt' for f in frames):
        print('!! prediction folder does not match val.txt exactly')

    report = {'tag': args.tag, 'pred': args.pred, 'frames': len(frames), 'classes': {}}

    for cls in CLASSES:
        thr = IOU_THRESH[cls]
        gt_by_frame, dt_by_frame = [], []
        for i in range(len(frames)):
            g, _ = boxes_of(gt_annos[i], cls)
            d, s = boxes_of(dt_annos[i], cls)
            gt_by_frame.append(g)
            dt_by_frame.append((d, s))

        # greedy matching, highest score first
        matches = []           # (score, is_tp, gt_distance, errors, bucket)
        fn_records = []        # (bucket, best_iou)
        fp_records = []        # (score, best_other_class_iou, bucket)
        n_gt_total = 0
        for i in range(len(frames)):
            gt = gt_by_frame[i]
            dt, score = dt_by_frame[i]
            n_gt_total += len(gt)
            if len(gt) and len(dt):
                iou = d3_box_overlap(gt, dt, -1)          # (n_gt, n_dt)
            else:
                iou = np.zeros((len(gt), len(dt)))
            taken_gt = np.zeros(len(gt), dtype=bool)
            for j in np.argsort(-score):
                if len(gt) == 0:
                    break
                col = iou[:, j]
                k = int(np.argmax(col))
                if col[k] >= thr and not taken_gt[k]:
                    taken_gt[k] = True
                    g = gt[k]
                    d = dt[j]
                    # centre error in the camera frame: x lateral, y vertical, z forward
                    matches.append((float(score[j]), 1, float(g[2]),
                                    (d[0] - g[0], d[1] - g[1], d[2] - g[2]),
                                    (d[3] - g[3], d[4] - g[4], d[5] - g[5]),
                                    d[6] - g[6]))
                else:
                    fp_records.append((float(score[j]), float(col.max()) if len(col) else 0.0,
                                       float(dt[j][2])))
            for k in range(len(gt)):
                if not taken_gt[k]:
                    best = float(iou[k].max()) if len(dt) else 0.0
                    fn_records.append((float(gt[k][2]), best))

        # --- AP per distance bucket, R11 with the same 11-point rule ---
        def ap_for(records, num_gt):
            if num_gt == 0:
                return None, 0
            rows = sorted(records, key=lambda r: -r[0])
            tp = np.cumsum([r[1] for r in rows]).astype(float)
            fp = np.cumsum([1 - r[1] for r in rows]).astype(float)
            rec = tp / num_gt
            prec = tp / np.maximum(tp + fp, 1e-9)
            for i in range(len(prec) - 2, -1, -1):
                prec[i] = max(prec[i], prec[i + 1])
            ap = 0.0
            for t in range(11):
                target = t / 10.0
                idx = np.where(rec >= target)[0]
                ap += (prec[idx[0]] if len(idx) else 0.0) / 11.0
            return float(ap * 100), len(rows)

        results = {}
        ap_all, n_dt = ap_for([(m[0], 1) for m in matches] +
                              [(f[0], 0) for f in fp_records], n_gt_total)
        results['all'] = {'ap_r11': ap_all, 'gt': n_gt_total, 'dt': n_dt,
                          'tp': len(matches), 'fp': len(fp_records), 'fn': len(fn_records)}
        for lo, hi in DIST_BUCKETS:
            recs = [(m[0], 1) for m in matches if lo <= m[2] < hi]
            recs += [(f[0], 0) for f in fp_records if lo <= f[2] < hi]
            n_gt_b = sum(1 for i in range(len(frames)) for g in gt_by_frame[i] if lo <= g[2] < hi)
            ap_b, _ = ap_for(recs, n_gt_b)
            fns = [f for f in fn_records if lo <= f[0] < hi]
            results['%d-%dm' % (lo, hi)] = {
                'ap_r11': ap_b, 'gt': n_gt_b,
                'fn_no_prediction': sum(1 for f in fns if f[1] < 0.1),
                'fn_localisation': sum(1 for f in fns if 0.1 <= f[1] < thr),
                'fn_total': len(fns),
            }

        centre = np.array([m[3] for m in matches]) if matches else np.zeros((1, 3))
        size = np.array([m[4] for m in matches]) if matches else np.zeros((1, 3))
        yaw = np.array([np.degrees(abs(((m[5] + np.pi / 2) % np.pi) - np.pi / 2))
                        for m in matches]) if matches else np.zeros(1)
        results['tp_errors'] = {
            'median_lateral': float(np.median(centre[:, 0])),
            'median_vertical': float(np.median(centre[:, 1])),
            'median_forward': float(np.median(centre[:, 2])),
            'p90_forward': float(np.percentile(centre[:, 2], 90)),
            'median_size_lwh': [float(x) for x in np.median(size, axis=0)],
            'median_yaw_deg': float(np.median(yaw)),
            'p90_yaw_deg': float(np.percentile(yaw, 90)),
        }
        results['fp_confusion'] = int(sum(1 for f in fp_records if f[1] >= thr))
        results['fp_total'] = len(fp_records)
        report['classes'][cls] = results

        print('\n=== %s (3D IoU threshold %.2f) ===' % (cls, thr))
        print('  whole area : AP_R11 %6.2f | GT %5d  TP %5d  FP %5d  FN %5d'
              % (results['all']['ap_r11'], n_gt_total, results['all']['tp'],
                 results['all']['fp'], results['all']['fn']))
        for lo, hi in DIST_BUCKETS:
            b = results['%d-%dm' % (lo, hi)]
            if b['gt'] == 0:
                continue
            print('  %2d-%2d m   : AP_R11 %6.2f | GT %5d | FN %4d (no pred %4d, localisation %4d)'
                  % (lo, hi, b['ap_r11'], b['gt'], b['fn_total'],
                     b['fn_no_prediction'], b['fn_localisation']))
        e = results['tp_errors']
        print('  TP centre error (m): lateral %+.3f vertical %+.3f forward %+.3f (p90 %+.3f)'
              % (e['median_lateral'], e['median_vertical'], e['median_forward'], e['p90_forward']))
        print('  TP size error (l,w,h) %s | yaw median %.2f deg (p90 %.2f)'
              % (['%+.3f' % v for v in e['median_size_lwh']], e['median_yaw_deg'], e['p90_yaw_deg']))
        print('  FP overlapping a GT of another class: %d / %d'
              % (results['fp_confusion'], results['fp_total']))

    out = args.out or os.path.join(HERE, 'errors_%s.json' % args.tag)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print('\nreport written to %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
