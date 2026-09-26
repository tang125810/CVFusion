"""Extract the per-evaluation metrics a training run printed into its log.

train.py evaluates every ``eval_interval`` epochs (and again for the last ten
epochs after training) and prints, per class, the KITTI-style blocks::

    Car AP@0.50, 0.50, 0.50:
    bbox AP: ...
    bev  AP: ...
    3d   AP: ...
    Car AP_R40@0.50, 0.50, 0.50:
    ...

This script turns one or more training logs into a table so candidates can be
compared epoch by epoch.  These numbers come from the project's own evaluator
(numba-CUDA rotated IoU), which runs ~0.3 mAP above the official exact-IoU
protocol - use them for trends, and ``score_epochs_official.py`` for the
numbers that go into a report.

Usage::

    python experiments/stage1_audit_20260926/parse_train_curve.py \
        --logs run_a/log_train_*.txt run_b/log_train_*.txt --labels imgmulti zvalid
"""

import argparse
import glob
import re
import sys

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
BLOCK_RE = re.compile(r'(Car|Pedestrian|Cyclist) AP(_R40)?@')
VALUE_RE = re.compile(r'(bbox|bev|3d)\s+AP(_R40)?:([\d.]+)')


def parse_log(path):
    """Return [(epoch_index, {(class, mode, metric): value}), ...] in log order."""
    evals = []
    cur = {}
    cls = mode = None
    for line in open(path, errors='ignore'):
        m = BLOCK_RE.search(line)
        if m:
            cls, mode = m.group(1), ('r40' if m.group(2) else 'r11')
            continue
        v = VALUE_RE.search(line)
        if v and cls and mode:
            # the value lines themselves are printed as "bbox AP:" in both the
            # R11 and the R40 block, so the mode must come from the block header
            metric, value = v.group(1), float(v.group(3))
            cur[(cls, mode, metric)] = value
            if cls == 'Cyclist' and mode == 'r40' and metric == '3d':
                evals.append(cur)
                cur = {}
    return evals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--interval', type=int, default=5)
    args = ap.parse_args()

    labels = args.labels or [p.split('/')[-2] for p in args.logs]
    for path, label in zip(args.logs, labels):
        paths = sorted(glob.glob(path)) or [path]
        rows = []
        for p in paths:
            rows += parse_log(p)
        print('=== %s : %d evaluations (%s) ===' % (label, len(rows), ', '.join(paths)))
        print('  %-11s %s' % ('epoch', '  '.join('%7s' % c[:7] for c in CLASSES) +
                              '  %7s | %7s' % ('mAP3D', 'mAPbev')))
        for i, r in enumerate(rows):
            epoch = args.interval * (i + 1) if i < 16 else \
                max(args.interval * 16, 0) + (i - 15)
            try:
                m3 = sum(r[(c, 'r11', '3d')] for c in CLASSES) / 3.0
                mb = sum(r[(c, 'r11', 'bev')] for c in CLASSES) / 3.0
            except KeyError:
                continue
            print('  %-11s %s  %7.3f | %7.3f' % (
                'eval %d' % epoch,
                '  '.join('%7.3f' % r[(c, 'r11', '3d')] for c in CLASSES), m3, mb))
        best = max(range(len(rows)),
                   key=lambda i: sum(rows[i][(c, 'r11', '3d')] for c in CLASSES))
        r = rows[best]
        print('  best 3D R11 mAP: %.3f at index %d | bev mAP %.3f'
              % (sum(r[(c, 'r11', '3d')] for c in CLASSES) / 3.0, best + 1,
                 sum(r[(c, 'r11', 'bev')] for c in CLASSES) / 3.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
