"""Build the final Stage1 comparison table from the per-epoch official scores.

Reads the ``epochs_<tag>_<iou>.json`` reports written by
``score_epochs_official.py``, picks each candidate's best epoch by entire-area
3D AP_R11, and prints the table the technical report asks for (three classes x
two regions, R11 and R40) next to the paper's Stage1-only row.

Usage::

    python experiments/stage1_audit_20260926/summarize_candidates.py \
        --reports epochs_zvalid_exact.json epochs_imgmulti_exact.json \
                  epochs_vtsample_exact.json epochs_radaronly_exact.json \
        --labels "baseline zvalid(ep78)" imgmulti vtsample radaronly
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
PAPER_STAGE1 = {'Car': 52.53, 'Pedestrian': 50.76, 'Cyclist': 75.80, 'mAP': 59.70}
PAPER_STAGE1_ROI = 76.66


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reports', nargs='+', required=True)
    p.add_argument('--labels', nargs='*', default=None)
    p.add_argument('--out', default=os.path.join(HERE, 'candidate_table.json'))
    return p.parse_args()


def best_row(report, region='entire_area', metric='3d_r11'):
    rows = [r for r in report['rows'] if region in r.get('regions', {})]
    if not rows:
        return None
    return max(rows, key=lambda r: r['regions'][region]['mAP'][metric])


def fmt(res, metric='3d_r11'):
    return ' | '.join('%6.2f' % res[c][metric] for c in CLASSES) + ' | %6.2f' % res['mAP'][metric]


def main():
    args = parse_args()
    labels = args.labels or [os.path.basename(r) for r in args.reports]
    table = []
    print('%-22s %5s  %s' % ('candidate', 'epoch', 'Entire 3D AP_R11:  Car |   Ped |   Cyc |   mAP'))
    print('-' * 100)
    print('%-22s %5s  %s' % ('论文 Stage1-only', '-',
                             ' | '.join('%6.2f' % PAPER_STAGE1[c] for c in CLASSES) +
                             ' | %6.2f' % PAPER_STAGE1['mAP']))
    for path, label in zip(args.reports, labels):
        if not os.path.isfile(path):
            print('%-22s  (missing %s)' % (label, path))
            continue
        report = json.load(open(path))
        row = best_row(report)
        if row is None:
            print('%-22s  (no entire_area rows)' % label)
            continue
        ent = row['regions']['entire_area']
        roi = row['regions'].get('roi', {})
        print('%-22s %5d  %s' % (label, row['epoch'], fmt(ent, '3d_r11')))
        print('%-22s %5s  %s' % ('', 'R40', fmt(ent, '3d_r40')))
        print('%-22s %5s  %s' % ('', 'BEV', fmt(ent, 'bev_r11')))
        if roi:
            print('%-22s %5s  %s   (论文 corridor %5.2f)'
                  % ('', 'roi', fmt(roi, '3d_r11'), PAPER_STAGE1_ROI))
        table.append({'label': label, 'report': path, 'epoch': row['epoch'],
                      'entire_area': ent, 'roi': roi,
                      'delta_vs_paper': {c: ent[c]['3d_r11'] - PAPER_STAGE1[c] for c in CLASSES},
                      'delta_mAP_vs_paper': ent['mAP']['3d_r11'] - PAPER_STAGE1['mAP']})
    with open(args.out, 'w') as f:
        json.dump(table, f, indent=2)
    print('\ntable written to %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
