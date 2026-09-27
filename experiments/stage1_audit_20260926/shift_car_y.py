"""Sweep a constant correction on the Car predictions' bottom height.

The error decomposition found a median bottom-height error of -0.238 m for the
Car GT that are "BEV-correct but 3D-failing".  If a *constant* shift recovers a
large part of the Car 3D AP, the error is dominated by a systematic bias (and a
principled prior fix is worth building); if the curve is flat around zero, the
error is variance and needs supervision, not calibration.

Diagnostic only: the shift is fitted on the validation split, so the numbers are
an upper bound used to separate bias from variance, not a reportable result.

Usage::
    python experiments/stage1_audit_20260926/shift_car_y.py \
        --pred experiments/stage1_audit_20260926/pred_imgmulti_ep72
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_official_protocol as sop  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--class', dest='cls', default='Car')
    ap.add_argument('--shifts', default='-0.6,-0.45,-0.3,-0.15,0,0.15,0.3,0.45,0.6')
    ap.add_argument('--out', default=os.path.join(HERE, 'shift_car_y.json'))
    args = ap.parse_args()

    with open(sop.FRAME_LIST) as f:
        frames = [line.strip() for line in f if line.strip()]
    gt_annos = sop.read_annos(sop.LABEL_DIR, frames, with_score=False)
    dt_annos = sop.read_annos(args.pred, frames, with_score=True)
    kitti_eval = sop.import_devkit()
    sop.set_iou_source(kitti_eval, 'exact')

    rows = []
    for shift in [float(x) for x in args.shifts.split(',')]:
        moved = []
        for d in dt_annos:
            new = {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in d.items()}
            m = np.zeros(len(new['name']), dtype=bool)
            if len(new['name']):
                m = np.asarray(new['name']) == args.cls
            new['location'][m, 1] += shift
            moved.append(new)
        res = sop.score_region(kitti_eval, gt_annos, moved, 'entire_area')
        rows.append({'shift': shift, 'car_3d_r11': res['Car']['3d_r11'],
                     'car_bev_r11': res['Car']['bev_r11'], 'mAP_3d_r11': res['mAP']['3d_r11']})
        print('shift %+5.2f m -> Car 3D R11 %7.3f | Car BEV %7.3f | mAP %7.3f'
              % (shift, res['Car']['3d_r11'], res['Car']['bev_r11'], res['mAP']['3d_r11']))
    with open(args.out, 'w') as f:
        json.dump({'pred': args.pred, 'class': args.cls, 'rows': rows}, f, indent=2)
    best = max(rows, key=lambda r: r['car_3d_r11'])
    print('\nbest shift %+.2f -> Car 3D %7.3f (gain %+.3f)'
          % (best['shift'], best['car_3d_r11'],
             best['car_3d_r11'] - rows[[r['shift'] for r in rows].index(0.0)]['car_3d_r11']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
