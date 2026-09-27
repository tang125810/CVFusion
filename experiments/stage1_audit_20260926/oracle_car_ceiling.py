"""How much Car AP is hiding in each geometric error term?

The error decomposition showed that of 4291 Car GT, 422 have a prediction whose
BEV IoU passes the threshold while the 3D IoU does not, and that the vertical
placement (bottom height) dominates those failures.  Before spending GPU time on
an intervention it is worth measuring the *ceiling*: replace one predicted
attribute at a time with the value of its target ground-truth box, keep every
other prediction untouched, and re-score with the official protocol.

Attributes are replaced using a generous correspondence (same class, nearest GT
by BEV centre within a radius, no IoU requirement), which is the right notion
for an oracle: we want to know what the detector could reach if that one term
were perfect, not to re-match boxes.

Usage::

    python experiments/stage1_audit_20260926/oracle_car_ceiling.py \
        --pred experiments/stage1_audit_20260926/pred_imgmulti_ep72 --classes Car
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_official_protocol as sop  # noqa: E402

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
# KITTI order in the txt: x1 y1 x2 y2 h w l x y z ry score
ATTRS = {
    'bottom_y': ('location', 1),
    'centre_z': ('location', 2),
    'length_l': ('dimensions', 0),
    'height_h': ('dimensions', 1),
    'width_w': ('dimensions', 2),
    'yaw': ('rotation_y', None),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True)
    p.add_argument('--classes', default='Car')
    p.add_argument('--radius', type=float, default=4.0, help='GT correspondence radius in BEV (m)')
    p.add_argument('--out', default=None)
    return p.parse_args()


def bev_centre(box):
    return box[0], box[2]          # camera x, z


def apply_oracle(gt_annos, dt_annos, frames, cls, attrs):
    """Return new dt_annos with the requested attributes copied from the GT."""
    out = []
    n_touched = 0
    for i, fid in enumerate(frames):
        g = gt_annos[i]
        d = dt_annos[i]
        new = {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in d.items()}
        gm = np.zeros(len(g['name']), dtype=bool)
        if len(g['name']):
            gm = np.asarray(g['name']) == cls
        gboxes = np.concatenate([g['location'][gm], g['dimensions'][gm],
                                 g['rotation_y'][gm][:, None]], axis=1) if gm.sum() else np.zeros((0, 7))
        dm = np.zeros(len(d['name']), dtype=bool)
        if len(d['name']):
            dm = np.asarray(d['name']) == cls
        idx = np.where(dm)[0]
        if len(gboxes) == 0 or len(idx) == 0:
            out.append(new)
            continue
        for j in idx:
            c = np.array([new['location'][j][0], new['location'][j][2]])
            gc = gboxes[:, [0, 2]]
            dist = np.linalg.norm(gc - c[None, :], axis=1)
            k = int(np.argmin(dist))
            if dist[k] > args_radius[0]:
                continue
            for name in attrs:
                field, sub = ATTRS[name]
                if field == 'rotation_y':
                    new['rotation_y'][j] = g['rotation_y'][gm][k]
                elif field == 'location':
                    new['location'][j][sub] = g['location'][gm][k][sub]
                else:
                    new['dimensions'][j][sub] = g['dimensions'][gm][k][sub]
            n_touched += 1
        out.append(new)
    return out, n_touched


args_radius = [4.0]


def main():
    args = parse_args()
    args_radius[0] = args.radius
    with open(sop.FRAME_LIST) as f:
        frames = [line.strip() for line in f if line.strip()]
    gt_annos = sop.read_annos(sop.LABEL_DIR, frames, with_score=False)
    dt_annos = sop.read_annos(args.pred, frames, with_score=True)
    kitti_eval = sop.import_devkit()
    sop.set_iou_source(kitti_eval, 'exact')

    scenarios = {
        'as-is': [],
        'oracle_vertical (bottom y)': ['bottom_y'],
        'oracle_length l': ['length_l'],
        'oracle_size (l+h+w)': ['length_l', 'height_h', 'width_w'],
        'oracle_yaw': ['yaw'],
        'oracle_vertical+length': ['bottom_y', 'length_l'],
    }
    report = {}
    for name, attrs in scenarios.items():
        if attrs:
            dt_use, n = apply_oracle(gt_annos, dt_annos, frames, args.classes, attrs)
        else:
            dt_use, n = dt_annos, 0
        res = sop.score_region(kitti_eval, gt_annos, dt_use, 'entire_area')
        report[name] = {'car_3d_r11': res['Car']['3d_r11'], 'car_bev_r11': res['Car']['bev_r11'],
                        'mAP_3d_r11': res['mAP']['3d_r11'], 'objects_touched': n}
        print('%-26s Car 3D R11 %7.3f | Car BEV R11 %7.3f | mAP %7.3f | touched %5d'
              % (name, res['Car']['3d_r11'], res['Car']['bev_r11'], res['mAP']['3d_r11'], n))
    out = args.out or os.path.join(HERE, 'oracle_%s_ceiling.json' % args.classes.lower())
    with open(out, 'w') as f:
        json.dump({'pred': args.pred, 'class': args.classes, 'radius': args.radius,
                   'scenarios': report}, f, indent=2)
    print('\nreport written to %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
