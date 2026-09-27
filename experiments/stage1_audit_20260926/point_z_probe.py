"""Can the raw radar points fix the vertical placement without touching BEV?

Constraint from the user: the BEV localisation must stay at the current best
level, so the intervention has to be orthogonal to the BEV prediction path.
A refinement that *only rewrites the vertical coordinate* of each predicted box
satisfies that by construction - the BEV boxes are literally unchanged, hence the
BEV AP cannot move.

This probe measures whether such a refinement has any signal to work with:
for every Car prediction it collects the raw 5-frame radar points inside the
box's BEV footprint and compares a vertical statistic of those points with the
true box bottom (matched GT by BEV centre proximity).

If the statistic explains the error (small residual spread, sensible slope),
a point-conditioned z refiner is worth building; if not, the radar returns do
not carry the missing height information and the gap needs a different route.

Usage::
    python experiments/stage1_audit_20260926/point_z_probe.py \
        --det experiments/stage1_audit_20260926/pred_imgmulti_ep72
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

from exact_rotated_iou import rotated_iou  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from score_official_protocol import FRAME_LIST, LABEL_DIR, read_annos  # noqa: E402

DATA_ROOT = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--det', required=True, help='folder of KITTI txt predictions')
    p.add_argument('--cls', default='Car')
    p.add_argument('--margin', type=float, default=0.3, help='BEV footprint margin (m)')
    p.add_argument('--out', default=os.path.join(HERE, 'point_z_probe.json'))
    return p.parse_args()


def main():
    args = parse_args()
    with open(FRAME_LIST) as f:
        frames = [l.strip() for l in f if l.strip()]
    gt_annos = read_annos(LABEL_DIR, frames, with_score=False)
    dt_annos = read_annos(args.det, frames, with_score=True)

    rows = []
    for i, fid in enumerate(frames):
        g = gt_annos[i]
        d = dt_annos[i]
        gm = np.asarray(g['name']) == args.cls if len(g['name']) else np.zeros(0, dtype=bool)
        dm = np.asarray(d['name']) == args.cls if len(d['name']) else np.zeros(0, dtype=bool)
        if gm.sum() == 0 or dm.sum() == 0:
            continue
        gb = np.concatenate([g['location'][gm], g['dimensions'][gm], g['rotation_y'][gm][:, None]], 1)
        db = np.concatenate([d['location'][dm], d['dimensions'][dm], d['rotation_y'][dm][:, None]], 1)
        calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', fid + '.txt'))
        pts = np.fromfile(os.path.join(DATA_ROOT, 'training/velodyne', fid + '.bin'),
                          dtype=np.float32).reshape(-1, 7)[:, :3]
        pc = calib.lidar_to_rect(pts)
        for j in range(len(db)):
            x, y, z, l, h, w, ry = db[j]
            dx, dz = pc[:, 0] - x, pc[:, 2] - z
            c, s = np.cos(ry), np.sin(ry)
            lx = dx * c - dz * s
            lz = dx * s + dz * c
            inside = ((np.abs(lx) <= l / 2 + args.margin) &
                      (np.abs(lz) <= w / 2 + args.margin) &
                      (pc[:, 2] > 1.0))
            npts = int(inside.sum())
            gc = gb[:, [0, 2]]
            dist = np.linalg.norm(gc - np.array([x, z])[None, :], axis=1)
            k = int(np.argmin(dist))
            rec = {'frame': fid, 'n_points': npts,
                   'pred_bottom': float(y), 'gt_bottom': float(gb[k, 1]) if dist[k] < 4.0 else None}
            if npts > 0:
                yp = pc[inside, 1]
                rec.update({'y_max': float(yp.max()), 'y_p90': float(np.percentile(yp, 90)),
                            'y_p75': float(np.percentile(yp, 75)), 'y_med': float(np.median(yp)),
                            'y_min': float(yp.min()),
                            'n_points': npts})
            rows.append(rec)

    with_gt = [r for r in rows if r.get('gt_bottom') is not None]
    with_pts = [r for r in with_gt if r['n_points'] > 0]
    print('Car predictions: %d | with GT match: %d | with >=1 point inside: %d (%.1f%%)'
          % (len(rows), len(with_gt), len(with_pts), 100 * len(with_pts) / max(len(with_gt), 1)))
    base_err = np.array([r['pred_bottom'] - r['gt_bottom'] for r in with_gt])
    print('current  bottom error: median %+.3f | MAD %.3f | |e|>0.3 %.1f%%'
          % (np.median(base_err), np.median(np.abs(base_err - np.median(base_err))),
             100 * np.mean(np.abs(base_err) > 0.3)))
    summary = {'n': len(with_gt), 'n_with_points': len(with_pts),
               'current_median': float(np.median(base_err)),
               'current_mad': float(np.median(np.abs(base_err - np.median(base_err))))}
    if with_pts:
        print('\n%-8s %-9s %-9s %-9s %-9s | %-24s' % ('npts', 'statistic', 'corr', 'slope', 'offset', 'residual MAD'))
        for stat in ('y_max', 'y_p90', 'y_p75', 'y_med', 'y_min'):
            S = np.array([r[stat] for r in with_pts])
            T = np.array([r['gt_bottom'] for r in with_pts])
            if S.std() < 1e-6:
                continue
            corr = float(np.corrcoef(S, T)[0, 1])
            A = np.stack([S, np.ones_like(S)], 1)
            slope, offset = np.linalg.lstsq(A, T, rcond=None)[0]
            resid = T - (slope * S + offset)
            mad = float(np.median(np.abs(resid - np.median(resid))))
            print('%-8s %-9s %+9.3f %9.3f %+9.3f | %24.3f'
                  % ('all', stat, corr, slope, offset, mad))
            summary[stat] = {'corr': corr, 'slope': float(slope), 'offset': float(offset), 'residual_mad': mad}
        for nb in (1, 2, 5):
            sub = [r for r in with_pts if r['n_points'] >= nb]
            if not sub:
                continue
            S = np.array([r['y_max'] for r in sub]); T = np.array([r['gt_bottom'] for r in sub])
            resid = T - S
            print('npts>=%d (%4d objs): median(y_max - gt) %+.3f | MAD %.3f'
                  % (nb, len(sub), -np.median(resid), float(np.median(np.abs(resid - np.median(resid))))))
    with open(args.out, 'w') as f:
        json.dump({'summary': summary, 'rows': rows[:5000]}, f, indent=2)
    print('\nreport written to %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
