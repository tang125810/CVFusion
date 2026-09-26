"""Score frozen predictions with the *project* evaluator (vod_eval=True).

The historic 52.5843 number was produced by
``pcdet.datasets.kitti.kitti_object_eval_python.eval.get_vod_eval_result``
called on the OpenPCDet infos (GT) and a ``result.pkl`` (DT).  That entry point
differs from the official devkit in exactly two ways:

* ``clean_data(..., vod_eval=True)`` applies **no** ignore rules: every GT of
  the class counts, and no detection is dropped for being short in the image;
* the rotated IoU comes from the project's numba-CUDA kernel.

This script re-runs that evaluator on the same frozen pickle, optionally with
the exact float64 IoU, so the protocol effect and the IoU effect can be told
apart instead of being lumped into one number.

Usage::

    python experiments/stage1_audit_20260926/score_inhouse_protocol.py \
        --det output/.../result.pkl --iou shipped
    python experiments/stage1_audit_20260926/score_inhouse_protocol.py \
        --det output/.../result.pkl --iou exact
"""

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_GT = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames/vod_infos_val.pkl')

CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--det', required=True, help='result.pkl')
    p.add_argument('--gt', default=DEFAULT_GT, help='vod_infos_val.pkl (GT annos source)')
    p.add_argument('--iou', choices=['shipped', 'exact'], default='shipped')
    p.add_argument('--tag', default='run')
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, HERE)
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

    if args.iou == 'exact':
        import exact_rotated_iou as ex
        kitti_eval.bev_box_overlap = lambda boxes, qboxes, criterion=-1: ex.rotated_iou(
            np.asarray(boxes, dtype=np.float64), np.asarray(qboxes, dtype=np.float64), criterion)
        kitti_eval.d3_box_overlap = lambda boxes, qboxes, criterion=-1: ex.d3_box_overlap(
            np.asarray(boxes, dtype=np.float64), np.asarray(qboxes, dtype=np.float64), criterion)

    with open(args.gt, 'rb') as f:
        infos = pickle.load(f)
    gt_annos = [copy.deepcopy(info['annos']) for info in infos]
    with open(args.det, 'rb') as f:
        dt_annos = pickle.load(f)
    print('GT frames %d | DT frames %d | IoU=%s' % (len(gt_annos), len(dt_annos), args.iou))

    _, ap_dict = kitti_eval.get_vod_eval_result(copy.deepcopy(gt_annos),
                                                copy.deepcopy(dt_annos), list(CLASSES))
    row = {}
    for cls in CLASSES:
        row[cls] = {
            '3d_r11': float(ap_dict['%s_3d/moderate' % cls]),
            '3d_r40': float(ap_dict.get('%s_3d/moderate_R40' % cls, float('nan'))),
            'bev_r11': float(ap_dict['%s_bev/moderate' % cls]),
            'bev_r40': float(ap_dict.get('%s_bev/moderate_R40' % cls, float('nan'))),
        }
    row['mAP'] = {k: float(np.mean([row[c][k] for c in CLASSES]))
                  for k in ('3d_r11', '3d_r40', 'bev_r11', 'bev_r40')}

    print('\n=== project protocol (vod_eval=True) :: %s :: IoU=%s ===' % (args.tag, args.iou))
    print('  %-11s %9s %9s %9s %9s' % ('class', '3D R11', '3D R40', 'BEV R11', 'BEV R40'))
    for cls in CLASSES + ['mAP']:
        r = row[cls]
        print('  %-11s %9.4f %9.4f %9.4f %9.4f'
              % (cls, r['3d_r11'], r['3d_r40'], r['bev_r11'], r['bev_r40']))

    out = args.out or os.path.join(HERE, 'inhouse_%s_%s.json' % (args.tag, args.iou))
    with open(out, 'w') as f:
        json.dump({'tag': args.tag, 'det': args.det, 'iou': args.iou, 'result': row}, f, indent=2)
    print('report written to %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
