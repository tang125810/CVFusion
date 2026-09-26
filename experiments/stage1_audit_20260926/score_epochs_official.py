"""Score a run's per-epoch predictions with the official protocol, in one process.

Training writes ``eval/eval_with_train/epoch_N/val/result.pkl`` for the epochs
it evaluates.  Comparing two candidates by a single checkpoint is misleading -
the verified baseline's automatic best checkpoint (epoch 80) is not its true
R11 optimum (epoch 78) - so this script re-scores a *range* of epochs under the
same validated protocol (official devkit bookkeeping + exact float64 IoU) and
prints the per-epoch curve.

Usage::

    python experiments/stage1_audit_20260926/score_epochs_official.py \
        --run output/cfgs/vod_models/vod_cvfusion_paper_stage1_imgmulti/paper_stage1_imgmulti_v1 \
        --tag imgmulti --epochs 70-80 --iou exact
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import score_official_protocol as sop  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='experiment directory (contains eval/eval_with_train)')
    p.add_argument('--tag', default='run')
    p.add_argument('--epochs', default='all', help='e.g. 70-80 or 78 or all')
    p.add_argument('--regions', default='entire_area,roi')
    p.add_argument('--iou', choices=['exact', 'shipped'], default='exact')
    p.add_argument('--out', default=None)
    return p.parse_args()


def epoch_dirs(run, spec):
    """Collect per-epoch result.pkl files.

    train.py writes two kinds: ``eval/eval_during_train/epoch_N/result.pkl`` for
    the every-``eval_interval`` evaluation and
    ``eval/eval_with_train/epoch_N/val/result.pkl`` for the last ten epochs
    (which only exist once the run has finished).  Prefer the latter when both
    are present.
    """
    base = os.path.join(run, 'eval')
    found = {}
    for name in sorted(os.listdir(os.path.join(base, 'eval_during_train'))) \
            if os.path.isdir(os.path.join(base, 'eval_during_train')) else []:
        if name.startswith('epoch_'):
            path = os.path.join(base, 'eval_during_train', name, 'result.pkl')
            if os.path.isfile(path):
                found[int(name.split('_')[1])] = path
    with_train = os.path.join(base, 'eval_with_train')
    if os.path.isdir(with_train):
        for name in os.listdir(with_train):
            if name.startswith('epoch_'):
                path = os.path.join(with_train, name, 'val', 'result.pkl')
                if os.path.isfile(path):
                    found[int(name.split('_')[1])] = path
    if spec != 'all':
        if '-' in spec:
            lo, hi = (int(x) for x in spec.split('-'))
            found = {e: p for e, p in found.items() if lo <= e <= hi}
        else:
            found = {int(spec): found[int(spec)]} if int(spec) in found else {}
    return dict(sorted(found.items()))


def main():
    args = parse_args()
    epochs = epoch_dirs(args.run, args.epochs)
    if not epochs:
        raise RuntimeError('no result.pkl found for the requested epochs')
    print('scoring epochs: %s' % list(epochs))

    import export_predictions_to_kitti as exp
    import exact_rotated_iou  # noqa: F401  (registered through sop)

    with open(sop.FRAME_LIST) as f:
        frames = [line.strip() for line in f if line.strip()]
    kitti_eval = sop.import_devkit()
    sop.set_iou_source(kitti_eval, args.iou)
    regions = [r for r in args.regions.split(',') if r]

    report = {'run': args.run, 'tag': args.tag, 'iou': args.iou, 'rows': []}
    workdir = tempfile.mkdtemp(prefix='vod_epochs_')
    try:
        gt_dir = sop.build_gt_subset(workdir, frames)
        gt_annos = sop.read_annos(gt_dir, frames, with_score=False)
        for epoch, pkl in epochs.items():
            pred_dir = os.path.join(workdir, 'pred_epoch_%d' % epoch)
            os.makedirs(pred_dir, exist_ok=True)
            with open(pkl, 'rb') as f:
                det_annos = pickle.load(f)
            by_frame = {str(a['frame_id']): a for a in det_annos}
            missing = [f for f in frames if f not in by_frame]
            if missing:
                raise RuntimeError('epoch %d is missing %d frames' % (epoch, len(missing)))
            for fid in frames:
                a = by_frame[fid]
                lines = [exp.kitti_line(str(a['name'][i]), 0.0, 0,
                                        float(a['alpha'][i]), [float(v) for v in a['bbox'][i]],
                                        [float(v) for v in a['dimensions'][i]],
                                        [float(v) for v in a['location'][i]],
                                        float(a['rotation_y'][i]), float(a['score'][i]))
                         for i in range(len(a['name']))]
                with open(os.path.join(pred_dir, fid + '.txt'), 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
            dt_annos = sop.read_annos(pred_dir, frames, with_score=True)
            row = {'epoch': epoch, 'regions': {}}
            for region in regions:
                res = sop.score_region(kitti_eval, gt_annos, dt_annos, region)
                row['regions'][region] = res
            report['rows'].append(row)
            ent = row['regions'].get('entire_area', {})
            roi = row['regions'].get('roi', {})
            print('epoch %3d | entire 3D R11 mAP %7.4f (Car %7.3f Ped %7.3f Cyc %7.3f) | roi mAP %7.4f'
                  % (epoch, ent.get('mAP', {}).get('3d_r11', float('nan')),
                     ent.get('Car', {}).get('3d_r11', float('nan')),
                     ent.get('Pedestrian', {}).get('3d_r11', float('nan')),
                     ent.get('Cyclist', {}).get('3d_r11', float('nan')),
                     roi.get('mAP', {}).get('3d_r11', float('nan'))))
            shutil.rmtree(pred_dir, ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    best = max(report['rows'], key=lambda r: r['regions'].get('entire_area', {})
               .get('mAP', {}).get('3d_r11', -1))
    print('\nbest entire-area 3D R11: epoch %d (mAP %.4f)'
          % (best['epoch'], best['regions']['entire_area']['mAP']['3d_r11']))
    out = args.out or os.path.join(HERE, 'epochs_%s_%s.json' % (args.tag, args.iou))
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print('report written to %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
