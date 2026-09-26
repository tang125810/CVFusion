"""Export frozen ``result.pkl`` detections to 16-column KITTI/VoD prediction txt.

The detector stores camera-frame boxes in ``result.pkl``::

    location    (N, 3)  [x, y, z]  rect-camera, y is the *bottom* centre (KITTI/OpenPCDet
                        convention produced by ``boxes3d_lidar_to_kitti_camera``)
    dimensions  (N, 3)  [l, h, w]  length, height, width
    rotation_y  (N,)    camera-frame yaw
    bbox        (N, 4)  [x1, y1, x2, y2] in *original* image pixels

The official View-of-Delft evaluator reads columns 8:11 as ``h w l``
(``evaluation_common.get_label_annotation``) and then reorders them to ``l h w``.
The previous export script unpacked ``dimensions`` as ``l, w, h`` although the
producer writes ``l, h, w``, so every detection was emitted with height and
width swapped.  For pedestrians and cyclists (h ~ 1.7 m, w ~ 0.6 m) that turns
the box into a flat slab, which is why their 3D AP collapsed under the official
protocol while the car AP (h ~ w) barely moved.

This script writes the columns in the devkit's order and asserts the layout
against a hand-built reference row before writing anything.

Usage::

    python experiments/stage1_audit_20260926/export_predictions_to_kitti.py \
        --det  output/.../rollback_20260926/result.pkl \
        --out  experiments/stage1_audit_20260926/pred_ep78_kitti
"""

import argparse
import os
import pickle

import numpy as np

DEFAULT_FRAME_LIST = 'data/VoD/view_of_delft_PUBLIC/radar_5frames/ImageSets/val.txt'
DEFAULT_LABEL_DIR = 'data/VoD/view_of_delft_PUBLIC/radar_5frames/training/label_2'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--det', required=True, help='result.pkl with per-frame det_annos')
    p.add_argument('--out', required=True, help='output folder for the txt files')
    p.add_argument('--frame-list', default=DEFAULT_FRAME_LIST)
    p.add_argument('--label-dir', default=DEFAULT_LABEL_DIR)
    p.add_argument('--keep-frames', action='store_true',
                   help='restrict to frames present both in the pickle and the frame list')
    return p.parse_args()


def kitti_line(name, truncated, occluded, alpha, bbox, dims_lhw, loc, ry, score):
    """One 16-column VoD/KITTI row; columns 8:11 are h, w, l on purpose."""
    l, h, w = dims_lhw
    x1, y1, x2, y2 = bbox
    x, y, z = loc
    return ('%s %.2f %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.6f'
            % (name, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry, score))


def layout_self_test():
    """Round-trip one hand-built box through the devkit's own parser."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '_devkit_stubs'))
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    sys.path.insert(0, os.path.join(project, '参考项目', 'SGDet3D-main', 'tools_det3d',
                                    'view-of-delft-dataset'))
    from vod.evaluation.evaluation_common import get_label_annotation
    import tempfile

    dims = (4.20, 1.90, 2.00)  # l, h, w
    loc = (1.0, 1.5, 20.0)
    line = kitti_line('Car', 0.0, 0, -1.5, (10.0, 20.0, 60.0, 80.0), dims, loc, 0.3, 0.9)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(line + '\n')
        path = f.name
    try:
        anno = get_label_annotation(path)
    finally:
        os.unlink(path)
    got_dims = tuple(float(v) for v in anno['dimensions'][0])
    got_loc = tuple(float(v) for v in anno['location'][0])
    got_ry = float(anno['rotation_y'][0])
    got_score = float(anno['score'][0])
    ok = (np.allclose(got_dims, dims, atol=1e-6) and np.allclose(got_loc, loc, atol=1e-6)
          and abs(got_ry - 0.3) < 1e-6 and abs(got_score - 0.9) < 1e-6)
    print('[layout self-test] wrote : %s' % line)
    print('[layout self-test] devkit read back dims=%s loc=%s ry=%.3f score=%.3f -> %s'
          % (got_dims, got_loc, got_ry, got_score, 'ok' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    args = parse_args()
    rc = layout_self_test()

    with open(args.det, 'rb') as f:
        det_annos = pickle.load(f)
    print('loaded %d frames from %s' % (len(det_annos), args.det))

    with open(args.frame_list) as f:
        frames = [line.strip() for line in f if line.strip()]
    print('frame list: %d frames' % len(frames))

    det_by_frame = {}
    for anno in det_annos:
        fid = str(anno['frame_id'])
        if fid in det_by_frame:
            raise RuntimeError('duplicate frame_id %s' % fid)
        det_by_frame[fid] = anno

    if args.keep_frames:
        frames = [f for f in frames if f in det_by_frame]
    missing = [f for f in frames if f not in det_by_frame]
    extra = [f for f in det_by_frame if f not in set(frames)]
    if missing:
        raise RuntimeError('pickle misses %d frames, e.g. %s' % (len(missing), missing[:5]))
    if extra:
        raise RuntimeError('pickle has %d frames outside the list, e.g. %s' % (len(extra), extra[:5]))
    no_label = [f for f in frames if not os.path.exists(os.path.join(args.label_dir, f + '.txt'))]
    if no_label:
        raise RuntimeError('%d frames have no GT label file, e.g. %s' % (len(no_label), no_label[:5]))

    os.makedirs(args.out, exist_ok=True)
    counts, n_obj, n_empty = {}, 0, 0
    for fid in frames:
        anno = det_by_frame[fid]
        lines = []
        for i in range(len(anno['name'])):
            lines.append(kitti_line(
                str(anno['name'][i]), 0.0, 0, float(anno['alpha'][i]),
                [float(v) for v in anno['bbox'][i]],
                [float(v) for v in anno['dimensions'][i]],   # producer order: l, h, w
                [float(v) for v in anno['location'][i]],
                float(anno['rotation_y'][i]), float(anno['score'][i])))
            counts[str(anno['name'][i])] = counts.get(str(anno['name'][i]), 0) + 1
            n_obj += 1
        if not lines:
            n_empty += 1
        with open(os.path.join(args.out, fid + '.txt'), 'w') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')

    print('wrote %d files to %s' % (len(frames), args.out))
    print('objects: %d, empty frames: %d, by class: %s' % (n_obj, n_empty, counts))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
