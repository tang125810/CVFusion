"""Learned z-only refiner: fix the Car vertical placement without touching BEV.

Why this module exists
----------------------
The error decomposition on the best Stage1 model shows the Car 3D AP gap is a
vertical-placement problem: replacing only the predicted bottom height with the
GT value takes Car 3D AP 42.82 -> 52.36 (+9.54) and the whole mAP 56.16 -> 59.35,
while a constant correction is worth +0.16 mAP (variance, not bias).  The error
grows with distance (MAD 0.063 m below 15 m, 0.172 m beyond 35 m).

Two interventions were measured and rejected:

* shared-feature height statistics (HEIGHT_STATS): vertical MAD -20% but the
  BEV-correct fraction fell 78.6% -> 73.3%, net negative;
* re-weighting the z regression term (zw): no vertical improvement at all and
  BEV -5.0 points.

Hand-read point statistics are useless too: the best percentile statistic,
fitted on the validation split, has a residual MAD of 0.42 m, 3.6x worse than
the model's own 0.116 m.

What is left is a *learned* refiner that reads the same points with a network
instead of a percentile - and that rewrites **only the vertical coordinate**, so
the BEV boxes are unchanged by construction and the BEV metrics cannot move.

Usage::

    # 1) build training pairs from the train split and fit the refiner
    python experiments/stage1_audit_20260926/zrefiner.py train \
        --det output/.../zrefiner_train_split/eval/epoch_72/train/train_split/result.pkl \
        --out experiments/stage1_audit_20260926/zrefiner.pt

    # 2) apply it to a frozen prediction folder and score
    python experiments/stage1_audit_20260926/zrefiner.py apply \
        --det experiments/stage1_audit_20260926/pred_imgmulti_ep72 \
        --model experiments/stage1_audit_20260926/zrefiner.pt \
        --out experiments/stage1_audit_20260926/pred_imgmulti_ep72_zref
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '_devkit_stubs'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(REPO)), '参考项目', 'SGDet3D-main',
                                'tools_det3d', 'view-of-delft-dataset'))

from pcdet.utils.calibration_kitti import Calibration  # noqa: E402
from exact_rotated_iou import rotated_iou  # noqa: E402

DATA_ROOT = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames')
MAX_POINTS = 32
CLASS = 'Car'


# ---------------------------------------------------------------------------
# feature extraction (shared by train and apply)
# ---------------------------------------------------------------------------

def load_points(fid, calib):
    path = os.path.join(DATA_ROOT, 'training/velodyne', fid + '.bin')
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    xyz_cam = calib.lidar_to_rect(pts[:, :3])
    return np.concatenate([xyz_cam, pts[:, 3:6]], axis=1)     # x,y,z,rcs,vr,vrc


def box_features(box, pts, margin=0.25):
    """Per-box point set (MAX_POINTS x 6) + mask + global vector.

    ``box`` is a camera-frame row ``[x, y, z, l, h, w, ry]`` (y = bottom centre),
    ``pts`` rows are ``[x, y, z, rcs, v_r, v_r_comp]`` in the same frame.
    Per-point features: local length/height/width offsets and the three radar
    attributes; the global vector carries the box geometry and the point count.
    """
    x, y, z, l, h, w, ry = box
    if len(pts) == 0:
        return None
    dx, dz = pts[:, 0] - x, pts[:, 2] - z
    c, s = np.cos(ry), np.sin(ry)
    lx = dx * c - dz * s
    lz = dx * s + dz * c
    inside = ((np.abs(lx) <= l / 2 + margin) & (np.abs(lz) <= w / 2 + margin)
              & (pts[:, 2] > 1.0))
    if not inside.any():
        return None
    lx_, lz_ = lx[inside], lz[inside]
    ly_ = np.clip(pts[inside, 1] - y, -4.0, 4.0)
    extra = pts[inside, 3:6]
    n = len(lx_)
    if n > MAX_POINTS:
        idx = np.linspace(0, n - 1, MAX_POINTS).astype(int)
        lx_, ly_, lz_, extra = lx_[idx], ly_[idx], lz_[idx], extra[idx]
        n = MAX_POINTS
    feats = np.stack([lx_, ly_, lz_, extra[:, 0], extra[:, 1], extra[:, 2]], axis=1).astype(np.float32)
    mask = np.ones(n, dtype=np.float32)
    if n < MAX_POINTS:
        feats = np.concatenate([feats, np.zeros((MAX_POINTS - n, feats.shape[1]), np.float32)], 0)
        mask = np.concatenate([mask, np.zeros(MAX_POINTS - n, np.float32)])
    global_vec = np.array([l, w, h, y, z, x, n / MAX_POINTS], dtype=np.float32)
    return feats, mask, global_vec


def boxes_from_anno(anno, cls=CLASS):
    m = np.asarray(anno['name']) == cls if len(anno['name']) else np.zeros(0, dtype=bool)
    if m.sum() == 0:
        return np.zeros((0, 7)), np.zeros(0)
    return (np.concatenate([anno['location'][m], anno['dimensions'][m],
                            anno['rotation_y'][m][:, None]], 1).astype(np.float64),
            np.asarray(anno['score'])[m].astype(np.float64))


def build_pairs(det_path, frame_list, gt_path=None):
    """Return (points_feat, mask, global, target_dy) for matched Car boxes."""
    with open(det_path, 'rb') as f:
        det = pickle.load(f)
    by_frame = {str(a['frame_id']): a for a in det}
    with open(frame_list) as f:
        frames = [l.strip() for l in f if l.strip()]
    if gt_path is None:
        gt_by_frame = None
    else:
        with open(gt_path, 'rb') as f:
            infos = pickle.load(f)
        gt_by_frame = {info['point_cloud']['lidar_idx']: info['annos'] for info in infos}

    P, M, G, T, meta = [], [], [], [], []
    for fid in frames:
        if fid not in by_frame:
            continue
        det_anno = by_frame[fid]
        dboxes, dscores = boxes_from_anno(det_anno)
        if len(dboxes) == 0:
            continue
        calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', fid + '.txt'))
        pts = load_points(fid, calib)
        gboxes = None
        if gt_by_frame is not None:
            ganno = gt_by_frame.get(fid)
            if ganno is not None:
                gboxes = np.concatenate([ganno['location'], ganno['dimensions'],
                                         ganno['rotation_y'][:, None]], 1).astype(np.float64) \
                    if len(ganno['name']) else np.zeros((0, 7))
                gm = np.asarray(ganno['name']) == CLASS if len(ganno['name']) else np.zeros(0, bool)
                gboxes = gboxes[gm]
        for j in range(len(dboxes)):
            feats = box_features(dboxes[j], pts)
            if feats is None:
                continue
            if gboxes is None or len(gboxes) == 0:
                target = 0.0
            else:
                ib = rotated_iou(gboxes[:, [0, 2, 3, 5, 6]],
                                 dboxes[j][None][:, [0, 2, 3, 5, 6]], -1)[:, 0]
                k = int(np.argmax(ib))
                if ib[k] < 0.5:
                    continue
                target = float(gboxes[k][1] - dboxes[j][1])
            P.append(feats[0]); M.append(feats[1]); G.append(feats[2]); T.append(target)
            meta.append((fid, j))
    if not P:
        raise RuntimeError('no training pairs built')
    return (np.stack(P), np.stack(M), np.stack(G), np.array(T, dtype=np.float32), meta)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(torch):
    import torch.nn as nn

    class PointNetRefiner(nn.Module):
        def __init__(self, in_pts=6, in_global=7):
            super().__init__()
            self.point_mlp = nn.Sequential(
                nn.Linear(in_pts, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 64), nn.ReLU(inplace=True))
            self.head = nn.Sequential(
                nn.Linear(64 + in_global, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

        def forward(self, pts, mask, glob):
            f = self.point_mlp(pts)
            f = f * mask[..., None]
            pooled = f.max(dim=1).values
            return self.head(torch.cat([pooled, glob], dim=1)).squeeze(-1)

    return PointNetRefiner()


def cmd_train(args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    frame_list = os.path.join(DATA_ROOT, 'ImageSets/train.txt')
    gt_path = os.path.join(DATA_ROOT, 'vod_infos_train.pkl')
    print('building training pairs from %s' % args.det)
    P, M, G, T, meta = build_pairs(args.det, frame_list, gt_path if os.path.isfile(gt_path) else None)
    print('pairs: %d | target dy: median %+.3f, MAD %.3f, |dy|>0.3 %.1f%%'
          % (len(T), np.median(T), np.median(np.abs(T - np.median(T))), 100 * np.mean(np.abs(T) > 0.3)))

    device = torch.device(args.device)
    model = build_model(torch).to(device)
    ds = TensorDataset(torch.tensor(P), torch.tensor(M), torch.tensor(G), torch.tensor(T))
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = torch.nn.SmoothL1Loss(beta=0.1)
    for ep in range(args.epochs):
        tot = 0.0
        for pb, mb, gb, tb in dl:
            pb, mb, gb, tb = pb.to(device), mb.to(device), gb.to(device), tb.to(device)
            pred = model(pb, mb, gb)
            loss = lossf(pred, tb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(tb)
        sched.step()
        print('epoch %2d  train SmoothL1 %.4f' % (ep + 1, tot / len(ds)))
    if args.val_det:
        Pv, Mv, Gv, Tv, _ = build_pairs(args.val_det, os.path.join(DATA_ROOT, 'ImageSets/val.txt'),
                                        os.path.join(DATA_ROOT, 'vod_infos_val.pkl'))
        with torch.no_grad():
            pred = model(torch.tensor(Pv).to(device), torch.tensor(Mv).to(device),
                         torch.tensor(Gv).to(device)).cpu().numpy()
        err = pred - Tv
        base = -Tv
        print('val pairs %d | refiner MAD %.3f (|e|>0.3 %.1f%%) | before MAD %.3f (|e|>0.3 %.1f%%)'
              % (len(Tv), np.median(np.abs(err - np.median(err))), 100 * np.mean(np.abs(err) > 0.3),
                 np.median(np.abs(base - np.median(base))), 100 * np.mean(np.abs(base) > 0.3)))
    torch.save({'state': model.state_dict()}, args.out)
    print('model written to %s' % args.out)
    return 0


def cmd_apply(args):
    import torch

    device = torch.device(args.device)
    model = build_model(torch).to(device)
    ck = torch.load(args.model, map_location=device)
    model.load_state_dict(ck['state'])
    model.eval()

    with open(args.frame_list) as f:
        frames = [l.strip() for l in f if l.strip()]
    if args.det.endswith('.pkl'):
        with open(args.det, 'rb') as f:
            det = pickle.load(f)
        by_frame = {str(a['frame_id']): a for a in det}
    else:
        from score_official_protocol import read_annos
        annos = read_annos(args.det, frames, with_score=True)
        by_frame = {fid: annos[i] for i, fid in enumerate(frames)}

    os.makedirs(args.out, exist_ok=True)
    import export_predictions_to_kitti as exp

    n_changed, deltas = 0, []
    for fid in frames:
        anno = by_frame[fid]
        calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', fid + '.txt'))
        pts = load_points(fid, calib)
        names = list(anno['name'])
        loc = np.array(anno['location'], dtype=np.float64).copy()
        dims = np.array(anno['dimensions'], dtype=np.float64)
        ry = np.array(anno['rotation_y'], dtype=np.float64)
        lines = []
        feats_batch, mask_batch, glob_batch, idx_batch = [], [], [], []
        for i, nm in enumerate(names):
            if str(nm) != CLASS:
                continue
            f = box_features(np.concatenate([loc[i], dims[i], [ry[i]]]), pts)
            if f is None:
                continue
            feats_batch.append(f[0]); mask_batch.append(f[1]); glob_batch.append(f[2]); idx_batch.append(i)
        if feats_batch:
            with torch.no_grad():
                dz = model(torch.tensor(np.stack(feats_batch)).to(device),
                           torch.tensor(np.stack(mask_batch)).to(device),
                           torch.tensor(np.stack(glob_batch)).to(device)).cpu().numpy()
            dz = np.clip(dz, -args.clip, args.clip)
            for k, i in enumerate(idx_batch):
                loc[i, 1] += float(dz[k])
                deltas.append(float(dz[k]))
                n_changed += 1
        for i in range(len(names)):
            lines.append(exp.kitti_line(str(names[i]), 0.0, 0, float(anno['alpha'][i]),
                                        [float(v) for v in anno['bbox'][i]],
                                        [float(v) for v in dims[i]],
                                        [float(v) for v in loc[i]],
                                        float(ry[i]), float(anno['score'][i])))
        with open(os.path.join(args.out, fid + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    print('applied to %d Car boxes | dz median %+.3f MAD %.3f (clip %.2f m)'
          % (n_changed, float(np.median(deltas)), float(np.median(np.abs(np.array(deltas) - np.median(deltas)))),
             args.clip))
    print('written to %s' % args.out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train')
    t.add_argument('--det', required=True, help='train-split result.pkl')
    t.add_argument('--val-det', default=None, help='optional val result.pkl for a quick check')
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=25)
    t.add_argument('--device', default='cuda')
    t.set_defaults(func=cmd_train)

    a = sub.add_parser('apply')
    a.add_argument('--det', required=True, help='result.pkl or folder of KITTI txt')
    a.add_argument('--model', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--clip', type=float, default=1.0)
    a.add_argument('--device', default='cuda')
    a.add_argument('--frame-list', default=os.path.join(DATA_ROOT, 'ImageSets/val.txt'))
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
