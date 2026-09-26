"""Geometry acceptance test for the sampling view transformation.

The technical report allows the sampling-based view transform to be built only
*after* the synthetic geometry checks pass, and it must be validated the same
way the scatter version was.  The decisive case is a delta test: place a single
bright feature at one image pixel, put all depth mass at the bin that matches
one chosen BEV cell, and require the lifted feature to light up exactly that
cell - with the cell coordinates derived independently in numpy.

Also checks shape/consistency against the verified scatter path.

Usage::

    python experiments/stage1_audit_20260926/verify_vt_sampling.py
"""

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from easydict import EasyDict  # noqa: E402
from pcdet.models.backbones_2d.paper_rgiter_fusion import PaperRGIterFusion  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402

DATA_ROOT = os.path.join(REPO, 'data/VoD/view_of_delft_PUBLIC/radar_5frames')
PC_RANGE = [0.0, -25.6, -3.0, 51.2, 25.6, 2.0]
FEAT_H, FEAT_W = 96, 152
BEV_H, BEV_W = 256, 256


def build_module():
    model_cfg = EasyDict({
        'IMAGE_SIZE': [384, 608], 'DEPTH_BINS': 64, 'DEPTH_MIN': 1.0, 'DEPTH_MAX': 51.2,
        'IMAGE_CHANNELS': 64, 'OUTPUT_CHANNELS': 128,
        'RADAR_SCALE_CHANNELS': [64, 64, 64],
        'RADAR_SCALE_KEYS': ['radar_bev_0', 'radar_bev_1', 'radar_bev_2'],
        'DEPTH_LOSS_WEIGHT': 0.0, 'GATE_LOSS_WEIGHT': 0.0, 'RPN_NUM_BLOCKS': 0,
        'PRETRAINED': '../../weights/swin_t-704ceda3.pth',
    })
    return PaperRGIterFusion(model_cfg, input_channels=64, point_cloud_range=PC_RANGE).eval()


def frame_geometry(frame='00000'):
    calib = Calibration(os.path.join(DATA_ROOT, 'training/calib', frame + '.txt'))
    old_h, old_w = 1216, 1936
    intrinsic = calib.P2[:, :3].astype(np.float64).copy()
    intrinsic[0] *= 608.0 / old_w
    intrinsic[1] *= 384.0 / old_h
    rect = np.eye(4)
    rect[:3, :3] = calib.R0
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :4] = calib.V2C
    cam_to_lidar = np.linalg.inv(rect @ velo_to_cam)
    return intrinsic, cam_to_lidar


def main():
    model = build_module()
    intrinsic, cam_to_lidar = frame_geometry()
    identity = np.eye(4)
    failures = []
    report = {}

    depths = np.linspace(1.0, 51.2, 64)
    x_res = (PC_RANGE[3] - PC_RANGE[0]) / BEV_W
    y_res = (PC_RANGE[4] - PC_RANGE[1]) / BEV_H
    z_levels = PC_RANGE[2] + (np.arange(model.vt_num_z) + 0.5) * \
        (PC_RANGE[5] - PC_RANGE[2]) / model.vt_num_z

    # ---- delta test: one bright pixel, depth mass at the matching bin --------
    # With a gather implementation a lit pixel is *not* expected to light a
    # single BEV cell: every cell whose centre projects into that pixel (and
    # whose depth falls in the lit bin) receives weight.  The test therefore
    # enumerates that set independently in numpy and requires an exact match.
    def expected_cells(px, py, bin_idx):
        cells = []
        for zi, z in enumerate(z_levels):
            for cy in range(BEV_H):
                y = PC_RANGE[1] + (cy + 0.5) * y_res
                for cx in range(BEV_W):
                    x = PC_RANGE[0] + (cx + 0.5) * x_res
                    cam = np.linalg.inv(cam_to_lidar) @ np.array([x, y, z, 1.0])
                    if cam[2] <= 1.0 or cam[2] >= 51.2:
                        continue
                    uv = intrinsic @ cam[:3]
                    u, v = uv[0] / cam[2], uv[1] / cam[2]
                    fx = (u + 0.5) * (FEAT_W / 608.0) - 0.5
                    fy = (v + 0.5) * (FEAT_H / 384.0) - 0.5
                    if not (-1.0 <= (2 * fx + 1) / FEAT_W - 1 <= 1.0
                            and -1.0 <= (2 * fy + 1) / FEAT_H - 1 <= 1.0):
                        continue
                    if int(np.floor(fx + 0.5)) != px or int(np.floor(fy + 0.5)) != py:
                        continue
                    b = (cam[2] - 1.0) / (51.2 - 1.0) * 63
                    if int(np.floor(b)) == bin_idx or int(np.ceil(b)) == bin_idx:
                        cells.append((cy, cx))
        return sorted(set(cells))

    for (cell_x, cell_y, z_idx) in ((60, 128, 3), (200, 120, 5), (30, 110, 1)):
        x = PC_RANGE[0] + (cell_x + 0.5) * x_res
        y = PC_RANGE[1] + (cell_y + 0.5) * y_res
        z = z_levels[z_idx]
        cam = np.linalg.inv(cam_to_lidar) @ np.array([x, y, z, 1.0])
        uv = intrinsic @ cam[:3]
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        depth = cam[2]
        fx = (u + 0.5) * (FEAT_W / 608.0) - 0.5
        fy = (v + 0.5) * (FEAT_H / 384.0) - 0.5
        px, py = int(round(fx)), int(round(fy))
        bin_idx = int(round((depth - 1.0) / (51.2 - 1.0) * 63))
        in_image = 0 <= px < FEAT_W and 0 <= py < FEAT_H and 1.0 < depth < 51.2

        image_feat = torch.zeros(1, 64, FEAT_H, FEAT_W)
        depth_prob = torch.zeros(1, 64, FEAT_H, FEAT_W)
        if in_image:
            image_feat[0, 0, py, px] = 1.0
            depth_prob[0, bin_idx, py, px] = 1.0
        out = model._lift_to_bev_sampling(
            image_feat, depth_prob, torch.tensor(intrinsic[None], dtype=torch.float32),
            torch.tensor(cam_to_lidar[None], dtype=torch.float32),
            torch.eye(4, dtype=torch.float32)[None], BEV_H, BEV_W)
        nz = sorted((int(a), int(b)) for a, b in (out[0].abs().sum(0) > 1e-6).nonzero().tolist())
        want = expected_cells(px, py, bin_idx) if in_image else []
        ok = (nz == want) and ((cell_y, cell_x) in nz if in_image else True)
        report['delta_cell_%d_%d_z%d' % (cell_x, cell_y, z_idx)] = {
            'pixel': [px, py], 'depth': float(depth), 'bin': bin_idx,
            'expected_cells': len(want), 'target_cell_hit': bool((cell_y, cell_x) in nz),
            'got': nz[:8], 'ok': bool(ok)}
        if not ok:
            failures.append('delta_cell_%d_%d_z%d' % (cell_x, cell_y, z_idx))

    # ---- shape / dtype parity with the scatter path -------------------------
    torch.manual_seed(0)
    image_feat = torch.rand(2, 64, FEAT_H, FEAT_W)
    depth_prob = torch.softmax(torch.randn(2, 64, FEAT_H, FEAT_W), dim=1)
    intr = torch.tensor(np.stack([intrinsic, intrinsic]), dtype=torch.float32)
    c2l = torch.tensor(np.stack([cam_to_lidar, cam_to_lidar]), dtype=torch.float32)
    aug = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
    out_sampling = model._lift_to_bev_sampling(image_feat, depth_prob, intr, c2l, aug, BEV_H, BEV_W)
    out_scatter = model._lift_to_bev(image_feat, depth_prob, intr, c2l, aug, BEV_H, BEV_W)
    parity = {
        'sampling_shape': list(out_sampling.shape), 'scatter_shape': list(out_scatter.shape),
        'sampling_absmean': float(out_sampling.abs().mean()),
        'scatter_absmean': float(out_scatter.abs().mean()),
        'sampling_nonzero_fraction': float((out_sampling.abs().sum(1) > 1e-6).float().mean()),
        'scatter_nonzero_fraction': float((out_scatter.abs().sum(1) > 1e-6).float().mean()),
        'cosine_similarity': float(torch.nn.functional.cosine_similarity(
            out_sampling.flatten(), out_scatter.flatten(), dim=0)),
    }
    report['parity'] = parity
    if parity['sampling_shape'] != parity['scatter_shape']:
        failures.append('shape_parity')

    print('delta tests:')
    for key, val in report.items():
        if key.startswith('delta'):
            print('  %-24s pixel %s depth %6.1f bin %2d | expected %s cells, got %s, '
                  'target hit %s -> %s'
                  % (key, val["pixel"], val["depth"], val["bin"], val["expected_cells"],
                     len(val["got"]), val["target_cell_hit"], "ok" if val["ok"] else "FAIL"))
    print('parity with the scatter path:')
    for key, val in parity.items():
        print('  %-26s %s' % (key, val))
    report['failures'] = failures
    print('\n%s' % ('SAMPLING VT GEOMETRY PASSED' if not failures
                    else 'SAMPLING VT GEOMETRY FAILED: %s' % failures))
    with open(os.path.join(HERE, 'vt_sampling_geometry.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
