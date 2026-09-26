"""Why does a black image destroy the Stage1 detector?  Measure the feature statistics.

``run_image_ablation.py`` shows that replacing the image with a constant frame
collapses entire-area mAP from 52.27 to 4.96, while a mismatched (previous)
frame costs only 4.0.  A collapse of that size can mean two very different
things:

1. the detector genuinely needs image semantics, or
2. the perturbation pushes an internal tensor far outside its trained range
   (scale/shift explosion), which is a *distribution shift* rather than an
   information effect.

This probe separates the two by recording, per perturbation mode, the running
statistics of the tensors that the ablation moves: the lifted camera BEV, the
radar-conditioned gates, and the fused proposal features.  It is a diagnostic
of the trained network, not an accuracy claim.

Usage::

    python experiments/stage1_audit_20260926/probe_fusion_stats.py --batches 3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.models.backbones_2d.paper_rgiter_fusion import PaperRGIterFusion  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from eval_utils.eval_utils import load_data_to_gpu  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', default='tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml')
    p.add_argument('--ckpt', default='output/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid/'
                                    'paper_stage1_zvalid_v1/ckpt/checkpoint_epoch_78.pth')
    p.add_argument('--batches', type=int, default=3)
    p.add_argument('--modes', default='none,zero,noise,wrong_frame')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=os.path.join(HERE, 'fusion_stats.json'))
    return p.parse_args()


class Probe:
    """Collect statistics of the camera BEV, the gates and the fused features."""

    def __init__(self, module):
        self.module = module
        self.mode = 'none'
        self.records = []
        self._prev = None
        self._orig_forward = module.forward
        self._last_gates = []

    def __enter__(self):
        probe = self

        def patched_forward(batch_dict):
            images = batch_dict['images']
            if probe.mode == 'zero':
                batch_dict['images'] = torch.zeros_like(images)
            elif probe.mode == 'noise':
                generator = torch.Generator(device=images.device).manual_seed(1234)
                batch_dict['images'] = torch.rand(images.shape, generator=generator,
                                                  device=images.device, dtype=images.dtype)
            elif probe.mode == 'wrong_frame':
                if probe._prev is not None and probe._prev.shape == images.shape:
                    batch_dict['images'] = probe._prev
                probe._prev = images.clone()
            probe._last_gates = []
            out = probe._orig_forward(batch_dict)
            camera_bev = batch_dict.get('camera_spatial_features_2d')
            fused = batch_dict.get('spatial_features_2d')
            record = {
                'mode': probe.mode,
                'frame': str(batch_dict['frame_id'][0]),
                'camera_bev_absmean': float(camera_bev.abs().mean()),
                'camera_bev_std': float(camera_bev.std()),
                'camera_bev_max': float(camera_bev.abs().max()),
                'fused_absmean': float(fused.abs().mean()),
                'fused_std': float(fused.std()),
                'fused_max': float(fused.abs().max()),
                'gate_mean': float(np.mean([g[0] for g in probe._last_gates])),
                'gate_max': float(np.max([g[1] for g in probe._last_gates])),
                'gate_logit_mean': float(np.mean([g[2] for g in probe._last_gates])),
                'gate_logit_max': float(np.max([g[3] for g in probe._last_gates])),
            }
            probe.records.append(record)
            return out

        self.module.forward = patched_forward
        for layer in self.module.weight_conv:
            layer.register_forward_hook(self._gate_hook)
        return self

    def _gate_hook(self, module, inputs, output):
        weight = torch.sigmoid(output)
        self._last_gates.append((float(weight.mean()), float(weight.max()),
                                 float(output.mean()), float(output.max())))

    def __exit__(self, *exc):
        self.module.forward = self._orig_forward
        return False


def main():
    args = parse_args()
    cfg_from_yaml_file(os.path.join(REPO, args.cfg_file), cfg)
    cfg.LOCAL_RANK = 0
    logger = common_utils.create_logger()
    device = torch.device(args.device)

    dataset, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                          batch_size=1, dist=False, workers=0,
                                          logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=os.path.join(REPO, args.ckpt), logger=logger, to_cpu=True)
    model.to(device).eval()

    fusion = None
    for name, module in model.named_modules():
        if isinstance(module, PaperRGIterFusion):
            fusion = module
            break
    if fusion is None:
        raise RuntimeError('PaperRGIterFusion not found in the model')

    batches = []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        batches.append(batch)

    summary = {}
    with Probe(fusion) as probe:
        for mode in args.modes.split(','):
            probe.mode = mode
            probe._prev = None
            for batch in batches:
                batch = dict(batch)
                load_data_to_gpu(batch) if device.type == 'cuda' else None
                if device.type != 'cuda':
                    batch = {k: (torch.as_tensor(v).to(device) if isinstance(v, np.ndarray)
                                 else v) for k, v in batch.items()}
                with torch.no_grad():
                    model(batch)
            records = [r for r in probe.records if r['mode'] == mode]
            summary[mode] = {key: float(np.mean([r[key] for r in records]))
                             for key in ('camera_bev_absmean', 'camera_bev_std', 'camera_bev_max',
                                         'fused_absmean', 'fused_std', 'fused_max',
                                         'gate_mean', 'gate_max',
                                         'gate_logit_mean', 'gate_logit_max')}
            print('%-12s camera BEV |mean| %8.4f std %8.4f max %8.2f | fused |mean| %8.4f '
                  'std %8.4f max %8.2f | gate mean %.3f max %.3f'
                  % (mode, summary[mode]['camera_bev_absmean'], summary[mode]['camera_bev_std'],
                     summary[mode]['camera_bev_max'], summary[mode]['fused_absmean'],
                     summary[mode]['fused_std'], summary[mode]['fused_max'],
                     summary[mode]['gate_mean'], summary[mode]['gate_max']))
            print('%-12s gate logits: mean %8.3f max %8.3f'
                  % ('', summary[mode]['gate_logit_mean'], summary[mode]['gate_logit_max']))

    with open(args.out, 'w') as f:
        json.dump({'ckpt': args.ckpt, 'batches': len(batches), 'summary': summary,
                   'records': probe.records}, f, indent=2)
    print('\nreport written to %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
