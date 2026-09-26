"""Camera-contribution diagnostics on the frozen Stage1 checkpoint.

The technical report asks for a sensitivity check of the *existing* model: how
much does the detection output depend on the image content?  Three test-time
perturbations are provided (no retraining, no weight change):

``zero``        images replaced by a constant black frame (after normalisation
                this is the dataset-mean image, i.e. the smallest possible
                camera signal that still exercises the depth head);
``noise``       images replaced by Gaussian noise - a control that keeps the
                feature statistics non-degenerate but removes all semantics;
``wrong_frame`` each sample receives the image of the *previous* validation
                frame, i.e. the correct scene statistics with wrong geometry.

These are inference-only ablations.  They measure how much the trained network
*uses* the camera, which is not the same as the accuracy a retrained radar-only
model would reach - the report says so explicitly - so the numbers are reported
as sensitivity, never as a radar-only baseline.

Usage::

    python experiments/stage1_audit_20260926/run_image_ablation.py \
        --cfg_file tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml \
        --ckpt output/cfgs/vod_models/vod_cvfusion_paper_stage1_zvalid/paper_stage1_zvalid_v1/ckpt/checkpoint_epoch_78.pth \
        --mode zero --extra_tag ablation_image_zero
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.models.backbones_2d.paper_rgiter_fusion import PaperRGIterFusion  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402
from eval_utils import eval_utils  # noqa: E402

MODES = ('none', 'zero', 'noise', 'wrong_frame')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg_file', default='tools/cfgs/vod_models/vod_cvfusion_active_stage1_best.yaml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--extra_tag', default=None)
    p.add_argument('--eval_tag', default=None)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max_batches', type=int, default=0, help='debug: stop after N batches')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return p.parse_args()


def install_image_hook(mode, logger, batch_size):
    """Wrap PaperRGIterFusion.forward so that ``batch_dict['images']`` is perturbed.

    ``batch_dict['images']`` is read exactly once in the whole model
    (paper_rgiter_fusion.py), so replacing it here is equivalent to replacing
    the camera input, and nothing else sees the perturbation.
    """
    if mode == 'none':
        return
    original = PaperRGIterFusion.forward
    state = {'prev': None}

    def patched(self, batch_dict):
        images = batch_dict['images']
        if mode == 'zero':
            batch_dict['images'] = torch.zeros_like(images)
        elif mode == 'noise':
            # images are CHW float32 in [0, 1]; keep the same range so the
            # perturbation is "plausible pixels, no semantics".
            generator = torch.Generator(device=images.device).manual_seed(1234)
            batch_dict['images'] = torch.rand(images.shape, generator=generator,
                                              device=images.device, dtype=images.dtype)
        elif mode == 'wrong_frame':
            if state['prev'] is not None and state['prev'].shape == images.shape:
                batch_dict['images'] = state['prev']
            state['prev'] = images.clone()
        return original(self, batch_dict)

    PaperRGIterFusion.forward = patched
    logger.info('installed image perturbation hook: %s (batch_size=%d)' % (mode, batch_size))


class _BatchLimiter:
    """Wrap a dataloader so that only the first ``n`` batches are consumed."""

    def __init__(self, loader, n):
        self.loader = loader
        self.n = n
        self.dataset = loader.dataset

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                break
            yield batch

    def __len__(self):
        return min(self.n, len(self.loader))


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    cfg_from_yaml_file(os.path.join(REPO, args.cfg_file), cfg)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    cfg.LOCAL_RANK = 0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    extra_tag = args.extra_tag or ('ablation_image_%s' % args.mode)
    eval_tag = args.eval_tag or datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / extra_tag
    eval_output_dir = output_dir / 'eval' / ('epoch_%s' % _ckpt_epoch(args.ckpt)) / \
        cfg.DATA_CONFIG.DATA_SPLIT['test'] / eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_ablation_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info('mode=%s ckpt=%s' % (args.mode, args.ckpt))
    logger.info('output=%s' % eval_output_dir)
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size, dist=False, workers=args.workers,
        logger=logger, training=False)
    if args.max_batches:
        test_loader = _BatchLimiter(test_loader, args.max_batches)
        # the built-in AP call needs detections for every GT frame; in a debug
        # subset it would assert, so short-circuit it (the recall stats and the
        # result pickle - the part we care about - are still produced).
        test_set.evaluation = lambda *a, **k: ('debug subset: AP skipped', {})
        logger.info('DEBUG: limiting evaluation to %d batches' % args.max_batches)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=os.path.join(REPO, args.ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    install_image_hook(args.mode, logger, args.batch_size)

    t0 = time.time()
    tb_dict = eval_utils.eval_one_epoch(cfg, model, test_loader, _ckpt_epoch(args.ckpt),
                                        logger, dist_test=False, save_to_file=False,
                                        result_dir=eval_output_dir)
    logger.info('ablation %s finished in %.1f s' % (args.mode, time.time() - t0))
    logger.info('result pickle: %s' % (eval_output_dir / 'result.pkl'))
    for key in sorted(tb_dict):
        if key.endswith('moderate') or key.endswith('moderate_R40'):
            logger.info('  %-40s %.4f' % (key, tb_dict[key]))
    return 0


def _ckpt_epoch(ckpt):
    import re
    name = os.path.basename(ckpt)
    found = re.findall(r'\d+', name)
    return found[-1] if found else 'no_number'


if __name__ == '__main__':
    sys.exit(main())
