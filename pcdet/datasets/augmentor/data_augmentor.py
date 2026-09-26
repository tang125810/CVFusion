from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enabled = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_enable=True
            )
            if enabled and 'lidar_aug_matrix' in data_dict:
                transform = np.eye(4, dtype=np.float32)
                axis_idx = 1 if cur_axis == 'x' else 0
                transform[axis_idx, axis_idx] = -1
                data_dict['lidar_aug_matrix'] = transform @ data_dict['lidar_aug_matrix']

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, angle = augmentor_utils.global_rotation(
            data_dict['gt_boxes'],
            data_dict['points'],
            rot_range=rot_range,
            return_rotation=True
        )
        if 'lidar_aug_matrix' in data_dict:
            c, sin = np.cos(angle), np.sin(angle)
            transform = np.eye(4, dtype=np.float32)
            transform[:2, :2] = np.array([[c, -sin], [sin, c]], dtype=np.float32)
            data_dict['lidar_aug_matrix'] = transform @ data_dict['lidar_aug_matrix']

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, scale = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'],
            return_scale=True
        )
        if 'lidar_aug_matrix' in data_dict:
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] *= scale
            data_dict['lidar_aug_matrix'] = transform @ data_dict['lidar_aug_matrix']
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
