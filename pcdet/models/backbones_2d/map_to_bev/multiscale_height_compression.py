import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleHeightCompression(nn.Module):
    """Project CVFusion's three sparse SECOND scales into the BEV plane.

    A channel-wise max over occupied height cells preserves sparse radar
    responses without materialising the very large 3D dense tensor produced by
    0.05 m voxels.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_keys = list(model_cfg.get(
            'FEATURE_KEYS', ['x_conv3', 'x_conv4', 'x_conv5']))
        in_channels = list(model_cfg.get('IN_CHANNELS', [64, 64, 64]))
        out_channels = list(model_cfg.get('OUT_CHANNELS', [64, 64, 64]))
        self.pool_method = str(model_cfg.get('POOL_METHOD', 'max')).lower()
        if self.pool_method not in ('max', 'height_attention'):
            raise ValueError('POOL_METHOD must be max or height_attention')
        self.occupancy_strides = list(model_cfg.get('OCCUPANCY_STRIDES', [4, 8, 16]))
        self.occupancy_dilation = list(model_cfg.get('OCCUPANCY_DILATION', [7, 5, 3]))
        if not (len(self.feature_keys) == len(in_channels) == len(out_channels) == 3):
            raise ValueError('CVFusion requires exactly three radar BEV scales')
        if len(self.occupancy_strides) != 3 or len(self.occupancy_dilation) != 3:
            raise ValueError('OCCUPANCY_STRIDES and OCCUPANCY_DILATION must have three entries')
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        # Sparse height attention preserves the vertical location discarded by
        # max pooling without materialising a prohibitively large dense 3D
        # tensor.  The final 1x1 fusion starts as an exact max-pool identity so
        # old checkpoints remain a stable warm-start; training can then learn
        # how much height-aware context to mix in.
        self.height_position = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, in_ch), nn.ReLU(inplace=True),
                nn.Linear(in_ch, in_ch))
            for in_ch in in_channels
        ])
        self.height_score = nn.ModuleList([
            nn.Linear(in_ch, 1) for in_ch in in_channels
        ])
        self.height_fusion = nn.ModuleList([
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False)
            for in_ch in in_channels
        ])
        for in_ch, layer in zip(in_channels, self.height_fusion):
            nn.init.zeros_(layer.weight)
            with torch.no_grad():
                channel = torch.arange(in_ch)
                layer.weight[channel, channel, 0, 0] = 1.0
        self.num_bev_features = out_channels[1]

    @staticmethod
    def _sparse_max_to_bev(sparse_tensor):
        features = sparse_tensor.features
        indices = sparse_tensor.indices.long()
        batch_size = sparse_tensor.batch_size
        height, width = (int(sparse_tensor.spatial_shape[1]),
                         int(sparse_tensor.spatial_shape[2]))
        linear = indices[:, 0] * (height * width) + indices[:, 2] * width + indices[:, 3]
        output = features.new_full(
            (batch_size * height * width, features.shape[1]), -torch.inf)
        output.scatter_reduce_(
            0, linear[:, None].expand(-1, features.shape[1]), features,
            reduce='amax', include_self=True)
        # Keep this out-of-place: ScatterReduceBackward retains its output for
        # gradient computation and rejects subsequent in-place modification.
        output = output.masked_fill(torch.isinf(output), 0)
        return output.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def _sparse_height_attention_to_bev(self, sparse_tensor, scale_idx):
        """Pool sparse Z cells with learned, position-aware grouped attention."""
        features = sparse_tensor.features
        indices = sparse_tensor.indices.long()
        batch_size = sparse_tensor.batch_size
        depth, height, width = (int(x) for x in sparse_tensor.spatial_shape)
        linear = indices[:, 0] * (height * width) + indices[:, 2] * width + indices[:, 3]

        z = indices[:, 1].to(features.dtype).unsqueeze(1)
        z = z.mul(2.0 / max(depth - 1, 1)).sub(1.0)
        positioned = features + self.height_position[scale_idx](z)
        scores = self.height_score[scale_idx](positioned).squeeze(1)

        num_cells = batch_size * height * width
        group_max = scores.new_full((num_cells,), -torch.inf)
        group_max.scatter_reduce_(0, linear, scores, reduce='amax', include_self=True)
        weights = torch.exp(scores - group_max[linear])
        denominator = weights.new_zeros((num_cells,))
        denominator.scatter_add_(0, linear, weights)
        weights = weights / denominator[linear].clamp_min(1e-6)

        output = features.new_zeros((num_cells, features.shape[1]))
        output.scatter_add_(
            0, linear[:, None].expand(-1, features.shape[1]),
            positioned * weights[:, None])
        return output.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, batch_dict):
        sparse_features = batch_dict['multi_scale_3d_features']
        bev_features = {}
        occupancy_maps = {}
        voxel_coords = batch_dict['voxel_coords'].long()
        batch_size = int(batch_dict['batch_size'])
        for idx, key in enumerate(self.feature_keys):
            max_bev = self._sparse_max_to_bev(sparse_features[key])
            if self.pool_method == 'height_attention':
                height_bev = self._sparse_height_attention_to_bev(
                    sparse_features[key], idx)
                max_bev = self.height_fusion[idx](torch.cat((max_bev, height_bev), dim=1))
            bev_features['radar_bev_%d' % idx] = self.projections[idx](max_bev)
            height, width = bev_features['radar_bev_%d' % idx].shape[-2:]
            stride = int(self.occupancy_strides[idx])
            batch_index = voxel_coords[:, 0]
            grid_y = torch.div(voxel_coords[:, 2], stride, rounding_mode='floor')
            grid_x = torch.div(voxel_coords[:, 3], stride, rounding_mode='floor')
            valid = ((batch_index >= 0) & (batch_index < batch_size) &
                     (grid_y >= 0) & (grid_y < height) &
                     (grid_x >= 0) & (grid_x < width))
            occupancy = bev_features['radar_bev_%d' % idx].new_zeros(
                (batch_size, 1, height, width))
            occupancy[batch_index[valid], 0, grid_y[valid], grid_x[valid]] = 1.0
            kernel = int(self.occupancy_dilation[idx])
            if kernel > 1:
                occupancy = F.max_pool2d(
                    occupancy, kernel_size=kernel, stride=1, padding=kernel // 2)
            occupancy_maps['radar_occupancy_%d' % idx] = occupancy
        batch_dict['multi_scale_2d_features'] = bev_features
        batch_dict['radar_occupancy_maps'] = occupancy_maps
        batch_dict['spatial_features_2d'] = bev_features['radar_bev_1']
        batch_dict['spatial_features_stride'] = 8
        return batch_dict
