import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t

from .base_bev_backbone import BaseBEVBackbone


class ResidualBEVBlock(nn.Module):
    """A receptive-field block for the proposal feature tower."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        # Identity initialization makes this safe when warm-starting an older
        # Stage1 checkpoint that fed RGIter directly into AnchorHeadSingle.
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, features):
        residual = features
        features = self.relu(self.bn1(self.conv1(features)))
        features = self.bn2(self.conv2(features))
        return self.relu(features + residual)


class PaperRGIterFusion(nn.Module):
    """CVFusion Stage1 RGIter-BEV fusion following paper equations (1)-(5)."""

    def __init__(self, model_cfg, input_channels, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.depth_bins = int(model_cfg.get('DEPTH_BINS', 64))
        self.depth_min = float(model_cfg.get('DEPTH_MIN', 1.0))
        self.depth_max = float(model_cfg.get('DEPTH_MAX', 51.2))
        self.depth_loss_weight = float(model_cfg.get('DEPTH_LOSS_WEIGHT', 0.0))
        self.gate_loss_weight = float(model_cfg.get('GATE_LOSS_WEIGHT', 0.0))
        camera_channels = int(model_cfg.get('IMAGE_CHANNELS', 64))
        radar_channels = list(model_cfg.get('RADAR_SCALE_CHANNELS', [64, 64, 64]))
        output_channels = int(model_cfg.get('OUTPUT_CHANNELS', 128))
        rpn_num_blocks = int(model_cfg.get('RPN_NUM_BLOCKS', 0))
        self.radar_keys = list(model_cfg.get(
            'RADAR_SCALE_KEYS', ['radar_bev_0', 'radar_bev_1', 'radar_bev_2']))
        if len(radar_channels) != 3 or len(self.radar_keys) != 3:
            raise ValueError('RGIter fusion requires exactly three radar scales')
        self.num_bev_features = output_channels

        network = swin_t(weights=None)
        weights_path = str(model_cfg.get('PRETRAINED', ''))
        if weights_path:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError('Swin-T weights not found: %s' % weights_path)
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
            network.load_state_dict(state.get('model', state), strict=True)
        # Paper FI has H/4 x W/4 resolution, corresponding to Swin stage 0.
        self.image_backbone = network.features[:2]
        for parameter in self.image_backbone.parameters():
            parameter.requires_grad_(False)
        self.image_backbone.eval()
        self.register_buffer(
            'image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.depth_head = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, self.depth_bins, 1))
        self.image_proj = nn.Conv2d(96, camera_channels, 1)
        self.weight_conv = nn.ModuleList([
            nn.Conv2d(channels, 1, 3, padding=1) for channels in radar_channels])
        self.iter_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(camera_channels, camera_channels, 3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(camera_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            for _ in range(2)])
        self.fuse_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels + camera_channels, output_channels, 3,
                          padding=1, bias=False),
                nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            for channels in radar_channels])
        self.out_conv = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True))
        self.rpn_refine = nn.Sequential(*[
            ResidualBEVBlock(output_channels) for _ in range(rpn_num_blocks)
        ])
        rpn_backbone_cfg = model_cfg.get('RPN_BACKBONE', None)
        if rpn_backbone_cfg is not None:
            self.rpn_backbone = BaseBEVBackbone(
                model_cfg=rpn_backbone_cfg, input_channels=output_channels)
            self.num_bev_features = self.rpn_backbone.num_bev_features
        else:
            self.rpn_backbone = None
        self.depth_loss = None
        self.gate_loss = None

    def reset_guidance_gates(self):
        """Clear the saturated solution while preserving a usable camera signal."""
        for layer in self.weight_conv:
            nn.init.zeros_(layer.weight)
            # Balanced occupancy BCE has a neutral prior of 0.5.  This also
            # avoids an abrupt removal of camera features during fine-tuning.
            nn.init.zeros_(layer.bias)

    def _radar_depth_loss(self, depth_logits, batch_dict):
        """Supervise image depth with radar points projected into the image."""
        points = batch_dict.get('points', None)
        if points is None or points.numel() == 0:
            return depth_logits.sum() * 0.0
        batch, _, feat_h, feat_w = depth_logits.shape
        image_h, image_w = self.model_cfg.IMAGE_SIZE
        losses = []
        for batch_idx in range(batch):
            xyz = points[points[:, 0].long() == batch_idx, 1:4].float()
            if xyz.numel() == 0:
                continue
            xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
            # Points have already received world augmentation. Undo it before
            # applying the original lidar-to-camera calibration.
            xyz_lidar = xyz_h @ torch.linalg.inv(
                batch_dict['lidar_aug_matrix'][batch_idx].float()).T
            xyz_camera = xyz_lidar @ torch.linalg.inv(
                batch_dict['camera_to_lidar'][batch_idx].float()).T
            depth = xyz_camera[:, 2]
            projection = xyz_camera[:, :3] @ batch_dict[
                'camera_intrinsics'][batch_idx, :3, :3].float().T
            u = projection[:, 0] / projection[:, 2].clamp_min(1e-5)
            v = projection[:, 1] / projection[:, 2].clamp_min(1e-5)
            grid_x = torch.floor((u + 0.5) * feat_w / float(image_w)).long()
            grid_y = torch.floor((v + 0.5) * feat_h / float(image_h)).long()
            valid = ((depth >= self.depth_min) & (depth <= self.depth_max) &
                     (grid_x >= 0) & (grid_x < feat_w) &
                     (grid_y >= 0) & (grid_y < feat_h))
            if not valid.any():
                continue
            linear = grid_y[valid] * feat_w + grid_x[valid]
            nearest = depth.new_full((feat_h * feat_w,), torch.inf)
            nearest.scatter_reduce_(0, linear, depth[valid], reduce='amin', include_self=True)
            supervised = torch.isfinite(nearest)
            target = torch.round(
                (nearest[supervised] - self.depth_min) * (self.depth_bins - 1) /
                (self.depth_max - self.depth_min)).long().clamp(0, self.depth_bins - 1)
            logits = depth_logits[batch_idx].reshape(self.depth_bins, -1).T[supervised]
            losses.append(F.cross_entropy(logits, target))
        return torch.stack(losses).mean() if losses else depth_logits.sum() * 0.0

    def train(self, mode=True):
        super().train(mode)
        self.image_backbone.eval()
        return self

    def _lift_to_bev(self, image_features, depth_probability, intrinsics,
                     camera_to_lidar, lidar_aug_matrix, bev_h, bev_w):
        batch, channels, feat_h, feat_w = image_features.shape
        device, dtype = image_features.device, image_features.dtype
        image_h, image_w = self.model_cfg.IMAGE_SIZE
        depths = torch.linspace(self.depth_min, self.depth_max, self.depth_bins,
                                device=device, dtype=torch.float32)
        v, u = torch.meshgrid(
            torch.arange(feat_h, device=device, dtype=torch.float32),
            torch.arange(feat_w, device=device, dtype=torch.float32),
            indexing='ij')
        u = (u + 0.5) * (float(image_w) / feat_w) - 0.5
        v = (v + 0.5) * (float(image_h) / feat_h) - 0.5
        pixels = torch.stack((u, v, torch.ones_like(u)), 0).reshape(3, -1)
        output = image_features.new_zeros((batch, channels, bev_h, bev_w))
        x_min, y_min, z_min = self.point_cloud_range[:3]
        z_max = self.point_cloud_range[5]
        x_res = (self.point_cloud_range[3] - x_min) / bev_w
        y_res = (self.point_cloud_range[4] - y_min) / bev_h

        for batch_idx in range(batch):
            rays = torch.linalg.solve(intrinsics[batch_idx].float(), pixels)
            xyz_camera = (rays[:, None, :] * depths[None, :, None]).permute(1, 2, 0)
            xyz_h = torch.cat((xyz_camera, torch.ones(
                (*xyz_camera.shape[:-1], 1), device=device)), dim=-1)
            xyz_lidar = xyz_h @ camera_to_lidar[batch_idx].float().T
            xyz_lidar = xyz_lidar @ lidar_aug_matrix[batch_idx].float().T
            xyz_lidar = xyz_lidar[..., :3]
            grid_x = torch.floor((xyz_lidar[..., 0] - x_min) / x_res).long()
            grid_y = torch.floor((xyz_lidar[..., 1] - y_min) / y_res).long()
            # A depth hypothesis belongs to the camera BEV volume only when
            # its complete 3D location is inside the detector range.  Keeping
            # hypotheses that are below/above the voxel volume collapses
            # unrelated parts of an image ray onto valid XY cells and badly
            # contaminates the lifted feature map.
            valid = ((grid_x >= 0) & (grid_x < bev_w) &
                     (grid_y >= 0) & (grid_y < bev_h) &
                     (xyz_lidar[..., 2] >= z_min) &
                     (xyz_lidar[..., 2] < z_max))
            linear = (grid_y * bev_w + grid_x)[valid]
            probability = depth_probability[batch_idx].reshape(
                self.depth_bins, -1)[valid]
            pixel_features = image_features[batch_idx].permute(
                1, 2, 0).reshape(-1, channels)
            lifted = pixel_features.unsqueeze(0).expand(
                self.depth_bins, -1, -1)[valid]
            output[batch_idx].view(channels, -1).T.index_add_(
                0, linear, (lifted.float() * probability[:, None].float()).to(dtype))
        return output

    def forward(self, batch_dict):
        radar_scales = batch_dict['multi_scale_2d_features']
        radar_bevs = [radar_scales[key] for key in self.radar_keys]
        with torch.no_grad():
            images = (batch_dict['images'] - self.image_mean) / self.image_std
            image_features = self.image_backbone(images).permute(0, 3, 1, 2).contiguous()
        depth_logits = self.depth_head(image_features.float())
        depth_probability = torch.softmax(depth_logits, dim=1)
        image_features = self.image_proj(image_features.float())
        batch_dict['image_front_features'] = image_features
        camera_bev = self._lift_to_bev(
            image_features, depth_probability, batch_dict['camera_intrinsics'],
            batch_dict['camera_to_lidar'], batch_dict['lidar_aug_matrix'],
            radar_bevs[0].shape[-2], radar_bevs[0].shape[-1])

        fused = []
        camera_scale = camera_bev
        gate_losses = []
        occupancy_maps = batch_dict.get('radar_occupancy_maps', {})
        for idx, radar_bev in enumerate(radar_bevs):
            if camera_scale.shape[-2:] != radar_bev.shape[-2:]:
                camera_scale = F.interpolate(camera_scale, radar_bev.shape[-2:],
                                             mode='bilinear', align_corners=False)
            weight_logits = self.weight_conv[idx](radar_bev)
            weight = torch.sigmoid(weight_logits)
            occupancy = occupancy_maps.get('radar_occupancy_%d' % idx, None)
            if self.training and occupancy is not None and self.gate_loss_weight > 0:
                positive = occupancy.sum().clamp_min(1.0)
                negative = occupancy.numel() - positive
                pos_weight = (negative / positive).clamp(1.0, 20.0)
                gate_losses.append(F.binary_cross_entropy_with_logits(
                    weight_logits, occupancy, pos_weight=pos_weight))
            weighted_camera = camera_scale * weight
            fused.append(self.fuse_conv[idx](
                torch.cat((radar_bev, weighted_camera), dim=1)))
            if idx < 2:
                camera_scale = self.iter_conv[idx](weighted_camera)

        # Paper Eq. (5): Down(k=0), k=1, Up(k=2), then concatenate.
        target_size = radar_bevs[1].shape[-2:]
        aligned = [F.interpolate(feature, target_size, mode='bilinear',
                                 align_corners=False)
                   if feature.shape[-2:] != target_size else feature
                   for feature in fused]
        batch_dict['camera_spatial_features_2d'] = camera_bev
        fusion_features = self.out_conv(torch.cat(aligned, dim=1))
        proposal_features = self.rpn_refine(fusion_features)
        if self.rpn_backbone is not None:
            # Keep the radar multi-scale dictionary intact: the standard
            # backbone uses the same key for its internal feature pyramid.
            proposal_dict = self.rpn_backbone({'spatial_features': proposal_features})
            proposal_features = proposal_dict['spatial_features_2d']
            batch_dict['rpn_multi_scale_2d_features'] = proposal_dict[
                'multi_scale_2d_features']
        batch_dict['spatial_features_2d'] = proposal_features
        if self.training:
            self.depth_loss = self._radar_depth_loss(depth_logits, batch_dict)
            self.gate_loss = (torch.stack(gate_losses).mean() if gate_losses
                              else depth_logits.sum() * 0.0)
        else:
            self.depth_loss = None
            self.gate_loss = None
        return batch_dict
