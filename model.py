"""
模型定义 - 基于 hyperrgb_mix_fleid.py
HyperNetwork + K-Planes + SKNet Mix 结构
"""
import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from plenoxels.ops.interpolation import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]"""
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    """将点归一化到 [-1, 1] 范围"""
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    """初始化网格参数"""
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = []
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        )
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)
    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: List[torch.Tensor],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    """多尺度特征插值"""
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    batch_size = pts.shape[0]
    if num_levels is None:
        num_levels = len(ms_grids)
    interfeaturelist = []
    
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            interfeaturelist.append(interp_out_plane.view(batch_size, 1, 8, 8))
    
    return interfeaturelist


class KPlaneField(nn.Module):
    """
    K-Planes 场模型
    
    核心结构:
    1. ImgEncoder: CNN 图像编码器
    2. mlp0-mlp5: 6 个 MLP 生成 6 个特征平面
    3. SKNet Mix: 特征混合模块
    4. sigma_net + color_net: 解码器
    """
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        features_channel: int,
        M=6,
        G=1,
        r=16,
        stride=1,
        L=32,
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        self.needsetgrids = True
        self.avg_flag = False

        # ============ SKNet Mix 模块 ============
        d = max(int(features_channel / r), L)
        self.M = M
        self.features_channel = features_channel
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features_channel, features_channel, kernel_size=3, stride=stride, padding=1, dilation=1, groups=G, bias=False),
                nn.BatchNorm2d(features_channel),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(features_channel, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features_channel, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

        # ============ 初始化平面 ============
        self.grids = []
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]

            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )

            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]

            self.grids.append(gp)

        log.info(f"Initialized model grids: {self.grids[0][0].shape}")

        # ============ Appearance embedding ============
        self.use_average_appearance_embedding = True
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = nn.Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # ============ 方向编码器 ============
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        # ============ 图像编码器 ============
        self.ImgEncoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (16, 256, 256)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 20, kernel_size=3, stride=2, padding=1),  # (20, 64, 64)
            nn.ReLU()
        )

        # ============ 平面生成 MLP ============
        self.mlp0 = self._create_mlp(90, 90)   # θ_h × θ_d
        self.mlp1 = self._create_mlp(180, 90)  # φ_d × θ_h
        self.mlp2 = self._create_mlp(39, 90)   # time × θ_h
        self.mlp3 = self._create_mlp(180, 90)  # φ_d × θ_d
        self.mlp4 = self._create_mlp(39, 90)   # time × θ_d
        self.mlp5 = self._create_mlp(39, 180)  # time × φ_d

        # ============ 解码器网络 ============
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers + 2,
                },
            )
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 2,
                },
            )
        else:
            self.geo_feat_dim = 32
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 2,
                },
            )
            self.in_dim_color = self.geo_feat_dim
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 128,
                    "n_hidden_layers": 3,
                },
            )

    def mix_feature(self, x):
        """SKNet 风格的特征混合"""
        batch_size = x[0].shape[0]
        num = 0
        feats = []
        for conv in self.convs:
            feats.append(conv(x[num]))
            num += 1

        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features_channel, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        feats_V = feats_V.view(batch_size, 64)

        return feats_V

    def _create_mlp(self, axis1, axis2):
        """创建平面生成 MLP"""
        mlp = nn.Sequential(
            nn.Linear(20 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64 * axis1 * axis2),
        )
        return mlp

    def setgrids(self, x):
        """从图像编码生成 K-Planes"""
        x1 = self.ImgEncoder(x)
        x1 = x1.view(x.size(0), -1)

        self.grids[0][0] = self.mlp0(x1).view(-1, 64, 90, 90)
        self.grids[0][1] = self.mlp1(x1).view(-1, 64, 180, 90)
        self.grids[0][3] = self.mlp3(x1).view(-1, 64, 180, 90)
        self.grids[0][2] = self.mlp2(x1).view(-1, 64, 39, 90)
        self.grids[0][4] = self.mlp4(x1).view(-1, 64, 39, 90)
        self.grids[0][5] = self.mlp5(x1).view(-1, 64, 39, 180)

        if self.avg_flag:
            # RGB 模式: 对时间维度取平均
            self.grids[0][2] = self.mlp2(x1).view(-1, 64, 39, 90).mean(dim=2, keepdim=True)
            self.grids[0][4] = self.mlp4(x1).view(-1, 64, 39, 90).mean(dim=2, keepdim=True)
            self.grids[0][5] = self.mlp5(x1).view(-1, 64, 39, 180).mean(dim=2, keepdim=True)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """计算密度特征"""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2
        else:
            pts = normalize_aabb(pts, self.aabb)

        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = (timestamps * 2 / 39) - 1
            pts = torch.cat((pts, timestamps), dim=-1)

        pts = pts.reshape(-1, pts.shape[-1])
        flist = interpolate_ms_features(
            pts, ms_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)

        features = self.mix_feature(flist)

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)
        else:
            features = self.sigma_net(features)

        return features

    def forward(self,
                img: torch.Tensor,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        """前向传播"""
        if self.needsetgrids:
            self.setgrids(img)
            self.needsetgrids = False

        features = self.get_density(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [features.view(-1, self.geo_feat_dim)]

        color_features = torch.cat(color_features, dim=-1)
        rgb = self.color_net(color_features).to(pts).view(n_rays, n_samples, 1)

        return {"rgb": rgb}

    def get_params1(self):
        """获取参数分组 (用于分组学习率)"""
        nn_params = [
            self.mlp0.named_parameters(prefix="mlp0"),
            self.mlp1.named_parameters(prefix="mlp1"),
            self.mlp2.named_parameters(prefix="mlp2"),
            self.mlp3.named_parameters(prefix="mlp3"),
            self.mlp4.named_parameters(prefix="mlp4"),
            self.mlp5.named_parameters(prefix="mlp5"),
            self.ImgEncoder.named_parameters(prefix="ImgEncoder"),
        ]
        spec_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
            self.direction_encoder.named_parameters(prefix="color_net"),
        ]
        nn_params = {k: v for plist in nn_params for k, v in plist}
        spec_params = {k: v for plist in spec_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in spec_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "other": list(other_params.values()),
            "spec": list(spec_params.values()),
        }
