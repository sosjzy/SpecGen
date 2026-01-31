"""
K-Planes BRDF 核心模块

主要组件:
- model: KPlaneField 模型定义
- train: 训练脚本
- inference: 推理/渲染脚本
- ops: 操作函数 (激活函数、插值)
- coords: Rusinkiewicz 坐标转换
- config: 配置文件
"""

from .model import KPlaneField
from .ops import init_density_activation, grid_sample_wrapper, trunc_exp
from .coords import get_rusinkiewicz_angles, get_io_from_rusinkiewicz_angles
from .train import Trainer, SpectralLoss, BRDFDataset, train_main
from .inference import load_model, render_single_image, batch_render

__all__ = [
    'KPlaneField',
    'init_density_activation',
    'grid_sample_wrapper',
    'trunc_exp',
    'get_rusinkiewicz_angles',
    'get_io_from_rusinkiewicz_angles',
    'Trainer',
    'SpectralLoss',
    'BRDFDataset',
    'train_main',
    'load_model',
    'render_single_image',
    'batch_render',
]
