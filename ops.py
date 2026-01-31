"""
操作函数 - 激活函数和插值
"""
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F

__all__ = (
    "trunc_exp",
    "init_density_activation",
    "grid_sample_wrapper",
)


class TruncatedExponential(Function):
    """截断指数函数 (防止梯度爆炸)"""
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, min=-15, max=15))


trunc_exp = TruncatedExponential.apply


def init_density_activation(activation_type: str):
    """初始化密度激活函数"""
    if activation_type == 'trunc_exp':
        return lambda x: trunc_exp(x - 1)
    elif activation_type == 'relu':
        return F.relu
    else:
        raise ValueError(activation_type)


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """网格采样包装器"""
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,
        coords,
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)
    interp = interp.squeeze()
    return interp
