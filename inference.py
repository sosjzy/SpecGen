"""
推理/渲染脚本 - 基于 hyperval_single.py
用于使用训练好的模型进行 BRDF 渲染
"""
import torch
import torch.nn as nn
import numpy as np
import math
import os
import sys
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_trans

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from model import KPlaneField
from ops import init_density_activation
from coords import get_rusinkiewicz_angles


def calc_rotate_angle(vec):
    """
    计算向量的旋转角度
    :param np.ndarray vec: (3,)
    :return: (theta, phi)
    """
    assert np.linalg.norm(vec) > 0

    theta = np.arccos(vec[2] / np.linalg.norm(vec))
    if np.linalg.norm([vec[0], vec[1]]) == 0:
        phi = 0.0
    else:
        phi = np.sign(vec[1]) * np.arccos(vec[0] / np.linalg.norm([vec[0], vec[1]]))

    return -theta, -phi


def normalize_to_neg1_pos1(array):
    """归一化到 [-1, 1] 范围"""
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array


def load_model(checkpoint_path, config_path, device):
    """加载训练好的模型"""
    # 读取配置
    config = {}
    with open(config_path, 'r') as file:
        exec(file.read(), config)

    aabb = torch.tensor([[0, 0, 0], [90, 90, 180]])
    density_act = init_density_activation('trunc_exp')

    model = KPlaneField(
        aabb=aabb,
        grid_config=config['config']['grid_config'],
        concat_features_across_scales=False,
        multiscale_res=[1],
        use_appearance_embedding=False,
        appearance_embedding_dim=0,
        spatial_distortion=None,
        density_activation=density_act,
        linear_decoder=False,
        linear_decoder_layers=2,
        num_images=1,
        features_channel=1
    )

    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # 移除 'module.' 前缀 (如果有)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    
    return model


def render_single_image(
    model,
    img_path,
    N_map_file,
    mask_file,
    L_file,
    out_dir,
    obj_name,
    device,
    num_wavelengths=20
):
    """
    渲染单张图像
    
    Args:
        model: 训练好的模型
        img_path: 输入图像路径
        N_map_file: 法向量图文件
        mask_file: 掩码文件
        L_file: 光照方向文件
        out_dir: 输出目录
        obj_name: 对象名称
        device: 设备
        num_wavelengths: 波长数量 (默认20)
    """
    # 加载数据
    N_map = np.load(N_map_file)
    mask = np.load(mask_file)
    N = N_map[mask > 0]  # (P, 3)
    N = N / np.linalg.norm(N, axis=1, keepdims=True)

    L = np.loadtxt(L_file)
    L = L.reshape(1, 3)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)

    v = np.array([0., 0., 1.], dtype=np.float64)

    # 加载输入图像
    image = Image.open(img_path)
    image_array = np.array(image)
    img = normalize_to_neg1_pos1(image_array)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)

    # 创建输出目录
    out_path = os.path.join(out_dir, obj_name)
    os.makedirs(out_path, exist_ok=True)
    print(f'===== {obj_name} start =====')

    L_ = L
    ret = []

    for i in tqdm(range(len(N)), desc="Rendering"):
        theta, phi = calc_rotate_angle(N[i])
        R = R_trans.from_euler('zyx', [phi, theta, 0], degrees=False).as_matrix()
        wo = (R @ v).tolist()

        ret1 = np.zeros([len(L_), num_wavelengths], dtype=float)

        for j in range(len(L_)):
            wi = (R @ L_[j]).tolist()

            wi = np.array(wi)
            wo = np.array(wo)
            r = get_rusinkiewicz_angles(wi, wo)

            theta_h = r[0]
            theta_d = r[1]
            phi_d = r[2]
            if phi_d < 0:
                phi_d = -phi_d

            # 转换为角度 (整数)
            theta_h = int(theta_h / math.pi * 180)
            theta_d = int(theta_d / math.pi * 180)
            phi_d = int(phi_d / math.pi * 180)

            # 准备输入
            inputs = np.array([theta_h, theta_d, phi_d], dtype=np.float32)
            inputs = np.tile(inputs, (num_wavelengths, 1))
            inputs = inputs.reshape(1, num_wavelengths, 3)
            inputs = torch.tensor(inputs).to(device)

            # 时间/波长采样点
            times = np.array([[i * 2 for i in range(num_wavelengths)]])
            times = np.expand_dims(times, axis=-1)
            times = torch.tensor(times).to(device)

            with torch.no_grad():
                outputs = model(img, inputs, times)
                rgb_output = outputs['rgb'].float()
                rgb_output_cpu = rgb_output.cpu()
                rgb_output_cpu = rgb_output_cpu.squeeze(dim=2)
                rho = rgb_output_cpu.numpy()

            # 边界检查
            if theta_h >= 90 or theta_d >= 90 or phi_d >= 180:
                rho = np.zeros((1, num_wavelengths))

            rho[rho < 0] = 0
            ret1[j] = rho[0]

        ret.append((i, ret1))

    ret.sort(key=lambda x: x[0])
    M = np.array([x[1] for x in ret], dtype=np.float64)

    # 重塑输出
    imgs = np.zeros((1, N_map.shape[0], N_map.shape[1], num_wavelengths))
    imgs[:, mask > 0] = M.transpose(1, 0, 2)

    # 保存结果
    print('Saving results...')
    np.save(os.path.join(out_path, 'normal.npy'), N_map)
    np.save(os.path.join(out_path, 'imgs.npy'), imgs)
    np.savetxt(os.path.join(out_path, 'light_directions.txt'), L_)
    np.save(os.path.join(out_path, 'mask.npy'), mask)

    print(f'===== {obj_name} done =====')
    return out_path


def batch_render(
    checkpoint_path,
    config_path,
    image_list_file,
    image_dir,
    N_map_file,
    mask_file,
    L_file,
    out_dir
):
    """批量渲染"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model(checkpoint_path, config_path, device)
    
    # 读取图像列表
    with open(image_list_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]

    for img_name in lines:
        img_path = os.path.join(image_dir, img_name + ".png")
        print(f"{img_name} is starting...")
        
        render_single_image(
            model=model,
            img_path=img_path,
            N_map_file=N_map_file,
            mask_file=mask_file,
            L_file=L_file,
            out_dir=out_dir,
            obj_name=img_name,
            device=device
        )


if __name__ == '__main__':
    # 示例用法
    batch_render(
        checkpoint_path="./checkpoints/best.pth",
        config_path="./config.py",
        image_list_file="./val.txt",
        image_dir="./images",
        N_map_file="./renderdata/sphere512.npy",
        mask_file="./renderdata/sphere512_mask.npy",
        L_file="./renderdata/L_0.txt",
        out_dir="./output"
    )
