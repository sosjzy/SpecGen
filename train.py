"""
训练脚本 - 基于 testgenerator.py
用于训练 HyperNetwork BRDF 预测模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Sequence, Any

import pandas as pd
import torch
import torch.utils.data

import sys
import os
import logging

import random
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from torch.cuda.amp import GradScaler, autocast

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录
parent_dir = os.path.dirname(current_dir)
# 将上级目录添加到sys.path
sys.path.append(parent_dir)
sys.path.append(current_dir)

from model import KPlaneField
from ops import init_density_activation

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

L1_loss = torch.nn.L1Loss()
L2_loss = torch.nn.MSELoss()


class SpectralLoss(nn.Module):
    """自定义光谱损失函数: MSE + Scale + TV"""
    def __init__(self, tv_weight=1):
        super(SpectralLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, pts, R, gt, mse_weight, scale_weight):
        # MSE 损失
        mse_loss = L2_loss(R, gt)

        # Scale 损失 (尺度不变性)
        scale = torch.matmul(R, gt.T) / torch.matmul(gt, gt.T)
        scale_loss = L2_loss(gt, torch.matmul(scale, R))

        # TV 损失 (总变分，平滑约束)
        pts = pts[0]
        R = R.T
        batch_size = pts.size(0)
        
        diff_pts = pts[1:] - pts[:-1]
        dist = torch.norm(diff_pts, dim=1)
        r_diff = torch.abs(R[1:] - R[:-1])
        r_diff = r_diff.squeeze(1)
        
        tv_loss = torch.sum(r_diff / (dist + 1))
        tv_loss /= batch_size
        
        if tv_loss < 0.003:
            tv_loss = 0

        return mse_weight * mse_loss + scale_weight * scale_loss + self.tv_weight * tv_loss


def normalize_to_neg1_pos1(array):
    """归一化到 [-1, 1] 范围"""
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array


def remove_module_prefix(state_dict):
    """移除 DataParallel 添加的 'module.' 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_pretrained_model(model, checkpoint_path):
    """加载预训练模型"""
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained model from {checkpoint_path}")
    else:
        print(f"Pre-trained model path {checkpoint_path} does not exist, training from scratch.")


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, checkpoint_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        self.writer = SummaryWriter(os.path.join(checkpoint_dir, 'tf_logs'))

    def detect_gradients(self):
        """检测梯度消失/爆炸"""
        if isinstance(self.model, torch.nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        params = model.get_params1()
        nn_total_norm = 0
        other_total_norm = 0

        for param in params['nn']:
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                nn_total_norm += param_norm.item() ** 2
        nn_total_norm = nn_total_norm ** (1. / 2)

        for param in params['other']:
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                other_total_norm += param_norm.item() ** 2
        other_total_norm = other_total_norm ** (1. / 2)

    def train(self, num_epochs, images_list):
        """训练主循环"""
        self.model.to(self.device)

        for epoch in range(num_epochs):
            logging.info(f"Start epoch: {epoch}")
            self.model.train()
            train_loss = 0.0
            stepnum = 0

            for batch in self.train_loader:
                inputs_batch, times_batch, targets_batch, brdfid_batch = batch

                grouped_inputs = defaultdict(list)
                grouped_times = defaultdict(list)
                grouped_targets = defaultdict(list)

                for i, brdfid in enumerate(brdfid_batch):
                    grouped_inputs[brdfid.item()].append(inputs_batch[i])
                    grouped_times[brdfid.item()].append(times_batch[i])
                    grouped_targets[brdfid.item()].append(targets_batch[i])

                total_loss_val = 0.0
                total_samples = 0
                self.optimizer.zero_grad()
                total_loss_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)
                
                mse_weight = 1

                for brdfid, inputs_group in grouped_inputs.items():
                    if brdfid > 0:
                        self.model.avg_flag = True
                        mse_weight = 1
                    else:
                        self.model.avg_flag = False
                        mse_weight = 1
                    
                    img_np = images_list[brdfid]
                    img = torch.tensor(img_np, dtype=torch.float32, device=self.device).permute(2, 0, 1)
                    img = img.unsqueeze(0)

                    inputs_group = torch.stack(inputs_group).unsqueeze(0).float().to(self.device)
                    times_group = torch.stack(grouped_times[brdfid]).unsqueeze(0).float().to(self.device)
                    targets_group = torch.stack(grouped_targets[brdfid]).unsqueeze(0).float().to(self.device)

                    with autocast():
                        outputs = self.model(img, inputs_group, times_group)
                        rgb = outputs['rgb'].float().squeeze(-1)
                        scale_weight = 1 - mse_weight
                        pts = torch.cat((inputs_group, times_group), dim=2)
                        sub_loss = self.criterion(pts, rgb, targets_group, mse_weight, scale_weight)

                    sub_loss.backward()

                    num_samples = targets_group.size(0)
                    total_loss_val += sub_loss.item() * num_samples
                    total_samples += num_samples
                    del img, inputs_group, times_group, targets_group, sub_loss

                self.optimizer.step()
                self.optimizer.zero_grad()

                average_loss = total_loss_val / total_samples
                train_loss += average_loss
                self.writer.add_scalar('step/loss', average_loss, stepnum + 1)
                stepnum += 1

            train_loss /= len(self.train_loader)
            val_loss = self.evaluate(images_list)
            self.writer.add_scalar('train/loss', train_loss, epoch + 1)
            self.writer.add_scalar('val/loss', val_loss, epoch + 1)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

    def evaluate(self, images_list):
        """验证"""
        self.model.eval()
        self.model.avg_flag = False
        
        total_samples = 0
        total_loss = 0.0
        mse_weight = 1
        scale_weight = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs_batch, times_batch, targets_batch, brdfid_batch = batch

                grouped_inputs = defaultdict(list)
                grouped_times = defaultdict(list)
                grouped_targets = defaultdict(list)

                for i, brdfid in enumerate(brdfid_batch):
                    grouped_inputs[brdfid.item()].append(inputs_batch[i])
                    grouped_times[brdfid.item()].append(times_batch[i])
                    grouped_targets[brdfid.item()].append(targets_batch[i])

                for brdfid, inputs_group in grouped_inputs.items():
                    img = images_list[brdfid]
                    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                    img = img.unsqueeze(0).to(self.device)

                    inputs_group = torch.stack(inputs_group).unsqueeze(0).float().to(self.device)
                    times_group = torch.stack(grouped_times[brdfid]).unsqueeze(0).float().to(self.device)
                    targets_group = torch.stack(grouped_targets[brdfid]).unsqueeze(0).float().to(self.device)

                    outputs = self.model(img, inputs_group, times_group)
                    rgb = outputs['rgb'].float()
                    rgb = rgb.squeeze(-1)

                    loss = self.criterion(inputs_group, rgb, targets_group, mse_weight, scale_weight)
                    num_samples = targets_group.size(0)
                    total_loss += loss.item() * num_samples
                    total_samples += num_samples

        val_loss = total_loss / total_samples
        return val_loss

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_{val_loss:.6f}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved to {checkpoint_path}")


class BRDFDataset(torch.utils.data.Dataset):
    """BRDF 数据集"""
    def __init__(self, data_dir, file_list, samples_per_brdf=512000):
        self.data = []
        self.samples_per_brdf = samples_per_brdf
        
        with open(file_list, 'r') as f:
            spec_names = [line.strip() for line in f.readlines()]
        
        count = 1
        for spec_name in spec_names:
            spec_path = os.path.join(data_dir, spec_name + "_spec")
            spec_file = os.path.join(spec_path, spec_name + "_spec.npy")

            if os.path.exists(spec_path):
                npdata = np.load(spec_file)
                
                for i in range(min(samples_per_brdf, len(npdata))):
                    points_i = np.array([npdata[i][0], npdata[i][1], npdata[i][2]])
                    rgb = npdata[i][4]
                    time = np.array([npdata[i][3] / 5])
                    item = {"points": points_i, "time": time, "rgb": rgb}
                    self.data.append(item)

                print(f"Loaded {len(self.data)} samples from {spec_name}")
                count += 1
            else:
                print(f"Folder does not exist: {spec_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]["points"]
        targets = self.data[idx]["rgb"]
        times = self.data[idx]["time"]
        brdfid = int(idx / self.samples_per_brdf)
        return inputs, times, targets, brdfid


def create_model(config_path):
    """创建模型"""
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
    return model


def train_main(
    config_path,
    train_file,
    val_file,
    image_dir,
    data_dir,
    checkpoint_dir,
    batch_size=2048,
    num_epochs=15,
    lr_nn=0.0001,
    lr_other=0.0001,
    lr_spec=0.0001
):
    """训练主函数"""
    random.seed(42)
    
    # 创建数据集
    train_dataset = BRDFDataset(data_dir, train_file)
    val_dataset = BRDFDataset(data_dir, val_file)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 加载图片
    images_list = []
    with open(train_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    
    for img_name in lines:
        img_path = os.path.join(image_dir, img_name + ".png")
        image = Image.open(img_path)
        image_array = np.array(image)
        img = normalize_to_neg1_pos1(image_array)
        images_list.append(img)
    
    print(f"Loaded {len(images_list)} images")

    # 创建模型
    model = create_model(config_path)
    
    # 损失函数和优化器
    criterion = SpectralLoss()
    params = model.get_params1()
    optimizer = optim.Adam([
        {'params': params['nn'], 'lr': lr_nn},
        {'params': params['other'], 'lr': lr_other},
        {'params': params['spec'], 'lr': lr_spec},
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建训练器并训练
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, checkpoint_dir)
    trainer.train(num_epochs, images_list)


if __name__ == "__main__":
    # 示例用法
    train_main(
        config_path="config.py",
        train_file="train.txt",
        val_file="val.txt",
        image_dir="./images",
        data_dir="./data",
        checkpoint_dir="./checkpoints"
    )
