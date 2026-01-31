import numpy as np
import cv2

# 读取 RGB mask 图像（确保读取的是彩色图）
mask_img = cv2.imread('/openbayes/home/code/K-Planes/plenoxels/renderdata/dragon_mask.png', cv2.IMREAD_COLOR)  # 读取后 shape 为 (256,256,3)

# 检查图像是否正确读取
if mask_img is None:
    raise ValueError("读取图像失败，请检查文件路径")

# 将图像从 BGR 转换为 RGB（如果需要）
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

# 生成二值 mask：如果最后一维任一通道不为 0，则 mask 对应位置为 1，否则为 0
mask_array = np.any(mask_img != 0, axis=2).astype(np.uint8)

# 保存 mask 数组为 npy 文件
np.save('dragon_mask.npy', mask_array)

print("mask.npy 文件已保存，形状为:", mask_array.shape)
