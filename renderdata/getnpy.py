import numpy as np
from PIL import Image

def process_normal_map(input_path, normal_output_path, mask_output_path):
    # 读取 PNG 法线贴图，转换为 RGB 模式以确保有 3 个通道
    img = Image.open(input_path).convert('RGB')
    
    # 将图像转换为 NumPy 数组，数据类型为 float32
    # 注意这里保留原始 uint8 数据以便生成 mask
    img_np_uint8 = np.array(img, dtype=np.uint8)
    
    # 将像素值从 [0, 255] 映射到 [-1, 1]
    normal_map = (img_np_uint8.astype(np.float32) / 127.5) - 1.0
    
    # 保存法线数组到 npy 文件
    np.save(normal_output_path, normal_map)
    
    # 生成 mask：只有当原始 PNG 中对应像素（RGB 三通道）至少有一个通道的值不为0时，该位置为1，否则为0
    mask = (np.any(img_np_uint8 != 0, axis=-1)).astype(np.uint8)
    
    # 保存 mask 到 npy 文件
    np.save(mask_output_path, mask)

# 使用示例
if __name__ == "__main__":
    input_path = "/openbayes/home/code/K-Planes/plenoxels/renderdata/1.png"         # 输入的 PNG 法线贴图文件路径
    normal_output_path = "smile.npy"   # 保存法线数组的输出路径
    mask_output_path = "smile_mask.npy"    # 保存 mask 的输出路径
    
    process_normal_map(input_path, normal_output_path, mask_output_path)
