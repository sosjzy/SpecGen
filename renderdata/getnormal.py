import numpy as np
import cv2

def process_normal_map(normal_png_path, mask_npy_path, output_npy_path):
    # 读取法线贴图，确保以彩色模式读取（返回 BGR 顺序）
    normal_img = cv2.imread(normal_png_path, cv2.IMREAD_COLOR)
    if normal_img is None:
        raise ValueError("无法读取法线贴图，请检查文件路径")
    
    # 将 BGR 转换为 RGB
    normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
    
    # 将图像转换为 float32 并归一化到 [0, 1]
    normal_img = normal_img.astype(np.float32) / 255.0
    
    # 将 [0, 1] 映射到 [-1, 1]，得到表面法线
    normals = normal_img * 2.0 - 1.0
    
    # 读取 mask 文件，假设 mask 为 (256,256) 的 npy 文件，mask 中非零值表示需要的区域
    mask = np.load(mask_npy_path)
    mask_bool = mask.astype(bool)
    
    # 对于 mask 中不需要的区域，将法线赋值为 0
    normals[~mask_bool] = 0.0
    
    # 保存处理后的法线为 npy 文件
    np.save(output_npy_path, normals)
    print(f"处理后的法线已保存到：{output_npy_path}")

# 使用示例（请替换为实际的文件路径）
normal_png_path = "/openbayes/home/code/K-Planes/plenoxels/renderdata/dragon_normal.png"    # 法线贴图 PNG 文件
mask_npy_path = "/openbayes/home/code/K-Planes/plenoxels/renderdata/dragon_mask.npy"            # mask 的 npy 文件
output_npy_path = "dragon_normals.npy"  # 输出文件路径

process_normal_map(normal_png_path, mask_npy_path, output_npy_path)
