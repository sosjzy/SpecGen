import numpy as np
import cv2
import matplotlib.pyplot as plt

# 加载npy文件
imgs= np.load('./imgs.npy')
print("imgs.shape:"+str(imgs.shape))

# 加载npy文件
mask= np.load('./mask.npy')
print("mask.shape:"+str(mask.shape))

# 加载npy文件
normal= np.load('./normal.npy')
print("normal.shape:"+str(normal.shape))



# normal= np.uint8(normal)
# # 保存为RGB彩色图像
# cv2.imwrite('output_image.jpg', cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

# # 从(32, 100, 100, 195)数组中提取一个(1,100,100,1)的子数组  
# img = imgs[0:1, :, :, 0:1]


# # 将子数组转换为灰度图像
# img = np.squeeze(img)  # 移除多余的维度

# img = np.uint8(img * 255)  # 将数据缩放到0-255范围内，并转换为uint8类型
# cv2.imwrite('img.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
for i in range(0,31):
    img = imgs[i]*255

    img = img.astype(np.uint8)  # 转换为无符号 8 位整数类型

    img_name = "img_"+str(i)+".png"
    # 使用 matplotlib 保存图像
    plt.imsave(img_name, img)

    print("图像已保存")
