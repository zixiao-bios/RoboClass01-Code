# 直接用python运行，命令：python generate_random_a.py
import numpy as np
import scipy.ndimage

# 生成随机加速度序列
n = 200  # 序列长度
random_accelerations = np.random.randn(n) * 2  # 生成随机加速度

# 应用高斯滤波进行平滑，sigma 控制平滑程度
smoothed_accelerations = scipy.ndimage.gaussian_filter(random_accelerations, sigma=0.8)

print(smoothed_accelerations)
np.save('accelerations.npy', np.array(smoothed_accelerations))
