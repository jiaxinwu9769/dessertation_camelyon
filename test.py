import numpy as np
import torch

# 生成随机样本
sample_size = 1000  # 样本大小
mean = 0  # 样本均值
std_dev = 1  # 样本标准差

# 生成服从正态分布的随机样本
samples = np.random.normal(loc=mean, scale=std_dev, size=sample_size)

# 计算样本的平均值和标准差
calculated_mean = np.mean(samples)
calculated_std_dev = np.std(samples)

print(f"Generated samples: {samples[:10]}...")  # 打印前10个样本
print(f"Sample size: {sample_size}")
print(f"Calculated Mean: {calculated_mean}")
print(f"Calculated Standard Deviation: {calculated_std_dev}")