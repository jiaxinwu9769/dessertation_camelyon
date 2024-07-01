import h5py
import numpy as np

def extract_subset(data_file, labels_file, output_data_file, output_labels_file, num_samples):
    # 读取原始数据和标签
    with h5py.File(data_file, 'r') as f:
        data = f['x'][:]
    
    with h5py.File(labels_file, 'r') as f:
        labels = f['y'][:]
    
    # 确保 num_samples 不超过数据集大小
    num_samples = min(num_samples, data.shape[0])
    
    # 随机选择 num_samples 个样本
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    data_subset = data[indices]
    labels_subset = labels[indices]
    
    # 保存提取的样本到新的 HDF5 文件
    with h5py.File(output_data_file, 'w') as f:
        f.create_dataset('x', data=data_subset)
    
    with h5py.File(output_labels_file, 'w') as f:
        f.create_dataset('y', data=labels_subset)

# 示例用法
data_file = 'camedata/camelyonpatch_level_2_split_test_x.h5'
labels_file = 'camedata/camelyonpatch_level_2_split_test_y.h5'
output_data_file = 'camedata/camelyonpatch_level_2_split_test_x_subset.h5'
output_labels_file = 'camedata/camelyonpatch_level_2_split_test_y_subset.h5'
num_samples = 10  # 提取10个样本

extract_subset(data_file, labels_file, output_data_file, output_labels_file, num_samples)
