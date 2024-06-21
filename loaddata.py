import h5py
import torch
import pandas as pd

# 定义加载数据和标签的函数,从HDF5文件中加载数据和标签，并将它们转换为PyTorch张量
def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        return torch.tensor(file['x'][:])

def load_h5_labels(file_path):
    with h5py.File(file_path, 'r') as file:
        return torch.tensor(file['y'][:])

# 加载CSV文件的函数
def load_csv(file_path):
    return pd.read_csv(file_path)

# 加载数据 
# train data
train_x = load_h5_data('data/camelyonpatch_level_2_split_train_x.h5')
train_y = load_h5_labels('data/camelyonpatch_level_2_split_train_y.h5')
train_meta = load_csv('data/camelyonpatch_level_2_split_train_meta.csv')

# valid data
valid_x = load_h5_data('data/camelyonpatch_level_2_split_valid_x.h5')
valid_y = load_h5_labels('data/camelyonpatch_level_2_split_valid_y.h5')
valid_meta = load_csv('data/camelyonpatch_level_2_split_valid_meta.csv')

# test data
test_x = load_h5_data('data/camelyonpatch_level_2_split_test_x.h5')
test_y = load_h5_labels('data/camelyonpatch_level_2_split_test_y.h5')
test_meta = load_csv('data/camelyonpatch_level_2_split_test_meta.csv') 

print(f"Train X shape: {train_x.shape}, dtype: {train_x.dtype}")
print(f"Train Y shape: {train_y.shape}, dtype: {train_y.dtype}")
print(f"Valid X shape: {valid_x.shape}, dtype: {valid_x.dtype}")
print(f"Valid Y shape: {valid_y.shape}, dtype: {valid_y.dtype}")
print(f"Test X shape: {test_x.shape}, dtype: {test_x.dtype}")
print(f"Test Y shape: {test_y.shape}, dtype: {test_y.dtype}")
print("Train meta shape:", train_meta.shape)
print("Validation meta shape:", valid_meta.shape)
print("Test meta shape:", test_meta.shape)
print("--------------------------------------------------------------------")

# 查看训练标签的唯一值和计数
unique_labels, counts = torch.unique(train_y, return_counts=True)
label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
print("Train label distribution:", label_distribution)

# 查看元数据的信息
print("Train meta info:")
print(train_meta.head())

