import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
from skimage.color import rgb2gray
from skimage import img_as_float
import numpy as np

# 定义数据增强变换
class CustomTransform:
    def __call__(self, img):
        # 水平翻转
        if random.random() > 0.5:
            img = F.hflip(img)
        
        # 随机旋转
        angles = [90, 180, 270]
        angle = random.choice(angles)
        img = F.rotate(img, angle)
        
        return img

data_transform = transforms.Compose([
    CustomTransform(),
    transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 图像是RGB
])

# 定义组织分割函数
def tissue_segmentation(image):
    # 将纯黑色像素转换为纯白色像素
    image[(image == 0).all(axis=-1)] = [255, 255, 255]
    
    # 转换为灰度图像
    gray_image = rgb2gray(image)
    
    # 将灰度图像归一化到 [0, 1]
    gray_image = img_as_float(gray_image)
    
    # 将小于或等于 0.8 的像素视为组织
    tissue_mask = gray_image <= 0.8
    
    return tissue_mask

# 定义自定义数据集类
class CamelyonDataset(Dataset):
    def __init__(self, data_file, labels_file, meta_file, transform=None):
        self.data = self.load_h5_data(data_file)
        self.labels = self.load_h5_labels(labels_file)
        self.meta = self.load_csv(meta_file)
        self.transform = transform

    def load_h5_data(self, file_path):
        with h5py.File(file_path, 'r') as file:
            return torch.tensor(file['x'][:])

    def load_h5_labels(self, file_path):
        with h5py.File(file_path, 'r') as file:
            return torch.tensor(file['y'][:])

    def load_csv(self, file_path):
        return pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].numpy().astype('uint8')
        
        # 进行组织分割
        tissue_mask = tissue_segmentation(image)
        
        # 应用组织分割掩码
        image[tissue_mask == False] = [255, 255, 255]
        
        image = Image.fromarray(image, 'RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 创建数据集和数据加载器
train_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_train_x.h5',
                                'camedata/camelyonpatch_level_2_split_train_y.h5',
                                'camedata/camelyonpatch_level_2_split_train_meta.csv',
                                transform=data_transform)

valid_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_valid_x.h5',
                                'camedata/camelyonpatch_level_2_split_valid_y.h5',
                                'camedata/camelyonpatch_level_2_split_valid_meta.csv',
                                transform=data_transform)

test_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_test_x.h5',
                               'camedata/camelyonpatch_level_2_split_test_y.h5',
                               'camedata/camelyonpatch_level_2_split_test_meta.csv',
                               transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 测试数据加载器
for images, labels in train_loader:
    print(images.shape, labels.shape)
    break
