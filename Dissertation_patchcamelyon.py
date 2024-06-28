import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, models
from PIL import Image
import random
from skimage.color import rgb2gray
from skimage import img_as_float
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch.optim as optim
import time
import copy
import torch.nn as nn  # 添加这一行来导入 torch.nn

# 定义数据增强函数
def apply_transforms(img):
    # 随机水平翻转
    if random.random() > 0.5:
        img = F.hflip(img)
    
    # 随机旋转
    angles = [90, 180, 270]
    angle = random.choice(angles)
    img = F.rotate(img, angle)
    
    # Color jitter
    img = transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04)(img)
    
    # Convert to tensor
    img_tensor = F.to_tensor(img)
    
    # Normalize
    img_tensor = F.normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    return img_tensor

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

# 定义小波变换函数
def apply_dwt_per_channel(image):
    channels = []
    for i in range(image.shape[2]):
        coeffs = pywt.dwt2(image[:, :, i], 'haar')
        LL, (LH, HL, HH) = coeffs
        channels.append(LL)
    return np.stack(channels, axis=-1)

# 定义自定义数据集类
class CamelyonDataset(Dataset):
    def __init__(self, data_file, labels_file, meta_file, transform=None):
        self.data = self.load_h5_data(data_file)
        self.labels = self.load_h5_labels(labels_file)
        self.meta = self.load_csv(meta_file)
        self.transform = transform

    def load_h5_data(self, file_path):
        with h5py.File(file_path, 'r') as file:
            return torch.tensor(file['x'][:].astype(np.float32))

    def load_h5_labels(self, file_path):
        with h5py.File(file_path, 'r') as file:
            return torch.tensor(file['y'][:].astype(np.float32))

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
        
        # 应用小波变换到每个通道
        dwt_image = apply_dwt_per_channel(image)
        
        original_image = Image.fromarray(image, 'RGB')
        dwt_image = (dwt_image - dwt_image.min()) / (dwt_image.max() - dwt_image.min())  # 归一化
        dwt_image_pil = Image.fromarray((dwt_image * 255).astype('uint8'), 'RGB')
        
        label = self.labels[idx]
        if self.transform:
            transformed_image = self.transform(dwt_image_pil)
        else:
            transformed_image = apply_transforms(dwt_image_pil)
        return transforms.ToTensor()(original_image), transformed_image, label

# 数据准备
train_dataset = CamelyonDataset('../camedata/camelyonpatch_level_2_split_train_x.h5',
                                '../camedata/camelyonpatch_level_2_split_train_y.h5',
                                '../camedata/camelyonpatch_level_2_split_train_meta.csv',
                                transform=apply_transforms)

val_dataset = CamelyonDataset('../camedata/camelyonpatch_level_2_split_valid_x.h5',
                              '../camedata/camelyonpatch_level_2_split_valid_y.h5',
                              '../camedata/camelyonpatch_level_2_split_valid_meta.csv',
                              transform=apply_transforms)

test_dataset = CamelyonDataset('../camedata/camelyonpatch_level_2_split_test_x.h5',
                               '../camedata/camelyonpatch_level_2_split_test_y.h5',
                               '../camedata/camelyonpatch_level_2_split_test_meta.csv',
                               transform=apply_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定义Inception v3模型
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.aux_logits = False  # 禁用辅助分类器
model.fc = nn.Linear(2048, 2)  # 修改最后一层以适应我们的二分类任务

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 展示函数
def show_images(original_images, transformed_images):
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    for i in range(5):
        original = original_images[i]
        transformed = transformed_images[i]

        original_np = original.permute(1, 2, 0).numpy()
        original_np = original_np * 255
        axes[i, 0].imshow(original_np.astype(np.uint8))
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        transformed_np = transformed.permute(1, 2, 0).numpy()
        transformed_np = transformed_np * 0.5 + 0.5  # 反归一化
        transformed_np = transformed_np * 255
        axes[i, 1].imshow(transformed_np.astype(np.uint8))
        axes[i, 1].set_title("Transformed Image")
        axes[i, 1].axis('off')

    plt.show()

# 获取一个批次的数据并展示五个示例
for original_images, transformed_images, labels in train_loader:
    show_images(original_images[:5], transformed_images[:5])
    break

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置模型为验证模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for original_images, transformed_images, labels in dataloader:
                original_images, transformed_images, labels = original_images.to(device), transformed_images.to(device), labels.to(device)
                
                # 前向传播
                # 只在训练阶段计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(transformed_images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    # 仅在训练阶段进行反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * transformed_images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 训练模型
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# 测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for original_images, transformed_images, labels in test_loader:
            original_images, transformed_images, labels = original_images.to(device), transformed_images.to(device), labels.to(device)
            outputs = model(transformed_images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.long())

            running_loss += loss.item() * transformed_images.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

# 测试模型
test_model(model, test_loader, criterion)

# 保存模型
torch.save(model.state_dict(), 'best_model.pth')

# 打印模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total Parameters: {count_parameters(model)}')
