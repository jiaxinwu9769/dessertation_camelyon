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
import torch.optim as optim
import time
import copy
import torch.nn as nn

# 1.定义数据增强函数
def apply_transforms(image):
    """Applies a series of data augmentation transformations to an image.

    This function performs the following transformations:
    - Random Horizontal Flip: With a probability of 0.5, the image is flipped horizontally.
    - Random Rotation: The image is randomly rotated by 90, 180, or 270 degrees.
    - Color Jitter: Randomly changes the brightness, contrast, saturation, and hue of the image.
    - Conversion to Tensor: The image is converted to a PyTorch tensor.
    - Normalization: The image tensor is normalized with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5].

    Args:
        image (PIL.Image): Input image to be transformed.

    Returns:
        torch.Tensor: Transformed image as a tensor, normalized to a specified mean and standard deviation.

    Example:
        >>> from PIL import Image
        >>> img = Image.open('path/to/image.jpg')
        >>> transformed_img = apply_transforms(img)
    """
    # 随机水平翻转
    if random.random() > 0.5:
        image = F.hflip(image)
    
    # 随机旋转
    angles = [90, 180, 270]
    angle = random.choice(angles)
    image = F.rotate(image, angle)
    
    # Color jitter
    image = transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04)(image)
    
    # Convert to tensor
    image_tensor = F.to_tensor(image)
    
    # Normalize
    image_tensor = F.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    return image_tensor

# 2.定义组织分割函数
def tissue_segmentation(image):
    """Segments tissue regions in an image by thresholding grayscale pixel values.

    This function performs the following steps:
    - Converts pure black pixels to pure white pixels.
    - Converts the RGB image to a grayscale image.
    - Normalizes the grayscale image to the range [0, 1].
    - Creates a binary mask where pixels with values less than or equal to 0.8 are considered as tissue.

    Args:
        image (numpy.ndarray): Input RGB image as a numpy array of shape (height, width, 3).

    Returns:
        numpy.ndarray: Binary mask where tissue regions are marked as True and non-tissue regions as False.

    Example:
        >>> import numpy as np
        >>> from skimage import io
        >>> image = io.imread('path/to/image.jpg')
        >>> tissue_mask = tissue_segmentation(image)
    """
    # 将纯黑色像素转换为纯白色像素
    image[(image == 0).all(axis=-1)] = [255, 255, 255]
    
    # 转换为灰度图像
    gray_image = rgb2gray(image)
    
    # 将灰度图像归一化到 [0, 1]
    gray_image = img_as_float(gray_image)
    
    # 将小于或等于 0.8 的像素视为组织
    tissue_mask = gray_image <= 0.8
    
    return tissue_mask

# 4.定义自定义数据集类
class CamelyonDataset(Dataset):
    def __init__(self, data_file, labels_file, meta_file, transform=None):
        self.data_file = data_file
        self.labels = self.load_h5_labels(labels_file)
        self.meta = self.load_csv(meta_file)
        self.transform = transform
        with h5py.File(data_file, 'r') as file:
            self.data_length = file['x'].shape[0]

    def load_h5_labels(self, file_path):
        with h5py.File(file_path, 'r') as file:
            labels = file['y'][:].astype(np.float32)
            return torch.tensor(labels).view(-1, 1)  # 调整标签形状为 (batch_size, 1)

    def load_csv(self, file_path):
        return pd.read_csv(file_path)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as file:
            image = file['x'][idx].astype('uint8')
        
        # 进行组织分割
        tissue_mask = tissue_segmentation(image)
        
        # 应用组织分割掩码
        image[tissue_mask == False] = [255, 255, 255]

        # 进行数据增强
        image = Image.fromarray(image, 'RGB')
        transformed_image = apply_transforms(image)
        
        label = self.labels[idx]
        
        return transforms.ToTensor()(image), transformed_image, label

# 定义模型
class CamelyonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

        n_params = sum([p.numel() for p in self.parameters()])
        print("\n")
        print("# " * 50)
        print(f"ResNet18 initialized with {n_params:.3e} parameters")
        print("# " * 50)
        print("\n")

    def forward(self, x):
        return self.backbone(x)

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

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
            for inputs, _, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                # 只在训练阶段计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs > 0.5  # 二分类任务的预测值

                    # 仅在训练阶段进行反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
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

# 定义测试函数
def test_model(model, test_loader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, _, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs > 0.5

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

# 主函数
if __name__ == '__main__':
    # 数据准备
    train_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_train_x_subset.h5',
                                    'camedata/camelyonpatch_level_2_split_train_y_subset.h5',
                                    'camedata/camelyonpatch_level_2_split_train_meta.csv',
                                    transform=apply_transforms)

    val_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_valid_x_subset.h5',
                                  'camedata/camelyonpatch_level_2_split_valid_y_subset.h5',
                                  'camedata/camelyonpatch_level_2_split_valid_meta.csv',
                                  transform=apply_transforms)

    test_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_test_x_subset.h5',
                                   'camedata/camelyonpatch_level_2_split_test_y_subset.h5',
                                   'camedata/camelyonpatch_level_2_split_test_meta.csv',
                                   transform=apply_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 定义损失函数和优化器
    model = CamelyonClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # 测试模型
    test_model(model, test_loader, criterion)

    # 保存模型
    torch.save(model.state_dict(), 'best_model.pth')

    # 打印模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total Parameters: {count_parameters(model)}')
