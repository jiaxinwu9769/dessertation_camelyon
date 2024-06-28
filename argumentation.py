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
import matplotlib.pyplot as plt
import pywt

# 定义数据增强变换
class CustomTransform:
     """
    Custom data augmentation transform that applies random horizontal flip and random rotation.
    
    Methods
    -------
    __call__(img)
        Applies the transformation to the input image.
    """
     def __call__(self, img):
        """
        Apply random horizontal flip and random rotation to the given image.

        Parameters
        ----------
        img : PIL.Image
            Input image to be transformed.

        Returns
        -------
        PIL.Image
            Transformed image.
        """
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
    transforms.ToTensor(), #将图像转换为 PyTorch 张量并将像素值缩放到范围 [0, 1]。
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 图像是RGB,使用每个信道给定的平均值和标准差对张量进行归一化。
])


# 定义组织分割函数
def tissue_segmentation(image):
    """
    Segments tissue regions in an image by thresholding grayscale pixel values.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image as a numpy array of shape (height, width, 3).
    
    Returns
    -------
    numpy.ndarray
        Binary mask where tissue regions are marked as True and non-tissue regions as False.
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
        
        # 应用小波变换到每个通道
        dwt_image = apply_dwt_per_channel(image)
        
        original_image = Image.fromarray(image, 'RGB')
        dwt_image = (dwt_image - dwt_image.min()) / (dwt_image.max() - dwt_image.min())  # 归一化
        dwt_image_pil = Image.fromarray((dwt_image * 255).astype('uint8'), 'RGB')
        
        label = self.labels[idx]
        if self.transform:
            transformed_image = self.transform(dwt_image_pil)
        else:
            transformed_image = transforms.ToTensor()(dwt_image_pil)
        return transforms.ToTensor()(original_image), transformed_image, label

# 创建训练数据集和数据加载器
train_dataset = CamelyonDataset('camedata/camelyonpatch_level_2_split_train_x.h5',
                                'camedata/camelyonpatch_level_2_split_train_y.h5',
                                'camedata/camelyonpatch_level_2_split_train_meta.csv',
                                transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 展示函数
def show_images(original_images, transformed_images):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for i in range(3):
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

# 获取一个批次的数据并展示三个示例
for original_images, transformed_images, labels in train_loader:
    show_images(original_images[:3], transformed_images[:3])
    break
