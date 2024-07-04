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
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import pywt


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


# 3.定义自定义数据集类
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

# 5.1 定义 ResNet 模型
class ResNetClassifier(nn.Module):
    """
    ResNetClassifier is a convolutional neural network model based on the ResNet-18 architecture,
    pre-trained on ImageNet. This model is adapted for binary classification tasks, specifically
    designed to classify whether an input image contains tumor tissue or not.

    Attributes:
        backbone (torchvision.models.resnet.ResNet): The ResNet-18 model with a modified fully connected layer.

    Methods:
        forward(x): Defines the forward pass of the model.
    """
    
    def __init__(self):
        """
        Initializes the ResNetClassifier model by loading a pre-trained ResNet-18 model,
        replacing its fully connected layer with a new layer for binary classification, and
        printing the number of parameters in the model.
        """
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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


# 5.2 定义 VGG 模型
class ModifiedVGG16(nn.Module):
    def __init__(self):
        super(ModifiedVGG16, self).__init__()
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 修改第一层的卷积层以适应 96x96 的输入
        self.backbone.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        num_ftrs = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid())

        n_params = sum([p.numel() for p in self.parameters()])
        print("\n")
        print("# " * 50)
        print(f"ModifiedVGG16 initialized with {n_params:.3e} parameters")
        print("# " * 50)
        print("\n")

    def forward(self, x):
        return self.backbone(x)


# 5.3 定义 MobileNetV3 模型
class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3Classifier is a convolutional neural network model based on the MobileNetV3-Large architecture,
    pre-trained on ImageNet. This model is adapted for binary classification tasks, specifically
    designed to classify whether an input image contains tumor tissue or not.

    Attributes:
        backbone (torchvision.models.mobilenet.MobileNetV3): The MobileNetV3-Large model with a modified fully connected layer.

    Methods:
        forward(x): Defines the forward pass of the model.
    """
    
    def __init__(self):
        """
        Initializes the MobileNetV3Classifier model by loading a pre-trained MobileNetV3-Large model,
        replacing its fully connected layer with a new layer for binary classification, and
        printing the number of parameters in the model.
        """
        super().__init__()
        bbackbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

        n_params = sum([p.numel() for p in self.parameters()])
        print("\n")
        print("# " * 50)
        print(f"MobileNetV3-Large initialized with {n_params:.3e} parameters")
        print("# " * 50)
        print("\n")

    def forward(self, x):
        return self.backbone(x)


# 5.4 定义 ShuffleNetV2 模型
class ShuffleNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

        n_params = sum([p.numel() for p in self.parameters()])
        print("\n")
        print("# " * 50)
        print(f"ShuffleNetV2 initialized with {n_params:.3e} parameters")
        print("# " * 50)
        print("\n")

    def forward(self, x):
        return self.backbone(x)

# 6.训练函数
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

# 7. 测试函数
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
    return test_loss, test_acc



# 8.预测肿瘤区域并生成热力图
def predict_tumor_regions(model, h5_file, patch_size):
    with h5py.File(h5_file, 'r') as file:
        images = file['x'][:]
    
    heatmap = np.zeros((images.shape[1], images.shape[2]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for idx in range(images.shape[0]):
            image = images[idx].astype('uint8')
            tissue_mask = tissue_segmentation(image)
            image[tissue_mask == False] = [255, 255, 255]
            image = Image.fromarray(image, 'RGB')
            image = transforms.ToTensor()(image).unsqueeze(0).to(device)
            
            output = model(image).cpu().numpy()
            heatmap += output[0, 0]  # 假设单通道输出

    return heatmap

def visualize_heatmap(image, heatmap, output_path, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path)
    
    


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
    criterion = nn.BCELoss()
    
    models_dict = {
        'ResNet18': ResNetClassifier(),
        'ModifiedVGG16': ModifiedVGG16(),
        'MobileNetV3': MobileNetV3Classifier(),
        'ShuffleNetV2': ShuffleNetV2Classifier()
    }
    
    results = {}
    heatmaps = {}

    for model_name, model in models_dict.items():
        print(f'\nTraining {model_name} model...\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        start_time = time.time()
        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
        end_time = time.time()
        
        # 测试模型
        test_loss, test_acc = test_model(model, test_loader, criterion)
        
        # 记录结果
        results[model_name] = {
            'Test Loss': test_loss,
            'Test Accuracy': test_acc.item(),
            'Runtime': end_time - start_time,
            'Total Parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

        h5_file = 'camedata/camelyonpatch_level_2_split_test_x_subset.h5'  # 修改为实际 H5 文件路径
        patch_size = 96
        
        heatmaps[model_name] = predict_tumor_regions(model, h5_file, patch_size)



        
    
    # 打印结果
    for model_name, metrics in results.items():
        print(f'\n{model_name} Results:')
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value}')

    # 可视化结果
    model_names = list(results.keys())
    accuracies = [results[model]['Test Accuracy'] for model in model_names]
    runtimes = [results[model]['Runtime'] for model in model_names]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.bar(model_names, accuracies, color=color, alpha=0.6, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Runtime (s)', color=color)
    ax2.plot(model_names, runtimes, color=color, marker='o', label='Runtime')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Model Performance Comparison')
    plt.savefig('model_performance_comparison.png')  # 保存模型结果图为文件
    

    # 生成并展示热力图（仅展示部分）
    for model_name in model_names:
        with h5py.File(h5_file, 'r') as file:
            image = file['x'][0].astype('uint8')  # 获取第一张图像
        visualize_heatmap(image, heatmaps[model_name], f'heatmap_{model_name}.png', f'{model_name} Heatmap')