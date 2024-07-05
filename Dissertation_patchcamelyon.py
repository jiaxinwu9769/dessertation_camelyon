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
from sklearn.metrics import roc_auc_score, roc_curve

# 1.定义数据增强函数
def apply_transforms(image):
    if random.random() > 0.5:
        image = F.hflip(image)
    
    angles = [90, 180, 270]
    angle = random.choice(angles)
    image = F.rotate(image, angle)
    
    image = transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04)(image)
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    return image_tensor

# 2.定义组织分割函数
def tissue_segmentation(image):
    image[(image == 0).all(axis=-1)] = [255, 255, 255]
    gray_image = rgb2gray(image)
    gray_image = img_as_float(gray_image)
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
            return torch.tensor(labels).view(-1, 1) 

    def load_csv(self, file_path):
        return pd.read_csv(file_path)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        with h5py.File(self.data_file, 'r') as file:
            image = file['x'][idx].astype('uint8')
        tissue_mask = tissue_segmentation(image)
        image[tissue_mask == False] = [255, 255, 255]
        image = Image.fromarray(image, 'RGB')
        transformed_image = apply_transforms(image)
        label = self.labels[idx]
        return transforms.ToTensor()(image), transformed_image, label

# 5.1 定义 ResNet 模型
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
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
        self.backbone = models.vgg16(weights='IMAGENET1K_V1')
        self.backbone.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        num_ftrs = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

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
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
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
        backbone = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
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
    best_auc = 0.0
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                dataloader = train_loader
            else:
                model.eval()   
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_outputs = []

            for inputs, _, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs > 0.5

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_auc = roc_auc_score(all_labels, all_outputs)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} AUC: {best_auc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc, best_auc

# 7. 测试函数
def test_model(model, test_loader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, _, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs > 0.5

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    test_auc = roc_auc_score(all_labels, all_outputs)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}')
    return test_loss, test_acc, test_auc

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
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
            
            output = model(image_tensor).cpu().numpy()
            heatmap += output[0, 0]  

    return heatmap / len(images)

def visualize_heatmap(image, heatmap, output_path, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    plt.savefig(output_path)

def save_display_images(original_images, transformed_images, filename):
    fig, axes = plt.subplots(2, len(original_images), figsize=(15, 5))
    for i in range(len(original_images)):
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title('Original Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(transformed_images[i].permute(1, 2, 0))
        axes[1, i].set_title('Transformed Image')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)

# 主函数
if __name__ == '__main__':
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

    criterion = nn.BCELoss()
    
    models_dict = {
        'ResNet18': ResNetClassifier(),
        'ModifiedVGG16': ModifiedVGG16(),
        'MobileNetV3': MobileNetV3Classifier(),
        'ShuffleNetV2': ShuffleNetV2Classifier()
    }
    
    results = {}
    heatmaps = {}

    # 随机选择几张图像并展示
    sample_indices = random.sample(range(len(test_dataset)), 5)
    original_images = []
    transformed_images = []

    for idx in sample_indices:
        original_image, transformed_image, _ = test_dataset[idx]
        original_images.append(original_image.permute(1, 2, 0).numpy())
        transformed_images.append(transformed_image)
    
    save_display_images(original_images, transformed_images, 'original_and_transformed_images.png')

    for model_name, model in models_dict.items():
        print(f'\nTraining {model_name} model...\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        model, best_acc, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1)
        end_time = time.time()
        
        test_loss, test_acc, test_auc = test_model(model, test_loader, criterion)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results[model_name] = {
            'Test Loss': test_loss,
            'Test Accuracy': test_acc.item(),
            'Test AUC': test_auc,
            'Runtime': end_time - start_time,
            'Total Parameters': total_params
        }

        h5_file = 'camedata/camelyonpatch_level_2_split_test_x_subset.h5'
        patch_size = 96
        
        heatmaps[model_name] = predict_tumor_regions(model, h5_file, patch_size)

    for model_name, metrics in results.items():
        print(f'\n{model_name} Results:')
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value}')

    model_names = list(results.keys())
    accuracies = [results[model]['Test Accuracy'] for model in model_names]
    aucs = [results[model]['Test AUC'] for model in model_names]
    runtimes = [results[model]['Runtime'] for model in model_names]
    params = [results[model]['Total Parameters'] for model in model_names]

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
    plt.title('Model Accuracy and Runtime Comparison')
    plt.savefig('model_accuracy_runtime_comparison.png')

    fig, ax3 = plt.subplots()

    color = 'tab:green'
    ax3.set_xlabel('Model')
    ax3.set_ylabel('AUC', color=color)
    ax3.plot(model_names, aucs, color=color, marker='o', label='AUC')
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Model AUC Comparison')
    plt.savefig('model_auc_comparison.png')

    fig, ax4 = plt.subplots()

    color = 'tab:purple'
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Total Parameters', color=color)
    ax4.bar(model_names, params, color=color, alpha=0.6, label='Total Parameters')
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Model Parameters Comparison')
    plt.savefig('model_parameters_comparison.png')

    for model_name in model_names:
        with h5py.File(h5_file, 'r') as file:
            image = file['x'][0].astype('uint8')
        visualize_heatmap(image, heatmaps[model_name], f'heatmap_{model_name}.png', f'{model_name} Heatmap')
