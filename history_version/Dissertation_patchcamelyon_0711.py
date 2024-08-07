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
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import cv2
from torch.optim.lr_scheduler import StepLR

# 数据增强
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

# 组织分割
def tissue_segmentation(image):
    image[(image == 0).all(axis=-1)] = [255, 255, 255]
    gray_image = rgb2gray(image)
    gray_image = img_as_float(gray_image)
    tissue_mask = gray_image <= 0.8
    return tissue_mask

# 自定义数据集
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

# 定义 MobileNetV3 模型
class MobileNetV3Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        num_ftrs = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(num_ftrs, 1)
        self.backbone = nn.Sequential(backbone, nn.Sigmoid())

    def forward(self, x):
        return self.backbone(x)

# 定义 ShuffleNetV2 模型
class ShuffleNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# 定义 SqueezeNet 模型
class SqueezeNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.squeezenet1_0(weights='IMAGENET1K_V1')
        backbone.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.backbone = nn.Sequential(backbone, nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 1)

# 定义 ResNet 模型
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# 定义 EfficientNet 模型
class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    since = time.time()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('-' * 20)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        

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

            if phase == 'train':
                train_losses.append(epoch_loss)
                scheduler.step()  # 在训练阶段结束时调用调度器的step方法
            else:
                val_losses.append(epoch_loss)
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(best_model_wts)
                    return model, best_acc, best_auc, train_losses, val_losses


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    
    print(f'\n Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} AUC: {best_auc:.4f}')

    model.load_state_dict(best_model_wts)
    
    return model, best_acc, best_auc, train_losses, val_losses

# 测试函数
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
    test_precision = precision_score(all_labels, np.round(all_outputs))
    test_recall = recall_score(all_labels, np.round(all_outputs))

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f}')
    return test_loss, test_acc, test_auc, test_precision, test_recall

# 计算验证集运行时间的函数
def validation_runtime(model, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for inputs, _, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

    end_time = time.time()
    runtime = end_time - start_time
    return runtime

# Grad-CAM函数
def grad_cam(model, img, target_layer):
    model.eval()
    features = []
    grads = []

    def forward_hook(module, input, output):
        features.append(output)

    def full_backward_hook(module, grad_input, grad_output):
        grads.append(grad_output[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(full_backward_hook)

    output = model(img)
    model.zero_grad()
    class_loss = output[0]
    class_loss.backward()

    handle_forward.remove()
    handle_backward.remove()

    gradients = grads[0]
    activations = features[0]

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.shape[2], img.shape[3]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam


# 预测肿瘤区域
def predict_tumor_regions(model, h5_file, model_name, patch_size=96, step_size=48):
    with h5py.File(h5_file, 'r') as file:
        images = file['x'][:]
    
    height, width = images.shape[1], images.shape[2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if model_name == 'MobileNetV3':
        target_layer = model.backbone[0].features[-1]
    elif model_name == 'ShuffleNetV2':
        target_layer = model.backbone.conv5
    elif model_name == 'SqueezeNet':
        target_layer = model.backbone[0].features[-1]
    elif model_name == 'ResNet18':
        target_layer = model.backbone.layer4[1].conv2
    elif model_name == 'EfficientNet':
        target_layer = model.backbone.features[-1]

    heatmaps = []
    for idx in range(images.shape[0]):
        image = images[idx].astype('uint8')
        tissue_mask = tissue_segmentation(image)
        image[tissue_mask == False] = [255, 255, 255]
        image = Image.fromarray(image, 'RGB')
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        
        cam = grad_cam(model, image_tensor, target_layer)
        heatmaps.append(cam)

    return heatmaps

# 可视化热力图对比
def visualize_heatmap_comparison(images, heatmaps, model_name, output_path):
    fig, axes = plt.subplots(5, 2, figsize=(10, 25))
    for i in range(5):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Original Image {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(images[i])
        heatmap_resized = cv2.resize(heatmaps[i], (images[i].shape[1], images[i].shape[0]))
        axes[i, 1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[i, 1].set_title(f'Heatmap {i}')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# 保存原始和增强后的图像
def save_display_images(original_images, transformed_images, filename):
    fig, axes = plt.subplots(2, len(original_images), figsize=(15, 5))
    for i in range(len(original_images)):
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title('Original Image', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(transformed_images[i].permute(1, 2, 0))
        axes[1, i].set_title('Transformed Image', fontsize=10)
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 主程序
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
        'MobileNetV3': MobileNetV3Classifier(),
        'ShuffleNetV2': ShuffleNetV2Classifier(),
        'SqueezeNet': SqueezeNetClassifier(),
        'ResNet18': ResNetClassifier(),
        'EfficientNet': EfficientNetClassifier()
    }
    
    results = {}
    heatmaps = {}
    train_losses_dict = {}
    val_losses_dict = {}
    val_runtimes = {}

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
        print(f'\n[Training {model_name} model...]\n')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adding L2 regularization
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by a factor of 0.1 every 10 epochs
        criterion = nn.BCELoss()
        early_stopping = EarlyStopping(patience=10, delta=0.01)

        # 训练模型并记录验证集运行时间
        model, best_acc, best_auc, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2)
        val_runtime = validation_runtime(model, val_loader)
        
        test_loss, test_acc, test_auc, test_precision, test_recall = test_model(model, test_loader, criterion)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inference_time = validation_runtime(model, test_loader) / len(test_loader.dataset)
        
        results[model_name] = {
            'Test Loss': test_loss,
            'Test Accuracy': test_acc.item(),
            'Test AUC': test_auc,
            'Precision': test_precision,
            'Recall': test_recall,
            'Total Parameters': total_params,
            'Inference Time per Image': inference_time
        }
        val_runtimes[model_name] = val_runtime

        train_losses_dict[model_name] = train_losses
        val_losses_dict[model_name] = val_losses

        h5_file = 'camedata/camelyonpatch_level_2_split_test_x_subset.h5'

        heatmaps[model_name] = predict_tumor_regions(model, h5_file, model_name)

    for model_name, metrics in results.items():
        print(f'\n{model_name} Results:')
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value}')

    model_names = list(results.keys())
    accuracies = [results[model]['Test Accuracy'] for model in model_names]
    aucs = [results[model]['Test AUC'] for model in model_names]
    precisions = [results[model]['Precision'] for model in model_names]
    recalls = [results[model]['Recall'] for model in model_names]
    params = [results[model]['Total Parameters'] for model in model_names]
    inf_times = [results[model]['Inference Time per Image'] for model in model_names]
    val_runtime_values = [val_runtimes[model] for model in model_names]

    # 图表1: Model Accuracy and Inference Time Comparison
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.bar(model_names, accuracies, color=color, alpha=0.6, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Inference Time (s)', color=color)
    ax2.plot(model_names, inf_times, color=color, marker='o', label='Inference Time')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('Model Accuracy and Inference Time Comparison')
    plt.savefig('model_accuracy_inference_time_comparison.png')

    # 图表2: Validation Runtime Comparison
    fig, ax3 = plt.subplots()

    color = 'tab:green'
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Validation Runtime (s)', color=color)
    ax3.bar(model_names, val_runtime_values, color=color, alpha=0.6, label='Validation Runtime')
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('Validation Runtime Comparison')
    plt.savefig('validation_runtime_comparison.png')

    fig, ax4 = plt.subplots()

    color = 'tab:purple'
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Total Parameters', color=color)
    ax4.bar(model_names, params, color=color, alpha=0.6, label='Total Parameters')
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('Model Parameters Comparison')
    plt.savefig('model_parameters_comparison.png')

    fig, ax5 = plt.subplots()

    color = 'tab:orange'
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Inference Time (s)', color=color)
    ax5.bar(model_names, inf_times, color=color, alpha=0.6, label='Inference Time')
    ax5.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('Model Inference Time Comparison')
    plt.savefig('model_inference_time_comparison.png')

    fig, ax6 = plt.subplots()

    color = 'tab:brown'
    ax6.set_xlabel('Model')
    ax6.set_ylabel('Precision', color=color)
    ax6.bar(model_names, precisions, color=color, alpha=0.6, label='Precision')
    ax6.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('Model Precision Comparison')
    plt.savefig('model_precision_comparison.png')

    fig, ax7 = plt.subplots()

    color = 'tab:cyan'
    ax7.set_xlabel('Model')
    ax7.set_ylabel('Recall', color=color)
    ax7.bar(model_names, recalls, color=color, alpha=0.6, label='Recall')
    ax7.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('Model Recall Comparison')
    plt.savefig('model_recall_comparison.png')

    for model_name in model_names:
        fig, ax = plt.subplots()
        ax.plot(range(len(train_losses_dict[model_name])), train_losses_dict[model_name], label='Training Loss')
        ax.plot(range(len(val_losses_dict[model_name])), val_losses_dict[model_name], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} Training and Validation Loss')
        ax.legend()
        plt.savefig(f'{model_name}_loss_curve.png')

        with h5py.File(h5_file, 'r') as file:
            original_images = [file['x'][i].astype('uint8') for i in sample_indices]
            heatmap_images = heatmaps[model_name]
            visualize_heatmap_comparison(original_images, heatmap_images, model_name, f'heatmap_comparison_{model_name}.png')
