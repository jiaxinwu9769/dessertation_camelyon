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
import cv2

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

def tissue_segmentation(image):
    image[(image == 0).all(axis=-1)] = [255, 255, 255]
    gray_image = rgb2gray(image)
    gray_image = img_as_float(gray_image)
    tissue_mask = gray_image <= 0.8
    return tissue_mask

# 2
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

# 3
class MobileNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

class ShuffleNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

class SqueezeNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.squeezenet1_0(weights='IMAGENET1K_V1')
        backbone.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.backbone = backbone
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)
        return x.view(-1, 1)

class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

#4 
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    since = time.time()

    train_losses = []
    val_losses = []

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

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} AUC: {best_auc:.4f}')

    model.load_state_dict(best_model_wts)
    
    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')

    return model, best_acc, best_auc

# 5
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

# 6
def grad_cam(model, img, target_layer):
    model.eval()
    features = []
    grads = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

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

def predict_tumor_regions(model, h5_file, model_name, patch_size=96, step_size=48):
    with h5py.File(h5_file, 'r') as file:
        images = file['x'][:]
    
    height, width = images.shape[1], images.shape[2]
    heatmap = np.zeros((height, width))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if model_name == 'MobileNetV2':
        target_layer = model.backbone.features[-1]
    elif model_name == 'ShuffleNetV2':
        target_layer = model.backbone.conv5
    elif model_name == 'SqueezeNet':
        target_layer = model.backbone.features[-1]
    elif model_name == 'ResNet18':
        target_layer = model.backbone.layer4[1].conv2

    for idx in range(images.shape[0]):
        image = images[idx].astype('uint8')
        tissue_mask = tissue_segmentation(image)
        image[tissue_mask == False] = [255, 255, 255]
        image = Image.fromarray(image, 'RGB')
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        
        cam = grad_cam(model, image_tensor, target_layer)
        heatmap += cam

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

# 7
def visualize_heatmap(image, heatmap, output_path, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

# 8
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
        'MobileNetV2': MobileNetV2Classifier(),
        'ShuffleNetV2': ShuffleNetV2Classifier(),
        'SqueezeNet': SqueezeNetClassifier(),
        'ResNet18': ResNetClassifier()
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
        model, best_acc, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2)
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

        heatmaps[model_name] = predict_tumor_regions(model, h5_file, model_name)

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
