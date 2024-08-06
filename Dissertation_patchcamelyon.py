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
import torchprofile
from thop import profile


### 0. Reproducibility seed 
def set_seed(seed):
    """
    Set the random seed for reproducibility
    Args: seed (int): The seed value to be used for random number generation.
    """
    random.seed(seed) # Set the random seed 
    np.random.seed(seed) # Set the random seed for NumPy's random number generator
    torch.manual_seed(seed) # Set the random seed for PyTorch's CPU random number generator
    
    # If CUDA is available, set the seed for PyTorch's GPU random number generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Set seed for the current GPU
        torch.cuda.manual_seed_all(seed)

    #Set CuDNN to ensure consistent results every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #Disabling auto-tuning

# Setting the random seed
set_seed(2024)




### 1. Image augmentation

def apply_transforms(self, image):
    """
        Apply a series of random transformations to the image.
        Args: image (PIL.Image): The input image to be transformed.
        Returns: Tensor: The transformed image as a tensor, normalized with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5].
    """
    if random.random() > 0.5:
        image = F.hflip(image)
        
    angles = [90, 180, 270]
    angle = random.choice(angles)
    image = F.rotate(image, angle)

    image = transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.3, hue=0.1)(image)
        
    image_tensor = F.to_tensor(image)
    image_tensor = F.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    return image_tensor




### 2. custom Dataset

class CamelyonDataset(Dataset):
    def __init__(self, data_file, labels_file, meta_file, transform=None, test=False):
        """
        Initialize the dataset.

        Args:
            data_file (str): Path to the data file (HDF5 format)
            labels_file (str): Path to the labels file (HDF5 format)
            meta_file (str): Path to the metadata file (CSV format).
            transform (callable, optional): A function/transform to apply to the image. Defaults to None.
            test (bool, optional): If True, apply only basic transformations and normalization. Defaults to False.
        """
        self.data_file = data_file
        self.labels = self.load_h5_labels(labels_file)
        self.meta = self.load_csv(meta_file)
        self.transform = transform
        self.test = test
        with h5py.File(data_file, 'r') as file:
            self.data_length = file['x'].shape[0]

    def load_h5_labels(self, file_path):
        """
        Load labels from an HDF5 file (y).
        Args:  file_path : Path to the HDF5 file
        Returns: Tensor: Tensor containing labels.
        """      
        with h5py.File(file_path, 'r') as file:
            labels = file['y'][:].astype(np.float32)
            return torch.tensor(labels).view(-1, 1) 

    def load_csv(self, file_path):
        return pd.read_csv(file_path) #Load metadata from a CSV file

    def __len__(self):
        return self.data_length # Return the number of samples in the dataset.

    def __getitem__(self, idx):
    
        with h5py.File(self.data_file, 'r') as file:
            image = file['x'][idx].astype('uint8')
        image = Image.fromarray(image, 'RGB')
       
        if self.test:
            image_tensor = F.to_tensor(image)
            image_tensor = F.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            image_tensor = apply_transforms(image)
        label = self.labels[idx]
 
        return transforms.ToTensor()(image), image_tensor, label





### 3.1 Define show_images functions to display and save images

def show_images(images, title, filename):
    """
    Display and save 20 images 

    Args:
        images: List of images to be displayed.
        title: The title for the set of images.
        filename: The filename where the image will be saved.
    """
    fig, axes = plt.subplots(4, 5, figsize=(20, 10))  #4 rows and 5 columns, 20 images
    fig.suptitle(title, fontsize=16)
    for img, ax in zip(images, axes.flatten()):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()



### 3.2 Randomly select 10 images from the training set and validation set for comparison, and classify and display images of 0 and 1

def compare_train_val_images(train_dataset, val_dataset, num_images=20):
    
    def get_images_by_label(dataset, label, num_images):
        """
        Obtain a specified number of images from the dataset based on labels

        argument:
        ataset: dataset
        label: labels are used to filter images
        num_images: Number of images

        return: selected images
        """
            
        # Filter out all image indexes that match the specified tags
        indices = [i for i in range(len(dataset)) if int(dataset[i][2].item()) == label]
        
        # A specified number of images are randomly selected from eligible indexes
        sampled_indices = random.sample(indices, min(num_images, len(indices)))
        
        # Extract and convert image data
        images = [dataset[i][0].permute(1, 2, 0).numpy() for i in sampled_indices]
        
        return images

    # apply 'get_images_by_label' to obtain images labeled 0 and 1 in the training set
    train_images_0 = get_images_by_label(train_dataset, 0, num_images)
    train_images_1 = get_images_by_label(train_dataset, 1, num_images)

    # obtain images with labels 0 and 1 in the validation set
    val_images_0 = get_images_by_label(val_dataset, 0, num_images)
    val_images_1 = get_images_by_label(val_dataset, 1, num_images)

    # The 'show_images' function is used to display the images with labels of 0 and 1 in the training and validation sets, respectively
    show_images(train_images_0, 'Train Images - Label 0', 'train_images_label_0.png')
    show_images(train_images_1, 'Train Images - Label 1', 'train_images_label_1.png')
    show_images(val_images_0, 'Validation Images - Label 0', 'val_images_label_0.png')
    show_images(val_images_1, 'Validation Images - Label 1', 'val_images_label_1.png')



### 4.3 Data exploration and Checking
def analyze_dataset(dataset, name):
    """
    Label distribution counts the number of times each label appears in the dataset.
    Duplicate image check identifies duplicate images by calculating the hash value of the image

    Parameters:
    dataset: dataset
    name: name of the dataset

    Output: Print the number of samples, label distribution, and number of duplicate images
    """
    
    # Collect all tags
    labels = []
    for _, _, label in dataset:
        labels.append(label.item())
    labels = np.array(labels)

     # Check for balance
    unique, counts = np.unique(labels, return_counts=True)
    label_distribution = dict(zip(unique, counts))

    # Check for duplicates
    image_hashes = set() # Used to store the hash value of each image
    duplicates = 0 # Used to count the number of repeated images
    
    # Iterate over each image in the dataset
    for idx in range(len(dataset)):
        image_tensor, _, _ = dataset[idx] # Get image tensor from dataset
        image_np = image_tensor.permute(1, 2, 0).numpy() # Convert image tensor to NumPy array
        image_hash = hash(image_np.tobytes()) # Calculate the hash value of the image
        if image_hash in image_hashes:
            duplicates += 1
        else:
            image_hashes.add(image_hash)
    
    # print results
    print(f"\n{name} Dataset Analysis:")
    print(f"Number of Samples: {len(dataset)}")
    print(f"Label Distribution: {label_distribution}")
    print(f"Number of Duplicates: {duplicates}")



### 4.4 Save original and enhanced images

def save_display_images(original_images, transformed_images, filename):
    """
    Save and display the original image and the transformed image. The original image is shown at the top and the transformed image is shown at the bottom.

    Parameters:
    original_images: list of original images
    transformed_images: list of transformed images
    filename: filename to save the image to
    """
    
    # 2-row grid of subimages, the first row is to display the original image, and the second row is used to display the transformed image
    fig, axes = plt.subplots(2, len(original_images), figsize=(15, 5))
    
    for i in range(len(original_images)):
        # Display the original image
        axes[0, i].imshow(original_images[i])
        axes[0, i].set_title('Original Image', fontsize=10)
        axes[0, i].axis('off')
        
        # Display the transformed image
        axes[1, i].imshow(transformed_images[i].permute(1, 2, 0))
        axes[1, i].set_title('Transformed Image', fontsize=10)
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



### 4.5 Calculate and compare data characteristics
def compare_dataset_statistics(train_dataset, val_dataset, test_dataset):
    def calculate_statistics(dataset):
        """
        The mean and standard deviation of the images in the dataset are calculated.
        Parameter: dataset
        Return: The mean and standard deviation of the images in the dataset.
        """
        means = []
        stds = []
        # Calculate the mean and standard deviation for each image
        for image, _, _ in dataset:
            means.append(image.mean().item())
            stds.append(image.std().item())
        # Calculate the mean of the mean and standard deviation of all images
        return np.mean(means), np.mean(stds)

    train_mean, train_std = calculate_statistics(train_dataset)
    val_mean, val_std = calculate_statistics(val_dataset)
    test_mean, test_std = calculate_statistics(test_dataset)
    
    # Print the comparison results
    print(f"\nCompare dataset statistics")
    print(f"Train Dataset - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    print(f"Validation Dataset - Mean: {val_mean:.4f}, Std: {val_std:.4f}")
    print(f"Test Dataset - Mean: {test_mean:.4f}, Std: {test_std:.4f}")
    print(f"\n")
    
    
    
### 4.6 print model summary
def print_model_summary(model, indent=0):
    for name, module in model.named_children():
        print(' ' * indent + f"{name}: {module.__class__.__name__}")
        print_model_summary(module, indent + 2)
    
    
    
    
    
    
### 5. Model   

class MobileNetV3Classifier(nn.Module):
    """
    Binary classification based on MobileNetV3.
    The pre-trained MobileNetV3-large was used as the feature extractor, and its classifier section was modified to accommodate the binary classification task.

    Method:
    '__init__': Initializes the model, adjusts the convolutional layer and classifier.
    'forward': Defines the forward propagation process.

    Attribute: 'backbone': Pre-trained MobileNetV3-large network
    
    """
    
    def __init__(self):
        """
        Initialize the MobileNetV3Classifier class
        Load the pretrained MobileNetV3-large model
        Modify the first layer of convolution to accommodate the number of channels of the input image
        Replace the last few layers of the classifier for binary classification tasks.
        """
        
        super().__init__()
        
        # Load the pre-trained MobileNetV3-large model
        self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        
        # Modify the first layer of convolution 
        self.backbone.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust first conv layer
        
        # Replace the classifier section to fit the binary classification task
        num_ftrs = self.backbone.classifier[3].in_features
        
        self.backbone.classifier[3] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8), # Dropout layer to reduce overfitting
            nn.Linear(512, 1),
            nn.Sigmoid() # Sigmoid activates the function to compress the output to between [0, 1].
        )

    # Forward propagation process
    def forward(self, x):
        return self.backbone(x)



class ShuffleNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained shufflenet_v2_x1_0 model
        self.backbone = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        # Adjust first conv layer 
        self.backbone.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)  
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


class SqueezeNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the pre-trained shufflenet_v2_x1_0 model
        self.backbone = models.squeezenet1_0(weights='IMAGENET1K_V1')
        
        # Adjust first conv layer 
        self.backbone.features[0] = nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, bias=False) 
        
        self.backbone.classifier[1] = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        self.backbone = nn.Sequential(self.backbone, nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 1)


class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the pre-trained resnet18 model
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Adjust first conv layer 
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove max pooling layer to handle smaller input
        self.backbone.maxpool = nn.Identity()  
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the pre-trained efficientnet_b0 model
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Adjust first conv layer 
        self.backbone.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust first conv layer
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(512, 1)
        )
        self.backbone = nn.Sequential(self.backbone, nn.Sigmoid())

    def forward(self, x):
        return self.backbone(x)






### 6. Training function

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    """
    Train and validate the model

    parameters:
    model (nn. Module): The model to be trained.
    train_loader (DataLoader): The DataLoader of the training data.
    val_loader (DataLoader): The DataLoader that validates the data.
    criterion (nn. Module): loss function.
    optimizer (torch.optim.Optimizer): Optimizer.
    num_epochs (int, optional): 50 epochs

    return:
    model (nn. Module): The model after the training is complete.
    best_acc: The best validation accuracy during training.
    best_auc: Best validated AUC score during training.
    train_losses : A list of losses for each training cycle.
    val_losses : A list of losses for each validation cycle.
    train_accuracies : A list of accuracy rates for each training cycle.
    val_accuracies: A list of accuracy rates for each validation cycle.

    """
    
    # Set up the training device (CUDA or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initial weights for the backup model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 # The best accuracy of initialization
    best_auc = 0.0 # The best AUC of initialization
    since = time.time() # Record the start time
    
    # Used to record loss and accuracy during training and validation
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    
    # iterate each training EPOCH
    for epoch in range(num_epochs):
        
        # 1.Print epoch start
        print('-' * 20)
        print(f'Epoch {epoch}/{num_epochs - 1}') # Print the current cycle
        
        # 2.Start a loop of iteration of training and validation
        for phase in ['train', 'val']:
            
            # 2.1 Set the model mode and select the data loader
            if phase == 'train':
                model.train()   # Set the model to training mode
                dataloader = train_loader # Select the Training Data Loader
            else:
                model.eval()   # Set the model to evaluation mode 
                dataloader = val_loader # Select the validation Data Loader
            
            # 2.2 Losses in the initialization phase, correctly predicted quantities, and result storage
            running_loss = 0.0 # Initialization loss
            running_corrects = 0 # Initialize the correct forecast quantity
            all_labels = [] # Store all labels
            all_outputs = [] # Store all outputs
            
            # 2.3 iterate each batch in the data loader
            for inputs, _, labels in dataloader:
                # Move input and label data to the device (GPU or CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # use gradient calculations in training mode
                with torch.set_grad_enabled(phase == 'train'):
                    # To perform forward propagation, the model instance is invoked and input data is passed, and the output is the prediction result
                    outputs = model(inputs)
                    # calculate loss 
                    loss = criterion(outputs, labels)
                    # Boolean conditional operation that compares each binary classification result in outputs to 0.5
                    preds = outputs > 0.5

                    if phase == 'train':
                        # optimizer is used to adjust the weights of the model
                        optimizer.zero_grad() # Clear the previously calculated gradient
                        loss.backward() # Perform backpropagation
                        optimizer.step() # Update the weights of the model with the calculated gradient

                
                # Update losses and correctly forecast quantities
                running_loss += loss.item() * inputs.size(0) # Accumulate the losses of the current batch
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())  # The current batch label is added to the list for subsequent calculations
                all_outputs.extend(outputs.detach().cpu().numpy()) # Add the current batch label to the list
            
            # 2.4 Calculate the average loss, accuracy, and AUC score for each epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_auc = roc_auc_score(all_labels, all_outputs)
               
            # 2.5 Record the results of the training and validation phases

            if phase == 'train':
                train_losses.append(epoch_loss) # 'append' adds the value of 'epoch_loss' to the 'train_losses' list
                train_accuracies.append(epoch_acc) # Recording accuracy
                scheduler.step()  # Call the scheduler's step method at the end of the training phase
            else:
                val_losses.append(epoch_loss)  # validation losses are recorded
                val_accuracies.append(epoch_acc) # Document the accuracy of validation
                early_stopping(epoch_loss, model) # Apply  early stop 

                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(best_model_wts) # Restore the best model weights
                    return model, best_acc, best_auc, train_losses, val_losses, train_accuracies, val_accuracies

            # 2.6 Print the loss and accuracy of the current stage
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ')
            
            # 2.7 Update the best models
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    # Calculate the training time
    time_elapsed = time.time() - since
    
    # Print training time, best validation accuracy, and AUC score
    print(f'\n Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} AUC: {best_auc:.4f}')
    
    # Restore the best model weights
    model.load_state_dict(best_model_wts)
    
    return model, best_acc, best_auc, train_losses, val_losses, train_accuracies, val_accuracies






####  7.Test function
def test_model(model, test_loader, criterion):
    """
    Evaluate the performance of a trained model on a test dataset.

    This function calculates and returns various performance metrics for the model,
    including loss, accuracy, AUC , precision, and recall.
    The model is evaluated in inference mode with no gradient calculations.

    Args:
        model: The trained model 
        test_loader : DataLoader providing batches of test data.
        criterion : Loss function used to compute the loss.

    Returns:
        tuple: A tuple containing the following metrics over the test dataset:
            test_loss : The average loss
            test_acc : The accuracy of the model
            test_auc: The AUC 
            test_precision: The precision 
            test_recall: The recall
    """
    
    # 7.1. Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 7.2. Set the model to evaluation mode
    model.eval()
    
    # 7.3. Initialize variables
    running_loss = 0.0
    running_corrects = 0
    all_labels = [] # Store all authentic tags
    all_outputs = [] # Store all output
    
    # 7.4. Gradient calculations are not used
    with torch.no_grad():
    
        # iterate each batch in the test data loader
        for inputs, _, labels in test_loader:
            inputs = inputs.to(device) # Move input data to the device (GPU or CPU)
            labels = labels.to(device) # Move the label to the device
            
            outputs = model(inputs) # Forward propagation
            loss = criterion(outputs, labels) # Calculate the loss
            preds = outputs > 0.5 # Boolean conditional operation that compares each binary classification result in outputs to 0.5

            running_loss
            running_loss += loss.item() * inputs.size(0) #Cumulative losses
            running_corrects += torch.sum(preds == labels.data) # Sum up the number of correctly predicted ones
            all_labels.extend(labels.cpu().numpy())  # The current batch label is added to the list for subsequent calculations
            all_outputs.extend(outputs.detach().cpu().numpy())  # The current batch output is added to the list for subsequent calculations

    
    # 7.5. Calculate the test set loss, accuracy，AUC, precision，recall
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    test_auc = roc_auc_score(all_labels, all_outputs)
    test_precision = precision_score(all_labels, np.round(all_outputs))
    test_recall = recall_score(all_labels, np.round(all_outputs))

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f} Precision: {test_precision:.4f} Recall: {test_recall:.4f}')
    return test_loss, test_acc, test_auc, test_precision, test_recall




### 8.1 Calculate the run time

def runtime(model, loader):
    """
    Measure the inference time of model
    The model is evaluated in inference mode, and the time is measured from the start of the inference process until it completes.

    Args:
        model: The model to be evaluated for inference time.
        loader : DataLoader, data for inference

    Returns: The total time (in seconds) taken for the model to perform inference on the dataset.
    """
    
    # 1. Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. Moves the model to the selected device
    model.to(device)
    
    # 3. Set the model to evaluation mode
    model.eval()
    
    # 4. Record the start time
    start_time = time.time()
    
    # 5. Gradient calculations are not used
    with torch.no_grad():
        # iterate each batch in the data loader
        for inputs, _, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) # Perform forward propagation to get the model output
    
    # 6. Record the end time
    end_time = time.time()
    
    # 7. Calculate and return the run time
    runtime = end_time - start_time
    
    return runtime


### 8.2 calculate FLOPs
def calculate_flops(model, input_size=(3, 96, 96)):
    """
    Calculate FLOPs and the number of parameters for a given model based on input size.
    This function uses the `profile` function from the `torchprofile` library to compute the FLOPs and the number of parameters of the model. 

    Args:
        model: The model for which to calculate FLOPs and parameters.
        input_size : The shape of the input tensor (channels, height, width). Default is (3, 96, 96).

    Returns:
        tuple: A tuple containing:
            - flops: The total number of floating-point operations performed by the model.
            - params: The total number of parameters in the model.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create an analog input tensor whose shape matches the input dimensions
    inputs = torch.randn(1, *input_size).to(device)
    # Use the profile function to calculate the number of FLOPs and parameters for the model
    flops, params = profile(model, inputs=(inputs,))
    
    return flops, params



### 9. EarlyStopping
class EarlyStopping:
    """
    Early stopping mechanism to halt training when the validation loss stops improving, to prevent overfitting and save computational resources.
    This class monitors the validation loss and stops the training if there is no improvement after a specified number of epochs (patience). 

    Attributes:
        patience: Number of epochs to wait for improvement after the last best validation loss.
        delta: Minimum change in validation loss to qualify as an improvement.
        counter: Counter for the number of epochs with no improvement.
        best_loss : Best observed validation loss so far.
        early_stop: Flag indicating whether training should stop.
    """
    
    def __init__(self, patience=7, delta=0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience: Number of epochs with no improvement to wait before stopping.
            delta: Minimum change in validation loss to be considered an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Check if training should be stopped based on the current validation loss.

        Args:
            val_loss: Current validation loss to be compared with the best loss.
            model: The model being trained (not used in this implementation).

        Updates:
            self.counter: Incremented if no improvement is observed.
            self.best_loss: Updated if the current validation loss is better.
            self.early_stop: Set to True if early stopping criteria are met.
        """           

        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



### Inspecting and converting tensors
def to_cpu_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    return tensor






### Main block

if __name__ == '__main__':
 
    # 1.数据集初始化
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
                                   test=True) # test set data do not apply data augmentation


    ### 2.Dataset exploration
    
    # explore train, validation, and test datasets
    analyze_dataset(train_dataset, "Train")
    analyze_dataset(val_dataset, "Validation")
    analyze_dataset(test_dataset, "Test")
    
    # Compare and display images
    compare_train_val_images(train_dataset, val_dataset)

    # Compare and display statistics for train, validation, and test datasets
    compare_dataset_statistics(train_dataset, val_dataset, test_dataset)

    # Randomly select a few images and display them
    sample_indices = random.sample(range(len(train_dataset)), 5)
    original_images = []
    transformed_images = []

    for idx in sample_indices:
        original_image, transformed_image, _ = train_dataset[idx]
        original_images.append(original_image.permute(1, 2, 0).numpy())
        transformed_images.append(transformed_image)
    
    save_display_images(original_images, transformed_images, 'original_and_transformed_images.png')


    ### 3.DataLoader initialization
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    
    
    ### 4.Model dictionary definition
    models_dict = {
        'MobileNetV3': MobileNetV3Classifier(),
        'ShuffleNetV2': ShuffleNetV2Classifier(),
        #'SqueezeNet': SqueezeNetClassifier(),
        'EfficientNet': EfficientNetClassifier(),
        'ResNet18': ResNetClassifier(),
    }
    
    # Define model names from the dictionary keys
    model_names = list(models_dict.keys())

    # Define loss function
    # criterion = nn.BCELoss()



    ### 5.Print the architecture of each model
    for model_name, model in models_dict.items():
        print(f"\n{model_name} structure:\n")
        print_model_summary(model)
    
    
    ### 6.Initialize dictionaries to store results
    results = {}
    heatmaps = {}
    train_losses_dict = {}
    val_losses_dict = {}
    train_accuracies_dict = {}
    val_accuracies_dict = {}
    flops_dict = {}


    ### 7.Train and evaluate each model
    for model_name, model in models_dict.items():
        print(f'\n[Training {model_name} model...]\n')
       
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Adding L2 regularization
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Define learning rate scheduler，reduce LR by a factor of 0.1 every 10 epochs
        criterion = nn.BCELoss()  # Define learning rate scheduler
        early_stopping = EarlyStopping(patience=15, delta=0.01) # early stopping 

        # Train the model and record the validation metrics
        model, best_acc, best_auc, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
        # Test the model and record test metrics
        test_loss, test_acc, test_auc, test_precision, test_recall = test_model(model, test_loader, criterion)
        
        # Calculate total number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # calculate inference time per image
        inference_time = runtime(model, test_loader) / len(test_loader.dataset)
        
        # calculate FLOPs
        flops, params = calculate_flops(model)
        flops_dict[model_name] = flops
        
        # Store results 
        results[model_name] = {
            'Test Loss': test_loss,
            'Test Accuracy': test_acc.item(),
            'Test AUC': test_auc,
            'Precision': test_precision,
            'Recall': test_recall,
            'Total Parameters': total_params,
            'Inference Time per Image': inference_time,
            'FLOPs': flops
        }

        # Store training and validation metrics
        train_losses_dict[model_name] = train_losses
        val_losses_dict[model_name] = val_losses
        train_accuracies_dict[model_name] = train_accuracies
        val_accuracies_dict[model_name] = val_accuracies

    ### 8.Print results for each model
    for model_name, metrics in results.items():
        print(f'\n{model_name} Results:')
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value}')
    
    ### 9.Extract and display metrics for comparison
    model_names = list(results.keys())
    accuracies = [results[model]['Test Accuracy'] for model in model_names]
    aucs = [results[model]['Test AUC'] for model in model_names]
    precisions = [results[model]['Precision'] for model in model_names]
    recalls = [results[model]['Recall'] for model in model_names]
    params = [results[model]['Total Parameters'] for model in model_names]
    inf_times = [results[model]['Inference Time per Image'] for model in model_names]
    flops_values = [results[model]['FLOPs'] for model in model_names]
    
    
    
    ### 10. Plots
    
    # Plot training and validation accuracy curves
    for model_name in model_names:
        train_accuracies = [to_cpu_numpy(acc) for acc in train_accuracies_dict[model_name]]
        val_accuracies = [to_cpu_numpy(acc) for acc in val_accuracies_dict[model_name]]
       
        plt.figure()
        plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
        plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch') # X-axis label
        plt.ylabel('Accuracy') # Y-axis label
        plt.title(f'{model_name} Training and Validation Accuracy') # plot title
        plt.legend()
        plt.savefig(f'{model_name}_accuracy_curve.png') # Save the plot as a PNG file
        plt.close()


    # 绘制训练和验证损失曲线
    for model_name in model_names:
        train_losses = [to_cpu_numpy(loss) for loss in train_losses_dict[model_name]]
        val_losses = [to_cpu_numpy(loss) for loss in val_losses_dict[model_name]]
        
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epoch') # X-axis label
        plt.ylabel('Loss') # Y-axis label
        plt.title(f'{model_name} Training and Validation Loss') # plot title
        plt.legend()
        plt.savefig(f'{model_name}_loss_curve.png') # Save the plot as a PNG file
        plt.close()


    
    # figure1: Model Accuracy Comparison
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Model') # X-axis label
    ax1.set_ylabel('Accuracy', color=color) # Y-axis label
    ax1.bar(model_names, accuracies, color=color, alpha=0.8, label='Accuracy') # plot accuracy bar chart
    ax1.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust the layout of the drawing
    plt.title('Model Accuracy Comparison') # plot title
    plt.savefig('model_accuracy_comparison.png') # Save the plot as a PNG file

   # figure2: Model AUC Comparison
    fig, ax2 = plt.subplots()

    color = 'tab:blue'
    ax2.set_xlabel('Model') # X-axis label
    ax2.set_ylabel('AUC', color=color) # Y-axis label
    ax2.bar(model_names, aucs, color=color, alpha=0.8, label='AUC') # plot bar chart
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust the layout of the drawing
    plt.title('Model AUC Comparison') # plot title
    plt.savefig('model_auc_comparison.png') # Save the plot as a PNG file


    # figure3: Model Recall Comparison
    fig, ax3 = plt.subplots()

    color = 'tab:blue'
    ax3.set_xlabel('Model') # X-axis label
    ax3.set_ylabel('Recall', color=color) # Y-axis label
    ax3.bar(model_names, recalls, color=color, alpha=0.8, label='Recall') # plot bar chart
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust the layout of the drawing
    plt.title('Model Recall Comparison') # plot title
    plt.savefig('model_recall_comparison.png') # Save the plot as a PNG file


    # figure4: Model Parameters Comparison
    fig, ax4 = plt.subplots()

    color = 'tab:green'
    ax4.set_xlabel('Model') # X-axis label
    ax4.set_ylabel('Total Parameters', color=color) # Y-axis label
    ax4.bar(model_names, params, color=color, alpha=0.6, label='Total Parameters') # plot bar chart
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust the layout of the drawing
    plt.title('Model Parameters Comparison') # plot title
    plt.savefig('model_parameters_comparison.png') # Save the plot as a PNG file


    # figure5: Model Inference Time Comparison
    fig, ax5 = plt.subplots()

    color = 'tab:green'
    ax5.set_xlabel('Model') # X-axis label
    ax5.set_ylabel('Inference Time (s)', color=color) # Y-axis label
    ax5.bar(model_names, inf_times, color=color, alpha=0.8, label='Inference Time') # plot bar chart
    ax5.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout of the drawing
    plt.title('Model Inference Time Comparison') # plot title
    plt.savefig('model_inference_time_comparison.png') # Save the plot as a PNG file


    # figure6: Model Precision Comparison
    fig, ax6 = plt.subplots()

    color = 'tab:blue'
    ax6.set_xlabel('Model') # X-axis label 
    ax6.set_ylabel('Precision', color=color) # Y-axis label
    ax6.bar(model_names, precisions, color=color, alpha=0.8, label='Precision') # plot bar chart
    ax6.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust the layout of the drawing
    plt.title('Model Precision Comparison') # plot title
    plt.savefig('model_precision_comparison.png') # Save the plot as a PNG file


     # figure7: Model FLOPs Comparison
    fig, ax7 = plt.subplots()

    color = 'tab:green'
    ax7.set_xlabel('Model') # X-axis label
    ax7.set_ylabel('FLOPs', color=color)  # Y-axis label
    ax7.bar(model_names, flops_values, color=color, alpha=0.8, label='FLOPs') # plot bar chart
    ax7.tick_params(axis='y', labelcolor=color)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust the layout of the drawing
    plt.title('Model FLOPs Comparison') # plot title
    plt.savefig('model_flops_comparison.png') # Save the plot as a PNG file

 

    # figure 8: Model Accuracy, FLOPs and Parameters Comparison (Bubble Chart)
    fig, ax8 = plt.subplots()

    scatter = ax8.scatter(accuracies, inf_times, s=[param / 1e4 for param in params], alpha=0.5, c='gray') # Draw a bubble chart
    
    for i, model_name in enumerate(model_names):
        ax8.annotate(model_name, (accuracies[i], inf_times[i])) # Label each bubble

    ax8.set_xlabel('Accuracy') # X-axis label
    ax8.set_ylabel('Inference Time (s)')  # Y-axis label
    ax8.set_title('Model Accuracy, Inference Time, Parameters comparison)') # plot title
    ax8.grid(True)
    
    plt.savefig('model_accuracy_inferencetime_Flops.png') # Save the plot as a PNG file


    print("Training and evaluation completed, and plots are saved as files.")
