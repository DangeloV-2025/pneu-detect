#import all neccesary libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, models, transforms
# Resnet# is number of layers
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# Add decay to penalize large weights
# (This prevents overfitting)
decay = 0.001
learning_rate = 0.001
# of images sent to GPU each time
   
batch_size = 144

# num of runs through entire dataset
num_epochs = 5

# if blank start training from random
#start_model = ""
#start_model = "resnet18_best.pth"
start_model = ""


save_path = "models_1:24/"
data_path = "data/"

# made a list for each folder in the dataset
data_groups = ["test", "train", "val"]
group_datasets = {}
# schedules batching... so cpu knows when to sit
group_loaders = {}

# Detect hardware env  
if torch.cuda.is_available():
    print("CUDA Detected: Using CUDA to run")
    device = torch.device("cuda")
else:
    print("CUDA not Detected: Using CPU to run")
    device = torch.device("cpu")
print()


img_transform = transforms.Compose(
    [
        # Rotate random up to 30 degrees
            # Creates more variation in the data
        transforms.RandomRotation(degrees=30),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize to standard resnet input
            # Recommended Resnet mean and std from docs
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Check what image look life after transformation
def show_transformed_image(dataset, index):
    img, _ = dataset[index]
    img_np = np.transpose(img.numpy(), (1, 2, 0))

    plt.imshow(img_np)
    plt.title(f"Class: {dataset.classes[index]}, Index: {index}")
    plt.axis("off")
    plt.show()


for data_group in data_groups:
    # get path
    print(f"Parsing data group: {data_group}")
    total_path = data_path + data_group
    gp_dataset = datasets.ImageFolder(root=total_path, transform=img_transform)
    group_datasets[data_group] = gp_dataset
   
    # Shuffle training data for generalizability 
    do_shuffle = data_group == "train"
    print(f"Loading data with shuffle={do_shuffle}")
    # made a loader for the dataset 
    group_loaders[data_group] = torch.utils.data.DataLoader(
        gp_dataset, batch_size=batch_size, shuffle=do_shuffle
    )

    total_classes = len(gp_dataset.classes)

    print(f"Identified {total_classes} classes:")
    for class_idx in range(total_classes):
        class_name = gp_dataset.classes[class_idx]
        imge_count = len(os.listdir(total_path + "/" + class_name))
        print(f"{class_idx}: {class_name}, {imge_count} images")
    print()

    # show_transformed_image(gp_dataset, 0)

# Load pretrained ResNet-18
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# Modify the classifier of ResNet-18
    # Baisically look for features
num_ftrs = resnet.fc.in_features
# Modify output to match number of classes
    # adds one more layer of two nodes because output layer needs to be two
    # in cases of binary classification
    # https://www.researchgate.net/figure/The-neural-network-structure-for-binary-classification_fig1_359138569 
resnet.fc = nn.Linear(num_ftrs, len(gp_dataset.classes))
# Load existing model weights if specified
    # Baisically lets you pause and go back to old b
if start_model:
    print(f"Loading saved model: {start_model}")
    resnet.load_state_dict(torch.load(save_path + start_model))
# move to CUDA if applicable
resnet = resnet.to(device)

# Set optimizer, learning rate, and loss function
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=decay)
# calculates difference in output node to ideal 
# https://miro.medium.com/v2/resize:fit:1358/1*TtyZT07Qc8ebtF8cWMau4A.png 
criterion = nn.CrossEntropyLoss()


def train_model(model, criterion, optimizer):
    for epoch in range(num_epochs):
        # Parameters to print during running
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training phase
        model.train()
        with tqdm(
            group_loaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as iter:
            for inputs, labels in iter:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # set all gradients to zero at start of epoch
                optimizer.zero_grad()
                # Forward propagation
                outputs = model(inputs)
                # calculates difference in expected 
                loss = criterion(outputs, labels)
                # Backpropagation
                loss.backward()
                # Calculates and applies change in weights and biases
                optimizer.step()

                # Log evaluation data
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Update status bar
                iter.set_postfix(
                    loss=running_loss / total_predictions,
                    acc=correct_predictions / total_predictions,
                )

        epoch_loss = running_loss / len(group_datasets["train"])
        epoch_accuracy = correct_predictions / total_predictions

        # Validation phase (evaluate on validation set)
        val_loss, val_accuracy = evaluate_model(model, criterion)

        # Print progress
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
            f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_accuracy:.4f}"
        )

        total_path = save_path + f"resnet18_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), total_path)
        print(f"Model saved at {total_path}")
        

        print()


def evaluate_model(model, criterion):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in group_loaders["val"]:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to CUDA

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(group_datasets["val"])
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy


print("Starting training")
print()
train_model(resnet, criterion, optimizer)