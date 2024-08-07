import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "data/"
batch_size = 144

# Define a simple transformation to convert images to tensors and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
data_groups = ["test", "train", "val"]
group_datasets = {}
group_loaders = {}

for data_group in data_groups:
    total_path = os.path.join(data_path, data_group)
    group_datasets[data_group] = datasets.ImageFolder(root=total_path, transform=transform)
    group_loaders[data_group] = torch.utils.data.DataLoader(
        group_datasets[data_group], batch_size=batch_size, shuffle=True
    )

# Function to convert dataset to numpy arrays
def dataset_to_numpy(loader):
    data_list = []
    label_list = []
    
    for inputs, labels in tqdm(loader):
        data_list.append(inputs.view(inputs.size(0), -1).numpy())
        label_list.append(labels.numpy())
    
        
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    
    print("DATA TRANSFER COMPLETE")
    return data, labels

# Convert train, val, and test datasets to numpy arrays
train_data, train_labels = dataset_to_numpy(group_loaders["train"])
val_data, val_labels = dataset_to_numpy(group_loaders["val"])
test_data, test_labels = dataset_to_numpy(group_loaders["test"])

# Standardize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# Initialize and train logistic regression model with a progress bar
clf = LogisticRegression(max_iter=1000, verbose=1)
clf.fit(train_data, train_labels)

# Evaluate on validation set
val_predictions = clf.predict(val_data)
print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
print(classification_report(val_labels, val_predictions, target_names=group_datasets["train"].classes))


# Evaluate on test set
test_predictions = clf.predict(test_data)
print("Test Accuracy:", accuracy_score(test_labels, test_predictions))
print(classification_report(test_labels, test_predictions, target_names=group_datasets["train"].classes))

# Confusion matrix for test set
test_cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=group_datasets["train"].classes, yticklabels=group_datasets["train"].classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.show()