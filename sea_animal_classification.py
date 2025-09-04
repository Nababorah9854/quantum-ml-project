"""# Library Imports"""

# General Utilities
import time
import os
import sys
import random
from collections import Counter

# Numerical Computation & Visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Quantum Machine Learning
import pennylane as qml

# PyTorch: Deep Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler

# Torchvision: Dataset & Model Utilities
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    ResNet34_Weights,
    ResNet101_Weights,
    DenseNet121_Weights,
)

# Scikit-learn: Evaluation Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

"""# Basic Setup & Hyperparameters"""

# === Configuration ===
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
N_QUBITS = 8
LEARNING_RATE = 0.00002
WEIGHT_DECAY = 5e-5
LR_FACTOR = 0.2
LR_PATIENCE = 8
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
UNDERSAMPLE_TURTLE_TORTOISE = True
DATA_DIR = "data/sea-animals-image-dataset"


"""# Reproducibility Controls"""

qml.numpy.random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ['NCCL_DEBUG'] = 'INFO'
"""# Device Configuration"""

if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"\n💻 Using device: {device} ({num_gpus} GPU(s))")
    if num_gpus > 1:
        torch.cuda.set_device(0)
        print(f"Setting default CUDA device to index 0.")
else:
    device = torch.device("cpu")
    num_gpus = 0
    print(f"\n💻 Using device: {device} (CPU)")

"""# Data Transformations"""

transform_train = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

"""# Dataset Loading & Summary"""

transform_initial = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
try:
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform_initial)
    classes = full_dataset.classes
    targets = np.array(full_dataset.targets)
    num_classes = len(classes)

    print("📂 Classes:", classes)
    print("📊 Number of classes:", num_classes)
    print("🖼️ Total number of images:", len(full_dataset.samples))
    print("\n🔍 Dataset Description:")
    class_counts_dict = Counter(full_dataset.targets)
    for i, class_name in enumerate(classes):
        num_samples = class_counts_dict[i]
        print(f"  - Class '{class_name}': {num_samples} images")

    # --- Displaying Sample Images ---
    num_samples_to_display = 3
    fig, axes = plt.subplots(num_classes, num_samples_to_display, figsize=(15, 5 * num_classes))
    fig.suptitle("Sample Images from Each Class", fontsize=16)
    for i, class_name in enumerate(classes):
        indices = np.where(targets == i)[0]
        random_indices = np.random.choice(indices, min(num_samples_to_display, len(indices)), replace=False)
        for j, img_index in enumerate(random_indices):
            image, _ = full_dataset[img_index]
            axes[i, j].imshow(image.permute(1, 2, 0))
            axes[i, j].set_title(f"{class_name}")
            axes[i, j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Bar Plot of Class Distribution ---
    plt.figure(figsize=(10, 5))
    plt.bar(classes, list(class_counts_dict.values()))
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Images per Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"⚠ Error loading dataset: {e}")
    sys.exit(1)

"""# Dataset Splitting & Class Imbalance Handling"""

# === Data Splitting and Handling Imbalance ===
train_dataset, val_dataset, test_dataset = None, None, None
if UNDERSAMPLE_TURTLE_TORTOISE and 'Turtle_Tortoise' in classes:
    turtle_tortoise_index = classes.index('Turtle_Tortoise')
    turtle_tortoise_indices = np.where(targets == turtle_tortoise_index)[0]
    other_indices = np.where(targets != turtle_tortoise_index)[0]
    target_turtle_count = int(np.median(list(class_counts_dict.values())))
    undersampled_turtle_indices = np.random.choice(turtle_tortoise_indices, size=target_turtle_count, replace=False)
    balanced_indices = np.concatenate([other_indices, undersampled_turtle_indices])
    balanced_targets = targets[balanced_indices]

    train_idx, val_test_idx = train_test_split(
        balanced_indices, test_size=VALIDATION_SPLIT + TEST_SPLIT, stratify=balanced_targets, random_state=SEED
    )
    val_test_targets = balanced_targets[np.isin(balanced_indices, val_test_idx)]
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT), stratify=val_test_targets, random_state=SEED
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
else:
    train_size = int(len(full_dataset) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )

if train_dataset and val_dataset and test_dataset:
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    test_dataset.dataset.transform = transform_test

    # === Data Loaders ===
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
else:
    print("Error: Datasets not properly created.")
    sys.exit(1)

"""# Quantum Layer Setup"""

# === Quantum Layer Setup ===
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_layer(inputs):
    """Quantum layer circuit."""
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
    for i in range(0, N_QUBITS - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS))

"""# Hybrid Quantum-Classical Model: HybridQCNN"""

# === Hybrid Quantum-Classical Model ===
class HybridQCNN(nn.Module):
    """Hybrid quantum-classical convolutional neural network."""

    def __init__(self, backbone, num_classes, n_qubits=N_QUBITS):
        super().__init__()
        in_features = self._get_in_features(backbone)
        self.feature_extractor = self._remove_classifier(backbone)
        self.n_qubits = n_qubits
        self.fc1 = nn.Sequential(
            nn.Linear(in_features + n_qubits, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        self.fc2 = nn.Linear(128, num_classes)

    def _get_in_features(self, model):
        if hasattr(model, "fc"):
            return model.fc.in_features
        elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
            return model.classifier.in_features
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            for layer in reversed(model.classifier):
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        raise ValueError("Cannot find the final fully connected layer for feature size.")

    def _remove_classifier(self, model):
        if hasattr(model, "fc"):
            model.fc = nn.Identity()
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        return model

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        q_feats = [quantum_layer(sample[:self.n_qubits].cpu().detach().numpy()) for sample in x]
        quantum_features = torch.tensor(q_feats, dtype=torch.float32).to(x.device)
        x = torch.cat((x, quantum_features), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

"""# Classical CNN Model"""

# === Classical CNN Model ===
class ClassicalCNN(nn.Module):
    """Classical convolutional neural network."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        in_features = self._get_in_features(backbone)
        self.feature_extractor = self._remove_classifier(backbone)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        self.fc2 = nn.Linear(128, num_classes)

    def _get_in_features(self, model):
        if hasattr(model, "fc"):
            return model.fc.in_features
        elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
            return model.classifier.in_features
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            for layer in reversed(model.classifier):
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        raise ValueError("Cannot find the final fully connected layer for feature size.")

    def _remove_classifier(self, model):
        if hasattr(model, "fc"):
            model.fc = nn.Identity()
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        return model

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

"""# Model Training Function"""

# === Model Training Function ===
def train_model(model, train_loader, val_loader, num_classes):
    """Trains the given model."""
    print(f"\n🚀 Training {model.__class__.__name__}...")
    model.to(device)
    if num_gpus > 1:
        print("Using DataParallel for multi-GPU training.")
        model = nn.DataParallel(model)

    # --- Calculate class weights ---
    class_counts_train = Counter()
    for _, labels in train_loader:
        class_counts_train.update(labels.cpu().numpy())
    total_samples_train = sum(class_counts_train.values())
    class_weights = torch.ones(num_classes, dtype=torch.float).to(device)
    if total_samples_train > 0:
        sorted_weights = [total_samples_train / (num_classes * class_counts_train.get(i, 1e-6)) for i in range(num_classes)]
        class_weights = torch.tensor(sorted_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    all_val_preds, all_val_labels = [], []
    all_val_images = []

    scaler = GradScaler()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        current_val_preds, current_val_labels = [], []
        current_val_images = []

        with torch.no_grad():
            for images, labels in val_loader:
                images_cpu = images.cpu()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                current_val_preds.extend(predicted.cpu().numpy())
                current_val_labels.extend(labels.cpu().numpy())
                current_val_images.extend(images_cpu.numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        all_val_preds = current_val_preds
        all_val_labels = current_val_labels
        all_val_images = current_val_images

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        scheduler.step(val_loss)

    return (
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        all_val_preds,
        all_val_labels,
        all_val_images
    )

"""# Model Testing Function"""

# === Model Testing Function ===
def test_model(model, test_loader, full_dataset_classes):
    """Tests the given model."""
    print("\n🔍 Testing started...")
    model.to(device)

    if num_gpus > 1:
        print("Using DataParallel for multi-GPU testing.")
        model = nn.DataParallel(model)

    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_test_preds, all_test_labels = [], []
    all_test_images = []
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images_cpu = images.cpu()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            all_test_images.extend(images_cpu.numpy())

    test_accuracy = 100 * correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    print(f"\n✅ Test Accuracy: {test_accuracy:.2f}%")
    print(f"📉 Test Loss: {avg_test_loss:.4f}")
    print("\n📊 Classification Report:\n")
    print(classification_report(all_test_labels, all_test_preds, target_names=full_dataset_classes))

    cm = confusion_matrix(all_test_labels, all_test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset_classes, yticklabels=full_dataset_classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return all_test_preds, all_test_labels, test_accuracy, all_test_images

"""# Model Comparison and Metrics Plotting Functions"""

# === Model Comparison Plot ===
def plot_model_comparison(all_model_accuracies):
    """Plots and compares the validation accuracies of all models."""
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in all_model_accuracies.items():
        plt.plot(
            range(1, len(accuracies) + 1), accuracies, label=f"{model_name}"
        )
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Model Comparison: Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    """Plots training and validation loss and accuracy."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(labels, predictions, classes, model_name):
    """Plots the confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def print_classification_report(labels, predictions, classes, model_name):
    """Prints the classification report."""
    print(f"\n📊 Classification Report for {model_name}:\n")
    print(classification_report(labels, predictions, target_names=classes))

# === Visualization Functions (Updated for Clean Images) ===
def denormalize_image(tensor):
    """Denormalize image tensor using ImageNet stats"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def display_predictions(images, true_labels, predicted_labels, classes, num_display=5):
    """Displays multiple sample images with true and predicted labels in a single horizontal row."""
    plt.figure(figsize=(15, 3))  # Adjust height as needed

    for i in range(min(num_display, len(images))):
        # Handle numpy arrays or tensors
        if isinstance(images[i], np.ndarray):
            img = torch.from_numpy(images[i])
        else:
            img = images[i]

        # Denormalize and convert to HWC format
        img = denormalize_image(img)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)  # Ensure valid pixel range

        plt.subplot(1, num_display, i + 1)
        plt.imshow(img)
        plt.title(f"True: {classes[true_labels[i]]}\nPred: {classes[predicted_labels[i]]}", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_dataset_images(dataset, classes, num_display=5):
    """Displays sample images from the dataset in a row."""
    plt.figure(figsize=(15, 3 * num_display))
    for i in range(min(num_display, len(dataset))):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)
        image = denormalize_image(image).cpu().squeeze().numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)

        plt.subplot(1, num_display, i + 1)
        plt.imshow(image)
        plt.title(f"Class: {classes[label]}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

"""# Model setup"""

# === Model Setup ===
def get_models(
    use_resnet34=True, use_resnet50=True, use_resnet101=True, use_densenet121=True
):
    """
    Returns a dictionary of models based on user-specified flags.
    """
    available_models = {}
    if use_resnet34:
        available_models["ResNet34 (Quantum)"] = models.resnet34(
            weights=ResNet34_Weights.DEFAULT
        )
        available_models["ResNet34 (Classical)"] = models.resnet34(
            weights=ResNet34_Weights.DEFAULT
        )
    if use_resnet50:
        available_models["ResNet50 (Quantum)"] = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )
        available_models["ResNet50 (Classical)"] = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )
    if use_resnet101:
        available_models["ResNet101 (Quantum)"] = models.resnet101(
            weights=ResNet101_Weights.DEFAULT
        )
        available_models["ResNet101 (Classical)"] = models.resnet101(
            weights=ResNet101_Weights.DEFAULT
        )
    if use_densenet121:
        available_models["DenseNet121 (Quantum)"] = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )
        available_models["DenseNet121 (Classical)"] = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )
    return available_models

"""# Main Function"""

# === Main Function ===
def main():
    """
    Main function to run the training and testing of selected models.
    """

    all_model_accuracies = {}
    all_test_results = {}

    # Select models to run
    backbones = get_models(
        use_resnet34=True, use_resnet50=True, use_resnet101=True, use_densenet121=True
    )


    for model_name, backbone in backbones.items():
        print(f"\n--- Training and Testing {model_name} ---")

        model = (
            HybridQCNN(backbone, num_classes, n_qubits=N_QUBITS).to(device)
            if "Quantum" in model_name
            else ClassicalCNN(backbone, num_classes).to(device)
        )

        (
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            val_preds,
            val_labels,
            val_images
        ) = train_model(model, train_loader, val_loader, num_classes)

        all_model_accuracies[model_name] = val_accs

        plot_metrics(train_losses, val_losses, train_accs, val_accs, model_name)
        plot_confusion_matrix(val_labels, val_preds, full_dataset.classes, f"{model_name} Validation")
        print_classification_report(
            val_labels, val_preds, full_dataset.classes, f"{model_name} Validation"
        )
        display_predictions(val_images, val_labels, val_preds, full_dataset.classes, num_display=5)


        test_preds, test_labels, test_acc, test_images = test_model(model, test_loader, full_dataset.classes)
        plot_confusion_matrix(
            test_labels,test_preds, full_dataset.classes, f"{model_name} Test"
        )
        print_classification_report(
            test_labels, test_preds, full_dataset.classes, f"{model_name} Test"
        )
        all_test_results[model_name] = test_acc
        display_predictions(test_images, test_labels, test_preds, full_dataset.classes, num_display=5)

    plot_model_comparison(all_model_accuracies)

    print("\n--- Test Accuracies ---")
    for model_name, test_acc in all_test_results.items():
        print(f"{model_name}: {test_acc:.2f}%")



if __name__ == "__main__":
    main()
