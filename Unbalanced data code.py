# import torch and related modules
import torch                 # Core PyTorch library for tensors and neural networks
import torch.nn as nn        # Neural network module (layers, activations)
import torch.optim as optim  # Optimizers for training (Adam)
from torch.utils.data import Dataset, DataLoader  # Dataset abstraction and batching utilities

# torchvision modules for image handling and pre-trained models
import torchvision.transforms as transforms  # For image preprocessing and augmentation
import torchvision.models as models          # Pre-trained models like ResNet, EfficientNet

# General-purpose libraries
import pandas as pd        # Dataframe handling for CSVs / tabular data
import numpy as np         # Numerical operations (arrays, bincounts, etc.)
import os                  # File and path management
import random              # Random sampling
import seaborn as sns      # Plotting library for visualizations like confusion matrix
import matplotlib.pyplot as plt  # Plotting library for graphs, images, etc.
from sklearn.metrics import confusion_matrix  # Compute confusion matrix
from PIL import Image      # For opening images
from sklearn.model_selection import train_test_split  # Splitting dataset into train/val/test
from sklearn.preprocessing import LabelEncoder        # Encode string labels as integers
from sklearn.metrics import classification_report
import time                # Timing utilities
from datetime import datetime  # For human-readable timestamps

# --------------------------------------------------------------------------
# 1. DEVICE SETUP
# --------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)  
# Explanation:
# We check if a GPU (CUDA) is available for training. Using a GPU greatly speeds up training
# for CNNs, especially with large datasets like HAM10000. If not, fallback to CPU.

# --------------------------------------------------------------------------
# 2. DATASET LOADING
# --------------------------------------------------------------------------
data_dir = "D:/gadiuationProject/archive"  

# Read metadata CSV that contains image IDs and labels
df = pd.read_csv(os.path.join(data_dir, "D:/gadiuationProject/archive/HAM10000_metadata.csv"))

# There are 2 folders for images, split across parts 1 and 2
image_dir1 = os.path.join(data_dir, "HAM10000_images_part_1")
image_dir2 = os.path.join(data_dir, "HAM10000_images_part_2")

# Helper function to return the correct image path for a given image_id
def get_image_path(image_id):
    path1 = os.path.join(image_dir1, image_id + ".jpg")
    if os.path.exists(path1):  # If image exists in part 1 folder
        return path1
    return os.path.join(image_dir2, image_id + ".jpg")  # Otherwise check part 2

# Add a column to dataframe with the full file path for each image
df["filepath"] = df["image_id"].apply(get_image_path)

# --------------------------------------------------------------------------
# DUPLICATE CHECKING
# --------------------------------------------------------------------------

# 1. Duplicate image IDs
duplicate_images = df[df.duplicated(subset=["image_id"])]
print("Duplicate image IDs:", len(duplicate_images))

# 2. Duplicate lesions (multiple images for same lesion)
lesion_counts = df["lesion_id"].value_counts()
duplicate_lesions = lesion_counts[lesion_counts > 1]

print("Lesions with multiple images:", len(duplicate_lesions))
print("Example:")
print(duplicate_lesions.head())

# 3. Duplicate file paths
duplicate_paths = df[df.duplicated(subset=["filepath"])]
print("Duplicate filepaths:", len(duplicate_paths))

# Encode string labels (diagnosis) as integers
le = LabelEncoder()
df["label"] = le.fit_transform(df["dx"])

# Save class names and number of classes
class_names = le.classes_
num_classes = len(class_names)
print(class_names)  # Print all skin lesion classes

# ------------------------------------------------------------
# CLASS NAME + COUNT
# ------------------------------------------------------------

for label in sorted(df["label"].unique()):
    class_name = le.inverse_transform([label])[0]
    count = len(df[df["label"] == label])
    
    print(f"Class {label} ({class_name}) → {count} images")

# --------------------------------------------------------------------------
# 3. TRAIN / VAL / TEST SPLIT
# --------------------------------------------------------------------------
# Split the data into train (80%) and test (20%) while preserving class distribution (stratify)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Further split train into train (80% of train) and validation (20% of train) 
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=42
)

# Explanation:
# Stratified splitting ensures that the proportion of each class is maintained across train, val, test.
# This is crucial for imbalanced datasets like HAM10000.

# --------------------------------------------------------------------------
# 4. CUSTOM DATASET CLASS
# --------------------------------------------------------------------------
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe       # Dataframe containing file paths and labels
        self.transform = transform  # Transformations / augmentations

    def __len__(self):
        return len(self.df)       # Total number of samples

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]  # Get image file path
        label = self.df.iloc[idx]["label"]        # Get integer label

        image = Image.open(img_path).convert("RGB")  # Open image and convert to 3 channels

        if self.transform:
            image = self.transform(image)  # Apply preprocessing / augmentation

        return image, label  # Return image tensor and label

# --------------------------------------------------------------------------
# 5. IMAGE TRANSFORMATIONS
# --------------------------------------------------------------------------

# Training transforms (no augmentation, only preprocessing)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to CNN input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# Validation/Test transforms (same preprocessing)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------------------------------
# 6. DATASET AND DATALOADER CREATION
# --------------------------------------------------------------------------
train_dataset = HAM10000Dataset(train_df, transform=train_transform)
val_dataset = HAM10000Dataset(val_df, transform=val_transform)
test_dataset = HAM10000Dataset(test_df, transform=val_transform)

# DataLoader handles batching, shuffling, and parallel loading
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Explanation:
# - shuffle=True for training so model sees data in random order each epoch (improves generalization)
# - batch_size=64 chosen for GPU memory balance

# --------------------------------------------------------------------------
# 7. CUSTOM CNN MODEL
# --------------------------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3 input channels (RGB), 64 filters
            nn.ReLU(),                                   # Activation
            nn.MaxPool2d(2),                             # Downsample by 2x

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
                        
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global average pooling to reduce spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),       # Dropout for regularization to reduce overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# Instantiate model and move to device (GPU or CPU)
model = CustomCNN(num_classes).to(device)

# --------------------------------------------------------------------------
# 8. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer with small learning rate

# ============================================================
# 9. HISTORY LISTS (INITIALIZE FIRST)
# ============================================================

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

# ============================================================
# 10. LOAD CHECKPOINT (IF EXISTS)
# ============================================================

checkpoint_path = "customcnn_checkpoint.pth"

if os.path.exists(checkpoint_path): 
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_acc_history = checkpoint['train_acc_history']
    val_acc_history = checkpoint['val_acc_history']
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']

    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Model will be trained.")

# --------------------------------------------------------------------------
# 11. TRAINING FUNCTION
# --------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs):

    # Record start time for total training
    training_start_time = time.time()
    print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # -------------------------
        # TRAINING LOOP
        # -------------------------
        model.train()  # Set model to training mode (enables dropout, batchnorm)
        train_correct = 0
        train_total = 0
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()        # Reset gradients
            outputs = model(images)      # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()              # Backward pass (compute gradients)
            optimizer.step()             # Update weights

            # Accumulate loss and compute number of correct predictions
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss = running_loss / train_total
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # -------------------------
        # VALIDATION LOOP
        # -------------------------
        model.eval()  # Set model to evaluation mode (disables dropout, batchnorm)
        val_correct = 0
        val_total = 0
        val_running_loss = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_running_loss / val_total
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Time: {epoch_duration:.2f} sec")

    # Print total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print("-" * 60)
    print("Training finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total Training Time: {total_training_time/60:.2f} minutes")

# ============================================================
# 12. TRAIN ONLY IF CHECKPOINT DOES NOT EXIST
# ============================================================

if not os.path.exists(checkpoint_path):
    
    train_model(model, train_loader, val_loader, epochs=60)
        
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
    'train_loss_history': train_loss_history,
    'val_loss_history': val_loss_history,
    }, checkpoint_path)
    
    print("Model saved successfully.")
else:
    print("Training skipped. Using saved model.")

# --------------------------------------------------------------------------
# 13. CLASS DISTRIBUTION
# --------------------------------------------------------------------------
labels_train = [label for _, label in train_dataset]
labels_val = [label for _, label in val_dataset]
labels_test = [label for _, label in test_dataset]

print("Train class distribution:", np.bincount(labels_train))
print("Validation class distribution:", np.bincount(labels_val))
print("Test class distribution:", np.bincount(labels_test))

# Plot training class distribution
counts = pd.Series([label for _, label in train_dataset]).value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Training Class Distribution")
plt.show()

# --------------------------------------------------------------------------
# 14. TEST ACCURACY
# --------------------------------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Test Accuracy:", 100 * correct / total)

# --------------------------------------------------------------------------
# 15. PLOTTING ACCURACY AND LOSS GRAPHS
# --------------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc_history, label="Training Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_loss_history, label="Training Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------------------------------------------
# 16. CONFUSION MATRIX
# --------------------------------------------------------------------------
y_true_cnn = []
y_pred_cnn = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true_cnn.extend(labels.cpu().numpy())
        y_pred_cnn.extend(predicted.cpu().numpy())

y_true_cnn = np.array(y_true_cnn)
y_pred_cnn = np.array(y_pred_cnn)

cm = confusion_matrix(y_true_cnn, y_pred_cnn)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()



# --------------------------------------------------------------------------
# DENORMALIZATION FUNCTION (to correctly display normalized images)
# --------------------------------------------------------------------------

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    img = img * std + mean   # reverse normalization
    img = torch.clamp(img, 0, 1)  # keep values between 0 and 1
    
    return img


# --------------------------------------------------------------------------
# SHOW SAMPLE PREDICTIONS (6 images in 2x3 grid)
# --------------------------------------------------------------------------

model.eval()

images_shown = 6
indices = random.sample(range(len(test_dataset)), images_shown)

fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes = axes.flatten()

for ax, i in zip(axes, indices):

    img, label = test_dataset[i]

    # Prepare image for model
    img_input = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_input)
        _, pred = torch.max(output, 1)

    # Denormalize for display
    img_display = denormalize(img)

    # Convert from (C,H,W) → (H,W,C)
    img_display = img_display.permute(1,2,0)

    ax.imshow(img_display)
    ax.set_title(
        f"True: {class_names[label]}\nPred: {class_names[pred.item()]}",
        fontsize=14
    )
    ax.axis("off")


# Hide unused subplot spaces
for j in range(images_shown, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# EXPERIMENT 2: EFFICIENTNET (TRANSFER LEARNING - FROZEN)
# ============================================================

print("\nStarting EfficientNet Experiment...")
print("-" * 60)

# ------------------------------------------------------------
# 1. Load Pretrained EfficientNet-B0
# ------------------------------------------------------------

efficientnet_model = models.efficientnet_b0(pretrained=True)

# Move to device
efficientnet_model = efficientnet_model.to(device)


# ------------------------------------------------------------
# Freeze most EfficientNet layers
# ------------------------------------------------------------

for param in efficientnet_model.parameters():
    param.requires_grad = False

# ------------------------------------------------------------
# Unfreeze last EfficientNet block
# ------------------------------------------------------------

for param in efficientnet_model.features[-1].parameters():
    param.requires_grad = True

# Classifier must also learn
for param in efficientnet_model.classifier.parameters():
    param.requires_grad = True

# ------------------------------------------------------------
# 3. Replace Final Classifier Layer
# ------------------------------------------------------------

num_features = efficientnet_model.classifier[1].in_features

efficientnet_model.classifier[1] = nn.Linear(num_features, num_classes)

efficientnet_model = efficientnet_model.to(device)


# ------------------------------------------------------------
# 4. Define Loss and Optimizer
# (Only train classifier parameters)
# ------------------------------------------------------------

efficientnet_criterion = nn.CrossEntropyLoss()

efficientnet_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, efficientnet_model.parameters()),
    lr=3e-5
)


# ------------------------------------------------------------
# 5. History Lists
# ------------------------------------------------------------

efficientnet_train_acc_history = []
efficientnet_val_acc_history = []
efficientnet_train_loss_history = []
efficientnet_val_loss_history = []


# ------------------------------------------------------------
# 6. Training Function for EfficientNet 
# ------------------------------------------------------------

def train_efficientnet(model, train_loader, val_loader, epochs):
    
    # Record start time for total training
    training_start_time = time.time()
    print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # ---- TRAINING ----
        model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            efficientnet_optimizer.zero_grad()
            outputs = model(images)
            loss = efficientnet_criterion(outputs, labels)
            loss.backward()
            efficientnet_optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss = running_loss / train_total

        efficientnet_train_acc_history.append(train_acc)
        efficientnet_train_loss_history.append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = efficientnet_criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_running_loss / val_total

        efficientnet_val_acc_history.append(val_acc)
        efficientnet_val_loss_history.append(val_loss)

        # Calculate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Epoch Time: {epoch_duration:.2f} sec")

    # Print total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print("-" * 60)
    print("Training finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total Training Time: {total_training_time/60:.2f} minutes")

# ------------------------------------------------------------
# 7. Train & Save Checkpoint
# ------------------------------------------------------------

efficientnet_checkpoint_path = "efficientnet_checkpoint.pth"

if os.path.exists(efficientnet_checkpoint_path):
    print("Loading existing EfficientNet checkpoint...")

    checkpoint = torch.load(efficientnet_checkpoint_path, map_location=device)
    
    # Load model weights (this is always safe)
    efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Try to load optimizer weights; ignore if incompatible
    try:
        efficientnet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer loaded successfully.")
    except ValueError:
        print("Optimizer state incompatible with current model. Using fresh optimizer.")
        # No need to reset the optimizer; it already has the correct parameter groups
    
    # Load histories if available
    efficientnet_train_acc_history = checkpoint.get('train_acc_history', [])
    efficientnet_val_acc_history = checkpoint.get('val_acc_history', [])
    efficientnet_train_loss_history = checkpoint.get('train_loss_history', [])
    efficientnet_val_loss_history = checkpoint.get('val_loss_history', [])

    print("EfficientNet loaded successfully.")
else:
    print("No checkpoint found. Starting training from scratch.")


    train_efficientnet(efficientnet_model, train_loader, val_loader, epochs=60)
    
    torch.save({
        'model_state_dict': efficientnet_model.state_dict(),
        'optimizer_state_dict': efficientnet_optimizer.state_dict(),
        'train_acc_history': efficientnet_train_acc_history,
        'val_acc_history': efficientnet_val_acc_history,
        'train_loss_history': efficientnet_train_loss_history,
        'val_loss_history': efficientnet_val_loss_history,
    }, efficientnet_checkpoint_path)
    
    print("EfficientNet model saved successfully.")   
# --------------------------------------------------------------------------
# 8. TEST ACCURACY
# --------------------------------------------------------------------------
efficientnet_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = efficientnet_model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

efficientnet_test_acc = 100 * correct / total
print(f"EfficientNet Test Accuracy: {efficientnet_test_acc:.2f}%")

    
# --------------------------------------------------------------------------
# 9. PLOTTING ACCURACY AND LOSS GRAPHS
# --------------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(efficientnet_train_acc_history, label="Efficientnet Training Accuracy")
plt.plot(efficientnet_val_acc_history, label="Efficientnet Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(efficientnet_train_loss_history, label="Efficientnet Training Loss")
plt.plot(efficientnet_val_loss_history, label="Efficientnet Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------------------------------
y_true_eff = []
y_pred_eff = []

efficientnet_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs =efficientnet_model(images)
        _, predicted = torch.max(outputs, 1)
        y_true_eff.extend(labels.cpu().numpy())
        y_pred_eff.extend(predicted.cpu().numpy())

y_true_eff = np.array(y_true_eff)
y_pred_eff = np.array(y_pred_eff)

cm = confusion_matrix(y_true_eff, y_pred_eff)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("EfficientNet Confusion Matrix")
plt.show()



# ------------------------------------------------------------
# 12. SHOW SAMPLE PREDICTIONS (6 images in 2x3 grid)
# ------------------------------------------------------------

# Denormalization function
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    img = img * std + mean   # reverse normalization
    img = torch.clamp(img, 0, 1)  # keep values between 0 and 1
    
    return img


efficientnet_model.eval()
images_shown = 6

# Random test samples
indices = random.sample(range(len(test_dataset)), images_shown)

fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes = axes.flatten()

for ax, i in zip(axes, indices):

    img, label = test_dataset[i]

    # Prepare input for model
    img_input = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = efficientnet_model(img_input)
        _, pred = torch.max(output, 1)

    # Denormalize image for visualization
    img_display = denormalize(img)

    # Convert from (C,H,W) → (H,W,C)
    img_display = img_display.permute(1,2,0).cpu().numpy()

    ax.imshow(img_display)

    ax.set_title(
        f"True: {class_names[label]}\nPred: {class_names[pred.item()]}",
        fontsize=14
    )

    ax.axis("off")


# Hide empty plots if any
for j in range(images_shown, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()




cnn_report = classification_report(
    y_true_cnn,   # make sure you saved them separately!
    y_pred_cnn,
    target_names=class_names,
    output_dict=True
)


eff_report = classification_report(
    y_true_eff,
    y_pred_eff,
    target_names=class_names,
    output_dict=True
)



comparison_data = []

for cls in class_names:
    comparison_data.append({
        "Class": cls,

        "CNN Precision": cnn_report[cls]["precision"],
        "CNN Recall": cnn_report[cls]["recall"],
        "CNN F1": cnn_report[cls]["f1-score"],

        "Eff Precision": eff_report[cls]["precision"],
        "Eff Recall": eff_report[cls]["recall"],
        "Eff F1": eff_report[cls]["f1-score"],
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n=== CNN vs EfficientNet Comparison ===")
print(comparison_df)



summary = pd.DataFrame({
    "Metric": ["Accuracy", "Macro F1", "Weighted F1"],

    "CNN": [
        cnn_report["accuracy"],
        cnn_report["macro avg"]["f1-score"],
        cnn_report["weighted avg"]["f1-score"]
    ],

    "EfficientNet": [
        eff_report["accuracy"],
        eff_report["macro avg"]["f1-score"],
        eff_report["weighted avg"]["f1-score"]
    ]
})

print("\n=== Overall Comparison ===")
print(summary)




comparison_df.set_index("Class")[["CNN F1", "Eff F1"]].plot(kind="bar", figsize=(10,5))
plt.title("F1-Score Comparison per Class")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.show()



# ============================================================
# ROC-AUC EVALUATION (SEPARATE FROM MAIN RESULTS)
# ============================================================

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

print("\nStarting ROC-AUC Evaluation...")
print("-" * 60)

# ------------------------------------------------------------
# 1. CNN ROC-AUC
# ------------------------------------------------------------
y_true_cnn = []
y_probs_cnn = []

model.eval()

with torch.no_grad():
    for imgs, labels in test_loader:

        imgs = imgs.to(device)

        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)

        y_probs_cnn.extend(probs.cpu().numpy())
        y_true_cnn.extend(labels.numpy())

y_true_cnn = np.array(y_true_cnn)
y_probs_cnn = np.array(y_probs_cnn)

# Binarize labels
y_true_bin_cnn = label_binarize(y_true_cnn, classes=range(num_classes))

# AUC scores
cnn_auc_macro = roc_auc_score(
    y_true_bin_cnn,
    y_probs_cnn,
    multi_class='ovr',
    average='macro'
)

cnn_auc_weighted = roc_auc_score(
    y_true_bin_cnn,
    y_probs_cnn,
    multi_class='ovr',
    average='weighted'
)

print("\n=== CNN ROC-AUC ===")
print(f"Macro AUC: {cnn_auc_macro:.4f}")
print(f"Weighted AUC: {cnn_auc_weighted:.4f}")


# ------------------------------------------------------------
# 2. EfficientNet ROC-AUC
# ------------------------------------------------------------
y_true_eff = []
y_probs_eff = []

efficientnet_model.eval()

with torch.no_grad():
    for imgs, labels in test_loader:

        imgs = imgs.to(device)

        outputs = efficientnet_model(imgs)

        probs = torch.softmax(outputs, dim=1)

        y_probs_eff.extend(probs.cpu().numpy())
        y_true_eff.extend(labels.numpy())

y_true_eff = np.array(y_true_eff)
y_probs_eff = np.array(y_probs_eff)

# Binarize labels
y_true_bin_eff = label_binarize(y_true_eff, classes=range(num_classes))

# AUC scores
eff_auc_macro = roc_auc_score(
    y_true_bin_eff,
    y_probs_eff,
    multi_class='ovr',
    average='macro'
)

eff_auc_weighted = roc_auc_score(
    y_true_bin_eff,
    y_probs_eff,
    multi_class='ovr',
    average='weighted'
)

print("\n=== EfficientNet ROC-AUC ===")
print(f"Macro AUC: {eff_auc_macro:.4f}")
print(f"Weighted AUC: {eff_auc_weighted:.4f}")


# ------------------------------------------------------------
# 3. COMPARISON SUMMARY
# ------------------------------------------------------------
print("\n=== ROC-AUC COMPARISON ===")
print(f"CNN        → Macro: {cnn_auc_macro:.4f} | Weighted: {cnn_auc_weighted:.4f}")
print(f"EfficientNet → Macro: {eff_auc_macro:.4f} | Weighted: {eff_auc_weighted:.4f}")


# ------------------------------------------------------------
# 4. ROC CURVES (PER CLASS)
# ------------------------------------------------------------

# ---- CNN ----
plt.figure(figsize=(10,7))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin_cnn[:, i], y_probs_cnn[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]}")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CNN Unbalanced ROC Curves")
plt.legend()
plt.show()


# ---- EfficientNet ----
plt.figure(figsize=(10,7))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin_eff[:, i], y_probs_eff[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]}")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("EfficientNet Unbalanced ROC Curves")
plt.legend()
plt.show()



# ============================================================
# FINAL MODEL COMPARISON TABLE (CNN vs EfficientNet)
# ============================================================
# -----------------------------
# Create final comparison table
# -----------------------------
final_comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro F1-score",
        "Weighted F1-score",
        "Macro ROC-AUC",
        "Weighted ROC-AUC"
    ],

    "CNN": [
        round(cnn_report["accuracy"], 4),
        round(cnn_report["macro avg"]["f1-score"], 4),
        round(cnn_report["weighted avg"]["f1-score"], 4),
        round(cnn_auc_macro, 4),
        round(cnn_auc_weighted, 4)
    ],

    "EfficientNet": [
        round(eff_report["accuracy"], 4),
        round(eff_report["macro avg"]["f1-score"], 4),
        round(eff_report["weighted avg"]["f1-score"], 4),
        round(eff_auc_macro, 4),
        round(eff_auc_weighted, 4)
    ]
})

print("\n=== FINAL MODEL COMPARISON ===")
print(final_comparison)

final_comparison.set_index("Metric").plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("CNN vs EfficientNet Unbalanced Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend()
plt.show()