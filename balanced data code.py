# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# To address the severe class imbalance in the HAM10000 dataset, a hybrid resampling strategy was applied.
# Majority classes (NV, MEL, BKL) were randomly downsampled to 514 samples each. 
# Minority classes (AKIEC, DF, VASC) were upsampled to the same size using data augmentation
# techniques including rotation (20°), horizontal/vertical flipping, zooming (scale factor 1.1), 
# and brightness adjustment (0.2). This produced a balanced dataset containing 514 samples per class.

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
# 3. DATA BALANCING
# --------------------------------------------------------------------------

target_size = 1000

majority_classes = ['nv', 'mel', 'bkl']
minority_classes = ['akiec', 'df', 'vasc', 'bcc']

balanced_dataframes = []

# -----------------------------
# Downsample majority classes
# -----------------------------
for cls in majority_classes:

    class_df = df[df["dx"] == cls]

    downsampled = class_df.sample(
        n=target_size,
        random_state=42
    )

    balanced_dataframes.append(downsampled)


# -----------------------------
# Upsample minority classes
# -----------------------------
for cls in minority_classes:

    class_df = df[df["dx"] == cls]

    current_size = len(class_df)
    needed = target_size - current_size

    upsampled = class_df.sample(
        n=needed,
        replace=True,
        random_state=42
    )

    combined = pd.concat([class_df, upsampled])

    balanced_dataframes.append(combined)




# -----------------------------
# Final balanced dataset
# -----------------------------
balanced_df = pd.concat(balanced_dataframes).reset_index(drop=True)


print("\nBalanced dataset distribution:")
print(balanced_df["dx"].value_counts())

# -----------------------------
# Shuffle dataset
# -----------------------------
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------------------------------------------------------------
# SHOW SAMPLES PER CLASS AFTER BALANCING (WITH CLASS NAMES)
# --------------------------------------------------------------------------
samples_per_class = 5
classes = sorted(balanced_df["dx"].unique())

fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(15,10))

for row_idx, cls in enumerate(classes):
    cls_df = balanced_df[balanced_df["dx"] == cls]
    sampled = cls_df.sample(samples_per_class, random_state=42)

    for col_idx, (_, sample) in enumerate(sampled.iterrows()):
        img = Image.open(sample["filepath"]).convert("RGB")
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis("off")
        axes[row_idx, col_idx].set_title(cls.upper(), fontsize=10)  # Class name on top

plt.suptitle("Sample Images Per Class After Balancing", fontsize=16)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 4. TRAIN / VAL / TEST SPLIT AND LABEL ENCODING
# --------------------------------------------------------------------------

# Encode string labels to integers
le = LabelEncoder()
balanced_df["label"] = le.fit_transform(balanced_df["dx"])
class_names = le.classes_.tolist()
num_classes = len(class_names)

# Split into train (80%) and temp (20%)
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.2,
    stratify=balanced_df["label"],
    random_state=42
)

# Split temp into validation and test (each 10% of total)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --------------------------------------------------------------------------
# 5. CUSTOM DATASET CLASS
# --------------------------------------------------------------------------
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]
        label = self.df.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------------------------------------------------------------
# 6. IMAGE TRANSFORMATIONS
# --------------------------------------------------------------------------
train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(25),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1),
        scale=(0.9,1.1)
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# --------------------------------------------------------------------------
# VGG16-SPECIFIC TRANSFORM
# --------------------------------------------------------------------------

vgg_train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# --------------------------------------------------------------------------
# VISUALIZE AUGMENTATION EFFECTS
# --------------------------------------------------------------------------
sample_image_path = balanced_df.iloc[0]["filepath"]
sample_image = Image.open(sample_image_path).convert("RGB")

num_augmented = 8

plt.figure(figsize=(12,6))

for i in range(num_augmented):
    augmented = train_transform(sample_image)

    # Remove normalization for visualization
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    augmented = augmented * std + mean
    augmented = torch.clamp(augmented,0,1)

    augmented = augmented.permute(1,2,0).numpy()

    plt.subplot(2,4,i+1)
    plt.imshow(augmented)
    plt.axis("off")
    plt.title("Augmented")

plt.suptitle("Examples of Data Augmentation")
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------------
# 7. DATASET AND DATALOADER
# --------------------------------------------------------------------------
train_dataset = HAM10000Dataset(train_df, transform=train_transform)
val_dataset = HAM10000Dataset(val_df, transform=val_transform)
test_dataset = HAM10000Dataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------------------------------------------------------
# VGG16 DATALOADER
# --------------------------------------------------------------------------

vgg_train_dataset = HAM10000Dataset(
    train_df,
    transform=vgg_train_transform
)

vgg_train_loader = DataLoader(
    vgg_train_dataset,
    batch_size=64,
    shuffle=True
)

# --------------------------------------------------------------------------
# 8. CUSTOM CNN MODEL
# --------------------------------------------------------------------------
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes=num_classes).to(device)

# --------------------------------------------------------------------------
# 9. LOSS FUNCTION AND OPTIMIZER
# --------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# --------------------------------------------------------------------------
# 10. HISTORY LISTS
# --------------------------------------------------------------------------
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

# --------------------------------------------------------------------------
# 11. CHECKPOINT SETUP
# --------------------------------------------------------------------------
checkpoint_path = "up_down_sample_customcnn_checkpoint.pth"

#Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_acc_history = checkpoint['train_acc_history']
    val_acc_history = checkpoint['val_acc_history']
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']
    print("Checkpoint loaded. Model will resume training or can be evaluated directly.")
else:
    print("No checkpoint found. Training from scratch.")

# --------------------------------------------------------------------------
# 12. TRAINING FUNCTION WITH CHECKPOINT SAVING
# --------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs):
    start_time = time.time()
    print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-"*60)

    for epoch in range(epochs):
        epoch_start = time.time()
        # Train
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, pred = torch.max(outputs,1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
        train_acc = 100*correct/total
        train_loss = running_loss/total
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*labels.size(0)
                _, pred = torch.max(outputs,1)
                val_correct += (pred==labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100*val_correct/val_total
        val_loss = val_loss/val_total
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        # Save checkpoint after each epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }, checkpoint_path)

        epoch_end = time.time()
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Time: {epoch_end-epoch_start:.2f}s")

    print("-"*60)
    print("Training finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total training time: {(time.time()-start_time)/60:.2f} minutes")

# --------------------------------------------------------------------------
# 13. TRAIN THE MODEL (if not already trained)
# --------------------------------------------------------------------------
if not os.path.exists(checkpoint_path):
    train_model(model, train_loader, val_loader, epochs=100)
else:
    print("Model already trained. Loaded from checkpoint.")

# --------------------------------------------------------------------------
# 14. TEST ACCURACY
# --------------------------------------------------------------------------
model.eval()
correct = 0
total = 0
y_true_cnn = []
y_pred_cnn = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred = torch.max(outputs,1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
        y_true_cnn.extend(labels.cpu().numpy())
        y_pred_cnn.extend(pred.cpu().numpy())

print("CNN Test Accuracy:", 100*correct/total)

# --------------------------------------------------------------------------
# 15. CONFUSION MATRIX
# --------------------------------------------------------------------------
cm = confusion_matrix(y_true_cnn, y_pred_cnn)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN Confusion Matrix")
plt.show()

# --------------------------------------------------------------------------
# 14. PLOT TRAIN/VAL ACCURACY AND LOSS
# --------------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Val Accuracy")
plt.title("CNN Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.title("CNN Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------------------------------------------
# 16. SAMPLE PREDICTIONS VISUALIZATION
# --------------------------------------------------------------------------
def denormalize(img):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img = img*std + mean
    img = torch.clamp(img,0,1)
    return img

model.eval()
num_samples = 9
indices = random.sample(range(len(test_dataset)), num_samples)

fig, axes = plt.subplots(3,3,figsize=(12,10))
axes = axes.flatten()

for ax, i in zip(axes, indices):
    img, label = test_dataset[i]
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_input)
        _, pred = torch.max(output,1)
    img_display = denormalize(img).permute(1,2,0)
    ax.imshow(img_display)
    ax.set_title(f"True: {class_names[label]}\nPred: {class_names[pred.item()]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

cnn_model = model

print("\n***************************")

# --------------------------------------------------------------------------
# 17. EFFICIENTNET MODEL
# --------------------------------------------------------------------------

# Load pretrained EfficientNet-B0
efficientnet_model = models.efficientnet_b0(weights="DEFAULT")

# ------------------------------------------------------------
# Freeze most EfficientNet layers
# ------------------------------------------------------------

for param in efficientnet_model.features.parameters():
    param.requires_grad = False

# ------------------------------------------------------------
# Unfreeze LAST TWO EfficientNet blocks
# ------------------------------------------------------------

for param in efficientnet_model.features[-1].parameters():
    param.requires_grad = True

for param in efficientnet_model.features[-2].parameters():
    param.requires_grad = True

# ------------------------------------------------------------
# Replace classifier
# ------------------------------------------------------------

in_features = efficientnet_model.classifier[1].in_features

efficientnet_model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)

# Move model to device
efficientnet_model = efficientnet_model.to(device)

print("EfficientNet-B0 loaded with partial fine-tuning.")

# --------------------------------------------------------------------------
# 18. LOSS FUNCTION AND OPTIMIZER (EFFICIENTNET)
# --------------------------------------------------------------------------

efficientnet_criterion = nn.CrossEntropyLoss()

efficientnet_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, efficientnet_model.parameters()),
    lr=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    efficientnet_optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# --------------------------------------------------------------------------
# 19. HISTORY LISTS
# --------------------------------------------------------------------------

eff_train_acc_history = []
eff_val_acc_history = []
eff_train_loss_history = []
eff_val_loss_history = []

# --------------------------------------------------------------------------
# 20. CHECKPOINT
# --------------------------------------------------------------------------

eff_checkpoint_path = "Balanced_efficientnet_checkpoint.pth"

if os.path.exists(eff_checkpoint_path):
    checkpoint = torch.load(eff_checkpoint_path, map_location=device)

    efficientnet_model.load_state_dict(checkpoint['model_state_dict'])
    efficientnet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    eff_train_acc_history = checkpoint['train_acc_history']
    eff_val_acc_history = checkpoint['val_acc_history']
    eff_train_loss_history = checkpoint['train_loss_history']
    eff_val_loss_history = checkpoint['val_loss_history']

    print("EfficientNet checkpoint loaded.")
else:
    print("No EfficientNet checkpoint found.")

# --------------------------------------------------------------------------
# 21. TRAIN EFFICIENTNET
# --------------------------------------------------------------------------

def train_efficientnet(model, train_loader, val_loader, epochs):

    start_time = time.time()

    print("EfficientNet Training started at:",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-"*60)

    for epoch in range(epochs):

        epoch_start = time.time()

        # TRAIN
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:

            imgs, labels = imgs.to(device), labels.to(device)

            efficientnet_optimizer.zero_grad()

            outputs = model(imgs)

            loss = efficientnet_criterion(outputs, labels)

            loss.backward()

            efficientnet_optimizer.step()

            running_loss += loss.item()*labels.size(0)

            _, pred = torch.max(outputs,1)

            correct += (pred==labels).sum().item()

            total += labels.size(0)

        train_acc = 100*correct/total
        train_loss = running_loss/total

        eff_train_acc_history.append(train_acc)
        eff_train_loss_history.append(train_loss)

        # VALIDATION
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for imgs, labels in val_loader:

                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)

                loss = efficientnet_criterion(outputs, labels)

                val_loss += loss.item()*labels.size(0)

                _, pred = torch.max(outputs,1)

                val_correct += (pred==labels).sum().item()

                val_total += labels.size(0)

        val_acc = 100*val_correct/val_total
        val_loss = val_loss/val_total

        eff_val_acc_history.append(val_acc)
        eff_val_loss_history.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': efficientnet_optimizer.state_dict(),
            'train_acc_history': eff_train_acc_history,
            'val_acc_history': eff_val_acc_history,
            'train_loss_history': eff_train_loss_history,
            'val_loss_history': eff_val_loss_history,
        }, eff_checkpoint_path)

        epoch_end = time.time()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Time: {epoch_end-epoch_start:.2f}s")

    print("-"*60)

    print("EfficientNet Training finished at:",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"Total training time: {(time.time()-start_time)/60:.2f} minutes")

# --------------------------------------------------------------------------
# 22. RUN TRAINING
# --------------------------------------------------------------------------

if not os.path.exists(eff_checkpoint_path):

    train_efficientnet(
        efficientnet_model,
        train_loader,
        val_loader,
        epochs=50
    )

else:

    print("EfficientNet already trained.")

# --------------------------------------------------------------------------
# 23. EFFICIENTNET TEST ACCURACY
# --------------------------------------------------------------------------

efficientnet_model.eval()

correct = 0
total = 0

y_true_eff = []
y_pred_eff = []

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs, labels = imgs.to(device), labels.to(device)

        outputs = efficientnet_model(imgs)

        _, pred = torch.max(outputs,1)

        correct += (pred==labels).sum().item()

        total += labels.size(0)

        y_true_eff.extend(labels.cpu().numpy())
        y_pred_eff.extend(pred.cpu().numpy())

print("EfficientNet Test Accuracy:", 100*correct/total)

plt.figure(figsize=(8,5))
plt.plot(eff_train_acc_history, label="Train Accuracy")
plt.plot(eff_val_acc_history, label="Val Accuracy")
plt.title("EfficientNet Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()


plt.figure(figsize=(8,5))
plt.plot(eff_train_loss_history, label="Train Loss")
plt.plot(eff_val_loss_history, label="Val Loss")
plt.title("EfficientNet Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

cm = confusion_matrix(y_true_eff, y_pred_eff)

plt.figure(figsize=(10,8))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("EfficientNet Confusion Matrix")

plt.show()

def denormalize(img):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img = img*std + mean
    img = torch.clamp(img,0,1)
    return img

model = efficientnet_model
num_samples = 9
indices = random.sample(range(len(test_dataset)), num_samples)

fig, axes = plt.subplots(3,3,figsize=(12,10))
axes = axes.flatten()

for ax, i in zip(axes, indices):
    img, label = test_dataset[i]
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_input)
        _, pred = torch.max(output,1)
    img_display = denormalize(img).permute(1,2,0)
    ax.imshow(img_display)
    ax.set_title(f"True: {class_names[label]}\nPred: {class_names[pred.item()]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\n***************************")

# --------------------------------------------------------------------------
# RESNET50 MODEL
# --------------------------------------------------------------------------

print("\nLoading ResNet50 model...")

# Load pretrained ResNet50
resnet_model = models.resnet50(weights="DEFAULT")

# ------------------------------------------------------------------
# Freeze all layers
# ------------------------------------------------------------------

# Freeze all layers first
for param in resnet_model.parameters():
    param.requires_grad = False

# Unfreeze ONLY last ResNet block
for param in resnet_model.layer4.parameters():
    param.requires_grad = True

# Unfreeze classifier
for param in resnet_model.fc.parameters():
    param.requires_grad = True

# ------------------------------------------------------------------
# Replace final fully connected layer
# ------------------------------------------------------------------

in_features = resnet_model.fc.in_features

# ------------------------------------------------------------------
# Improved ResNet50 classifier
# ------------------------------------------------------------------

in_features = resnet_model.fc.in_features

resnet_model.fc = nn.Sequential(

    nn.Linear(in_features, 1024),

    nn.BatchNorm1d(1024),

    nn.ReLU(),

    nn.Dropout(0.4),

    nn.Linear(1024, 512),

    nn.BatchNorm1d(512),

    nn.ReLU(),

    nn.Dropout(0.3),

    nn.Linear(512, num_classes)
)

# Move model to GPU/CPU
resnet_model = resnet_model.to(device)

print("ResNet50 loaded successfully.")
# --------------------------------------------------------------------------
# RESNET50 CHECKPOINT
# --------------------------------------------------------------------------

resnet_checkpoint_path = "best_resnet50.pth"

if os.path.exists(resnet_checkpoint_path):

    print("Loading saved ResNet50 checkpoint...")

    checkpoint = torch.load(
        resnet_checkpoint_path,
        map_location=device
    )

    resnet_model.load_state_dict(
        checkpoint['model_state_dict']
    )

    res_train_acc_history = checkpoint['train_acc']
    res_val_acc_history = checkpoint['val_acc']

    res_train_loss_history = checkpoint['train_loss']
    res_val_loss_history = checkpoint['val_loss']

    print("ResNet50 checkpoint loaded successfully.")

else:

    print("No ResNet50 checkpoint found.")

# --------------------------------------------------------------------------
# RESNET50 TRAINING SETUP
# --------------------------------------------------------------------------

resnet_criterion = nn.CrossEntropyLoss()

resnet_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, resnet_model.parameters()),
    lr=3e-5,
    weight_decay=5e-5
)

# Learning rate scheduler
resnet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    resnet_optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# --------------------------------------------------------------------------
# RESNET50 HISTORY TRACKING
# --------------------------------------------------------------------------

res_train_acc_history = []
res_val_acc_history = []

res_train_loss_history = []
res_val_loss_history = []

# --------------------------------------------------------------------------
# RESNET50 TRAINING FUNCTION
# --------------------------------------------------------------------------
def train_resnet(model, train_loader, val_loader, epochs=50):

    print("\nResNet50 Training started...")
    print("-" * 60)

    start_time = time.time()
    
    # ------------------------------------------------
    # EARLY STOPPING VARIABLES
    # ------------------------------------------------
    
    best_val_loss = float('inf')
    patience = 7
    counter = 0

    for epoch in range(epochs):

        # ============================================================
        # TRAINING
        # ============================================================

        model.train()

        train_correct = 0
        train_total = 0
        train_loss = 0

        epoch_start = time.time()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = resnet_criterion(outputs, labels)

            # Backward pass
            resnet_optimizer.zero_grad()
            loss.backward()
            resnet_optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / train_total

        # ============================================================
        # VALIDATION
        # ============================================================

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = resnet_criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / val_total
        
        # ------------------------------------------------
        # EARLY STOPPING CHECK
        # ------------------------------------------------
        
        if val_loss < best_val_loss:
        
            best_val_loss = val_loss
            counter = 0
        
            # Save best model
            torch.save({

                'model_state_dict': model.state_dict(),
            
                'train_acc': res_train_acc_history,
                'val_acc': res_val_acc_history,
            
                'train_loss': res_train_loss_history,
                'val_loss': res_val_loss_history
            
            }, 'best_resnet50.pth')
        
        else:
            counter += 1
        
        # Stop training if no improvement
        if counter >= patience:
        
            print(f"\\nEarly stopping triggered at epoch {epoch+1}")
            break

        # Scheduler update
        resnet_scheduler.step(val_loss)

        # Save history
        res_train_acc_history.append(train_acc)
        res_val_acc_history.append(val_acc)

        res_train_loss_history.append(train_loss)
        res_val_loss_history.append(val_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    print("-" * 60)
    print(f"Total training time: {total_time/60:.2f} minutes")

    return model

# --------------------------------------------------------------------------
# TRAIN RESNET50
# --------------------------------------------------------------------------

if not os.path.exists(resnet_checkpoint_path):

    trained_resnet = train_resnet(
        resnet_model,
        train_loader,
        val_loader,
        epochs=50
    )

else:

    print("ResNet50 already trained.")

# ----------------------------------------------    
#Load best saved model
# ----------------------------------------------

checkpoint = torch.load(
    resnet_checkpoint_path,
    map_location=device
)

resnet_model.load_state_dict(
    checkpoint['model_state_dict']
)

res_train_acc_history = checkpoint['train_acc']
res_val_acc_history = checkpoint['val_acc']

res_train_loss_history = checkpoint['train_loss']
res_val_loss_history = checkpoint['val_loss']
# --------------------------------------------------------------------------
# RESNET50 TEST EVALUATION
# --------------------------------------------------------------------------

resnet_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet_model(images)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
resnet_accuracy = accuracy_score(all_labels, all_preds)

print(f"\nResNet50 Test Accuracy: {resnet_accuracy * 100:.2f}%")

# --------------------------------------------------------------------------
# RESNET50 CONFUSION MATRIX
# --------------------------------------------------------------------------

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("ResNet50 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()

# --------------------------------------------------------------------------
# RESNET50 ACCURACY CURVE
# --------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(res_train_acc_history, label='Train Accuracy')
plt.plot(res_val_acc_history, label='Val Accuracy')

plt.title('ResNet50 Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.legend()
plt.show()

# --------------------------------------------------------------------------
# RESNET50 LOSS CURVE
# --------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(res_train_loss_history, label='Train Loss')
plt.plot(res_val_loss_history, label='Val Loss')

plt.title('ResNet50 Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()
plt.show()

# ============================================================
# RESNET50 SAMPLE PREDICTIONS
# ============================================================

def denormalize(img):

    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    img = img * std + mean

    img = torch.clamp(img,0,1)

    return img

resnet_model.eval()

num_samples = 9

indices = random.sample(
    range(len(test_dataset)),
    num_samples
)

fig, axes = plt.subplots(3,3, figsize=(12,10))

axes = axes.flatten()

for ax, i in zip(axes, indices):

    img, label = test_dataset[i]

    img_input = img.unsqueeze(0).to(device)

    with torch.no_grad():

        output = resnet_model(img_input)

        _, pred = torch.max(output,1)

    img_display = denormalize(img).permute(1,2,0)

    ax.imshow(img_display)

    ax.set_title(
        f"True: {class_names[label]}\nPred: {class_names[pred.item()]}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

print("\n***************************")

# ============================================================
# VGG16
# ============================================================

print("Loading VGG16 model...")

vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

for param in vgg_model.features[:20].parameters():
    param.requires_grad = False

for param in vgg_model.features[20:].parameters():
    param.requires_grad = True
    
# Replace classifier
in_features = vgg_model.classifier[6].in_features

# Number of classes
num_classes = 7

# Replace classifier
vgg_model.classifier = nn.Sequential(

    nn.Linear(25088, 1024),

    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(1024, 256),

    nn.ReLU(),

    nn.Dropout(0.4),

    nn.Linear(256, num_classes)
)

# Move model to device
vgg_model = vgg_model.to(device)

print("VGG16 loaded successfully.")

# ============================================================
# VGG16 CHECKPOINT
# ============================================================

vgg_checkpoint_path = "best_vgg16.pth"

if os.path.exists(vgg_checkpoint_path):

    print("Loading saved VGG16 checkpoint...")

    checkpoint = torch.load(
        vgg_checkpoint_path,
        map_location=device
    )

    vgg_model.load_state_dict(
        checkpoint['model_state_dict']
    )

    vgg_train_acc = checkpoint['train_acc']
    vgg_val_acc = checkpoint['val_acc']

    vgg_train_loss = checkpoint['train_loss']
    vgg_val_loss = checkpoint['val_loss']

    print("VGG16 checkpoint loaded successfully.")

else:

    print("No VGG16 checkpoint found.")

# ============================================================
# LOSS FUNCTION & OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, vgg_model.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_vgg(model, train_loader, val_loader, epochs=50):

    train_accuracies = []
    val_accuracies = []

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    patience = 5
    counter = 0

    start_time = time.time()

    print("\nVGG16 Training started...")
    print("------------------------------------------------------------")

    for epoch in range(epochs):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        train_correct = 0
        train_total = 0
        train_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)

            train_correct += (predicted == labels).sum().item()

            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / train_total

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs, 1)

                val_correct += (predicted == labels).sum().item()

                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / val_total

        scheduler.step(val_loss)

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            counter = 0

            torch.save({

                'model_state_dict': model.state_dict(),
            
                'train_acc': train_accuracies,
                'val_acc': val_accuracies,
            
                'train_loss': train_losses,
                'val_loss': val_losses
            
            }, 'best_vgg16.pth')

        else:
            counter += 1

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if counter >= patience:

            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        # ====================================================
        # STORE METRICS
        # ====================================================

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        start_time = time.time()

    print("------------------------------------------------------------")

    total_time = time.time() - start_time

    print(f"Total training time: {total_time/60:.2f} minutes")

    return (
        train_accuracies,
        val_accuracies,
        train_losses,
        val_losses
    )

# ============================================================
# TRAIN MODEL
# ============================================================

if not os.path.exists(vgg_checkpoint_path):

    (
        vgg_train_acc,
        vgg_val_acc,
        vgg_train_loss,
        vgg_val_loss
    ) = train_vgg(
        vgg_model,
        vgg_train_loader,
        val_loader,
        epochs=50
    )

else:

    print("VGG16 already trained.")

# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    'best_vgg16.pth',
    map_location=device
)

vgg_model.load_state_dict(
    checkpoint['model_state_dict']
)

vgg_train_acc = checkpoint['train_acc']
vgg_val_acc = checkpoint['val_acc']

vgg_train_loss = checkpoint['train_loss']
vgg_val_loss = checkpoint['val_loss']

# ============================================================
# TEST EVALUATION
# ============================================================

vgg_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = vgg_model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("VGG16 Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()

# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    vgg_train_acc,
    label='Train Accuracy'
)

plt.plot(
    vgg_val_acc,
    label='Val Accuracy'
)

plt.title("VGG16 Accuracy per Epoch")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.legend()

plt.show()

# ============================================================
# LOSS PLOT
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    vgg_train_loss,
    label='Train Loss'
)

plt.plot(
    vgg_val_loss,
    label='Val Loss'
)

plt.title("VGG16 Loss per Epoch")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()

# ============================================================
# VGG16 SAMPLE PREDICTIONS
# ============================================================

vgg_model.eval()

num_samples = 9

indices = random.sample(
    range(len(test_dataset)),
    num_samples
)

fig, axes = plt.subplots(3,3, figsize=(12,10))

axes = axes.flatten()

for ax, i in zip(axes, indices):

    img, label = test_dataset[i]

    img_input = img.unsqueeze(0).to(device)

    with torch.no_grad():

        output = vgg_model(img_input)

        _, pred = torch.max(output,1)

    img_display = denormalize(img).permute(1,2,0)

    ax.imshow(img_display)

    ax.set_title(
        f"True: {class_names[label]}\nPred: {class_names[pred.item()]}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(all_labels, all_preds)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average='macro'
)

weighted_f1 = f1_score(
    all_labels,
    all_preds,
    average='weighted'
)

macro_precision = precision_score(
    all_labels,
    all_preds,
    average='macro'
)

macro_recall = recall_score(
    all_labels,
    all_preds,
    average='macro'
)

print(f"\nVGG16 Test Accuracy: {accuracy*100:.2f}%")

# ============================================================
# FINAL COMPREHENSIVE MODEL COMPARISON
# CNN vs EFFICIENTNET vs RESNET50 vs VGG16
# ============================================================

print("\n" + "="*70)
print("FINAL COMPREHENSIVE MODEL COMPARISON")
print("="*70)

# ============================================================
# FUNCTION TO GET PREDICTIONS + PROBABILITIES
# ============================================================

def get_predictions_and_probs(model, loader):

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

# ============================================================
# GET RESULTS FOR ALL MODELS
# ============================================================

cnn_labels, cnn_preds, cnn_probs = get_predictions_and_probs(
    cnn_model,
    test_loader
)

eff_labels, eff_preds, eff_probs = get_predictions_and_probs(
    efficientnet_model,
    test_loader
)

res_labels, res_preds, res_probs = get_predictions_and_probs(
    resnet_model,
    test_loader
)

vgg_labels, vgg_preds, vgg_probs = get_predictions_and_probs(
    vgg_model,
    test_loader
)

# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

cnn_report = classification_report(
    cnn_labels,
    cnn_preds,
    target_names=class_names,
    output_dict=True
)

eff_report = classification_report(
    eff_labels,
    eff_preds,
    target_names=class_names,
    output_dict=True
)

res_report = classification_report(
    res_labels,
    res_preds,
    target_names=class_names,
    output_dict=True
)

vgg_report = classification_report(
    vgg_labels,
    vgg_preds,
    target_names=class_names,
    output_dict=True
)

# ============================================================
# PER-CLASS COMPARISON TABLE
# ============================================================

comparison_data = []

for cls in class_names:

    comparison_data.append({

        "Class": cls,

        "CNN F1":
            round(cnn_report[cls]["f1-score"], 3),

        "EfficientNet F1":
            round(eff_report[cls]["f1-score"], 3),

        "ResNet50 F1":
            round(res_report[cls]["f1-score"], 3),

        "VGG16 F1":
            round(vgg_report[cls]["f1-score"], 3),
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n=== PER-CLASS COMPARISON ===")
print(comparison_df)

# ============================================================
# OVERALL METRICS TABLE
# ============================================================

summary_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Macro F1",
        "Weighted F1"
    ],

    "CNN": [
        round(cnn_report["accuracy"], 3),
        round(cnn_report["macro avg"]["f1-score"], 3),
        round(cnn_report["weighted avg"]["f1-score"], 3)
    ],

    "EfficientNet": [
        round(eff_report["accuracy"], 3),
        round(eff_report["macro avg"]["f1-score"], 3),
        round(eff_report["weighted avg"]["f1-score"], 3)
    ],

    "ResNet50": [
        round(res_report["accuracy"], 3),
        round(res_report["macro avg"]["f1-score"], 3),
        round(res_report["weighted avg"]["f1-score"], 3)
    ],

    "VGG16": [
        round(vgg_report["accuracy"], 3),
        round(vgg_report["macro avg"]["f1-score"], 3),
        round(vgg_report["weighted avg"]["f1-score"], 3)
    ]
})

print("\n=== OVERALL PERFORMANCE COMPARISON ===")
print(summary_df)

# ============================================================
# BAR PLOT OF F1 SCORES
# ============================================================

comparison_df.set_index("Class").plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Per-Class F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)

plt.show()

# ============================================================
# ROC-AUC EVALUATION
# ============================================================

print("\n=== ROC-AUC EVALUATION ===")

# Binarize labels
y_true_bin = label_binarize(
    cnn_labels,
    classes=range(num_classes)
)

# ------------------------------------------------------------
# CNN
# ------------------------------------------------------------

cnn_auc_macro = roc_auc_score(
    y_true_bin,
    cnn_probs,
    multi_class='ovr',
    average='macro'
)

# ------------------------------------------------------------
# EfficientNet
# ------------------------------------------------------------

eff_auc_macro = roc_auc_score(
    y_true_bin,
    eff_probs,
    multi_class='ovr',
    average='macro'
)

# ------------------------------------------------------------
# ResNet50
# ------------------------------------------------------------

res_auc_macro = roc_auc_score(
    y_true_bin,
    res_probs,
    multi_class='ovr',
    average='macro'
)

# ------------------------------------------------------------
# VGG16
# ------------------------------------------------------------

vgg_auc_macro = roc_auc_score(
    y_true_bin,
    vgg_probs,
    multi_class='ovr',
    average='macro'
)

# ============================================================
# ROC SUMMARY
# ============================================================

print(f"\nCNN ROC-AUC: {cnn_auc_macro:.4f}")
print(f"EfficientNet ROC-AUC: {eff_auc_macro:.4f}")
print(f"ResNet50 ROC-AUC: {res_auc_macro:.4f}")
print(f"VGG16 ROC-AUC: {vgg_auc_macro:.4f}")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

final_comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Macro F1-score",
        "Weighted F1-score",
        "Macro ROC-AUC"
    ],

    "CNN": [
        round(cnn_report["accuracy"], 4),
        round(cnn_report["macro avg"]["f1-score"], 4),
        round(cnn_report["weighted avg"]["f1-score"], 4),
        round(cnn_auc_macro, 4)
    ],

    "EfficientNet": [
        round(eff_report["accuracy"], 4),
        round(eff_report["macro avg"]["f1-score"], 4),
        round(eff_report["weighted avg"]["f1-score"], 4),
        round(eff_auc_macro, 4)
    ],

    "ResNet50": [
        round(res_report["accuracy"], 4),
        round(res_report["macro avg"]["f1-score"], 4),
        round(res_report["weighted avg"]["f1-score"], 4),
        round(res_auc_macro, 4)
    ],

    "VGG16": [
        round(vgg_report["accuracy"], 4),
        round(vgg_report["macro avg"]["f1-score"], 4),
        round(vgg_report["weighted avg"]["f1-score"], 4),
        round(vgg_auc_macro, 4)
    ]
})

print("\n=== FINAL MODEL COMPARISON ===")
print(final_comparison)

# ============================================================
# FINAL BAR PLOT
# ============================================================

final_comparison.set_index("Metric").plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Final Model Performance Comparison")
plt.ylabel("Score")

plt.xticks(rotation=20)

plt.show()

# ============================================================
# ROC CURVES FOR ALL MODELS
# ============================================================

models_dict = {
    "CNN": cnn_probs,
    "EfficientNet": eff_probs,
    "ResNet50": res_probs,
    "VGG16": vgg_probs
}

for model_name, probs in models_dict.items():

    plt.figure(figsize=(10,7))

    for i in range(num_classes):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            probs[:, i]
        )

        plt.plot(
            fpr,
            tpr,
            label=class_names[i]
        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"{model_name} ROC Curves")

    plt.legend()

    plt.show()


# ============================================================
# FINAL MODEL RANKING (BEST → WORST)
# ============================================================

ranking_df = pd.DataFrame({

    "Model": [
        "CNN",
        "EfficientNet",
        "ResNet50",
        "VGG16"
    ],

    "Accuracy": [
        cnn_report["accuracy"],
        eff_report["accuracy"],
        res_report["accuracy"],
        vgg_report["accuracy"]
    ],

    "Macro_F1": [
        cnn_report["macro avg"]["f1-score"],
        eff_report["macro avg"]["f1-score"],
        res_report["macro avg"]["f1-score"],
        vgg_report["macro avg"]["f1-score"]
    ],

    "Weighted_F1": [
        cnn_report["weighted avg"]["f1-score"],
        eff_report["weighted avg"]["f1-score"],
        res_report["weighted avg"]["f1-score"],
        vgg_report["weighted avg"]["f1-score"]
    ],

    "ROC_AUC": [
        cnn_auc_macro,
        eff_auc_macro,
        res_auc_macro,
        vgg_auc_macro
    ]
})

# ============================================================
# CREATE FINAL SCORE
# ============================================================

ranking_df["Final Score"] = (

    ranking_df["Accuracy"] * 0.35 +

    ranking_df["Macro_F1"] * 0.35 +

    ranking_df["Weighted_F1"] * 0.20 +

    ranking_df["ROC_AUC"] * 0.10
)

# ============================================================
# SORT BEST → WORST
# ============================================================

ranking_df = ranking_df.sort_values(
    by="Final Score",
    ascending=False
).reset_index(drop=True)

# Add ranking number
ranking_df.index = ranking_df.index + 1

print("\n" + "="*70)
print("FINAL MODEL RANKING (BEST → WORST)")
print("="*70)

print(ranking_df)
















