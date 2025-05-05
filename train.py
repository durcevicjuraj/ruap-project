import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE   = 224
LR         = 1e-4
EPOCHS     = 1  # Adjust epochs as needed

# Data directories
TRAIN_DIR  = "datasets/Training"
TEST_DIR   = "datasets/Testing"

# Device configuration
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Create datasets & loaders
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Class names and count
classes = train_ds.classes
num_classes = len(classes)
print("Classes:", classes)

# Load pretrained ResNet50 and modify final layer
model = models.resnet50(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# Training and evaluation functions
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(loader, desc="Train"):        
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def eval_model(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().tolist())
            truths.extend(labels.tolist())
    return accuracy_score(truths, preds)

# Main training flow with MLflow logging and local MLflow model export
def main():
    mlflow.set_experiment("Brain_Tumor_Classification")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("model", "resnet50")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("lr", LR)
        mlflow.log_param("epochs", EPOCHS)

        # Training loop
        for epoch in range(1, EPOCHS+1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            test_acc   = eval_model(model, test_loader)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)
            print(f"Epoch {epoch}/{EPOCHS}: Loss={train_loss:.4f}  Test Acc={test_acc:.4f}")

        # Register model in MLflow Tracking
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="resnet50_brain_tumor",
            registered_model_name="ResNet50BrainTumor"
        )

        # Export MLflow model locally (creates MLmodel directory)
        local_path = "mlflow_model/resnet50_brain_tumor"
        os.makedirs(local_path, exist_ok=True)
        mlflow.pytorch.save_model(
            pytorch_model=model,
            path=local_path,
            conda_env=None
        )
        print(f"MLflow model saved locally at '{local_path}'")

if __name__ == "__main__":
    main()
