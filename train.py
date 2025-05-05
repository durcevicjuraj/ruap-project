import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Hyperparams
BATCH_SIZE = 32
IMG_SIZE   = 224
LR         = 1e-4
EPOCHS     = 10

TRAIN_DIR  = "datasets/Training"
TEST_DIR   = "datasets/Testing"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4.1 Define transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# 4.2 Create datasets & loaders
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 4.3 Class names
classes = train_ds.classes  # ['glioma_tumor', 'meningioma_tumor', ...]
num_classes = len(classes)
print("Classes:", classes)

# 5.1 Load model
model = models.resnet50(pretrained=True)

# 5.2 Replace the classifier head
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(DEVICE)

# 6.1 Train for one epoch
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

# 6.2 Evaluate on test set
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

def main():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        test_acc   = eval_model(model, test_loader)
        print(f"Epoch {epoch}/{EPOCHS}: Loss={train_loss:.4f}  Test Acc={test_acc:.4f}")

    # 8. Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/resnet50_brain_tumor.pth")
    print("Model saved to saved_models/resnet50_brain_tumor.pth")

if __name__ == "__main__":
    main()

