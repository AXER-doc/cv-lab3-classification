import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18  # Опционально для fine-tuning

# Аугментация данных
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

import kagglehub

# Download latest version
path = kagglehub.dataset_download("abhinavnayak/catsvdogs-transformed")

print("Path to dataset files:", path)

# Group files into train/cat, train/dog, val/cat, val/dog
# exactly 200 cats and 200 dogs should be in test dataset

from os import listdir, makedirs, rename
from random import randint

cats_left = 1000
cats_test = 200
dogs_left = 1000
dogs_test = 200

for dirname in ("train", "val", "train/cat", "train/dog", "val/cat", "val/dog"):
    makedirs(dirname, exist_ok=True)

for img in listdir(path+"/train_transformed/"):
    filepath = path+"/train_transformed/"+img
    if img[:3] == "cat":
        if randint(1, cats_left) <= cats_test:
            rename(filepath, "val/cat/"+img)
            cats_test -= 1
        else:
            rename(filepath, "train/cat/"+img)
        cats_left -= 1
    elif img[:3] == "dog":
        if randint(1, dogs_left) <= dogs_test:
            rename(filepath, "val/dog/"+img)
            dogs_test -= 1
        else:
            rename(filepath, "train/dog/"+img)
        dogs_left -= 1

# Загрузка данных (замените пути)
train_ds = datasets.ImageFolder('train/', transform=transform_train)
val_ds = datasets.ImageFolder('val/', transform=transform_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Модель (простая или fine-tune ResNet)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Для 224x224
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

nocuda = not torch.cuda.is_available()
if nocuda:
    print("No CUDA available")

model = SimpleCNN() if nocuda else SimpleCNN().cuda()  # Или resnet18(pretrained=True), замените fc
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
for epoch in range(10):
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        if not nocuda:
            imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Валидация (accuracy)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            if not nocuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            out = model(imgs)
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f'Epoch {epoch}: Train Loss {train_loss/len(train_loader):.3f}, Val Acc {100.*correct/total:.2f}%')
