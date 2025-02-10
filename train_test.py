import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import precision_score
from torch.cuda import Event
from sklearn.metrics import recall_score
import random
from LLMPSR import Net
import numpy as np


seed = 3407
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


start_event = Event()
end_event = Event()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0688, 0.0688, 0.0688], std=[0.0926, 0.0926, 0.0926])
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0688, 0.0688, 0.0688], std=[0.0926, 0.0926, 0.0926])
])


train_dataset = ImageFolder(root='data/MeltPool/train', transform=train_transform)
val_dataset = ImageFolder(root='data/MeltPool/val', transform=transform)
test_dataset = ImageFolder(root='data/MeltPool/test', transform=transform)


batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model = model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

milestones = [30, 1000]
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


num_epochs = 1000


best_val_accuracy = 0.0
second_best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.cpu()

    train_accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Train Accuracy: {train_accuracy:.2%}')

    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss = criterion(outputs, labels)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy:.2%}')

    val_precision = precision_score(all_labels, all_predicted, average='weighted')
    print(f'Validation Precision: {val_precision:.2%}')

    recall = recall_score(all_labels, all_predicted, average='weighted')
    print(f'Validation Recall: {recall:.2%}')


    correct = 0
    total = 0
    all_predicted = []
    all_labels = []


    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.2%}')

    val_precision = precision_score(all_labels, all_predicted, average='weighted')
    print(f'Test Precision: {val_precision:.2%}')

    recall = recall_score(all_labels, all_predicted, average='weighted')
    print(f'Test Recall: {recall:.2%}')




