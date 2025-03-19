import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import confusion_matrix
import time
from thop import profile
from thop import clever_format
from PIL import Image
from torch.cuda import Event
import scipy.io

from model.SeConvNet import SeConvNet

# 训练模型
num_epochs = 1
# 定义保存文件名
save_file_name = 'SavedData/training_val_results_loss_acc_T0.mat'
best_model_weight_path = 'pretrainedWeight/best_model_weights_T0.pth'
second_model_weight_path = 'pretrainedWeight/second_model_weights_T0.pth'
# 初始化存储训练结果的字典
training_results = {
    'loss': [],          # 存储训练过程中的 loss0
    'train_accuracy': [], # 存储训练集准确率
    'val_loss': [],      # 存储验证集损失-+
    'val_accuracy': []   # 存储验证集准确率
}

# 创建事件对象
start_event = Event()
end_event = Event()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0688, 0.0688, 0.0688], std=[0.0926, 0.0926, 0.0926])
])

# 加载数据集
train_dataset = ImageFolder(root='data/train', transform=transform)
val_dataset = ImageFolder(root='data/val', transform=transform)
test_dataset = ImageFolder(root='data/test', transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 创建模型并移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeConvNet()
model = model.to(device)

# 查看网络细节 遍历网络的每一层并打印层的名字和类型
for name, layer in model.named_children():
    print(f"Layer Name: {name}, Layer Type: {layer.__class__.__name__}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义学习率衰减策略
milestones = [30]  # 在第 10 个 epoch 时学习率衰减
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  # 学习率衰减为原来的 0.1

# 初始化最好和第二好的验证准确率
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
    scheduler.step()  # 调用学习率衰减策略



    # 验证模型
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
    # 计算 Precision 和 F1-Score
    val_precision = precision_score(all_labels, all_predicted, average='weighted')
    val_f1_score = f1_score(all_labels, all_predicted, average='weighted')
    print(f'Validation Precision: {val_precision:.2%}')
    from sklearn.metrics import recall_score
    # 计算召回率
    recall = recall_score(all_labels, all_predicted, average='weighted')
    print(f'Validation Recall: {recall:.2%}')

    # 保存训练过程中的指标
    training_results['loss'].append(train_loss.item())
    training_results['train_accuracy'].append(train_accuracy)
    training_results['val_loss'].append(val_loss.cpu())
    training_results['val_accuracy'].append(val_accuracy)

    # 保存最好和第二好的模型权重
    if val_accuracy > best_val_accuracy:
        second_best_val_accuracy = best_val_accuracy
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_weight_path)  # 保存最好模型权重
    elif val_accuracy > second_best_val_accuracy:
        second_best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), second_model_weight_path)  # 保存第二好模型权重

# 保存训练结果到 .mat 文件
scipy.io.savemat(save_file_name, training_results)
print(f"Training results saved to {save_file_name}")

print('###################################################')

# 测试模型
correct = 0
total = 0
all_predicted = []
all_labels = []
# 加载最好的模型权重并在测试集上进行测试
best_model = SeConvNet()
best_model.load_state_dict(torch.load(best_model_weight_path))
best_model = best_model.to(device)
best_model.eval()


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.2%}')
# 计算 Precision 和 F1-Score
val_precision = precision_score(all_labels, all_predicted, average='weighted')
val_f1_score = f1_score(all_labels, all_predicted, average='weighted')
print(f'Test Precision: {val_precision:.4f}')
print(f'Test F1-Score: {val_f1_score:.4f}')



