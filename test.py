import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import confusion_matrix
import time
from thop import profile
from thop import clever_format
from PIL import Image
import scipy.io
import torch.nn as nn

from model.SeConvNet import SeConvNet

# 测试模型
correct = 0
total = 0
all_predicted = []
all_labels = []
# 加载最好的模型权重并在测试集上进行测试
best_model = SeConvNet()
best_model_weight_path = 'pretrainedWeight/best_model_weights_T0.pth'
best_model.load_state_dict(torch.load(best_model_weight_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = best_model.to(device)
best_model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0688, 0.0688, 0.0688], std=[0.0926, 0.0926, 0.0926])
])

test_dataset = ImageFolder(root='data/test', transform=transform)
# 创建数据加载器
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
print(f'Test Precision: {val_precision:.2%}')
from sklearn.metrics import recall_score
# 计算召回率
recall = recall_score(all_labels, all_predicted, average='weighted')
print(f'Test Recall: {recall:.2%}')

# 计算混淆矩阵
confusion = confusion_matrix(all_labels, all_predicted)
# 归一化混淆矩阵
normalized_confusion = confusion / confusion.sum(axis=1, keepdims=True)

# 保存预测标签、实际标签和混淆矩阵到 .mat 文件
mat_file_path = 'SavedData/normalized_confusion_Matrix.mat'
scipy.io.savemat(mat_file_path, {
    'predicted_labels': all_predicted,
    'actual_labels': all_labels,
    'confusion_matrix': confusion
})

print(f"Results saved to {mat_file_path}")

# 在创建模型并加载权重之后，加载一张图像并进行预处理，计算 FLOPs，Parameters，以及运行时间
image_path = 'test.jpg'  # 替换为实际图像路径
image = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0688, 0.0688, 0.0688], std=[0.0926, 0.0926, 0.0926])
])
input_tensor = preprocess(image).unsqueeze(0).to(device)

# 计算模型的 FLOPs
flops, params = profile(best_model, inputs=(input_tensor,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}, Parameters: {params}")


# 计算模型推理时间 GPU 的 吞吐量 Throughput
# 设置批量大小（batch size）和重复次数
batch_size = 64  # 替换为你想要测试的批量大小
num_runs = 100   # 替换为你想要测试的总次数
total_time = 0
with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time()
        outputs = best_model(input_tensor.repeat(batch_size, 1, 1, 1))  # 重复输入以匹配批量大小

        torch.cuda.synchronize()  # 等待所有 GPU 计算完成
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        # 在每个批次中，找到最大值的索引作为预测标签


average_time_per_batch = total_time / num_runs
throughput = batch_size / average_time_per_batch
throughput_fps = throughput * 1.0  # fps
print(f"Average Inference Time per Batch: {average_time_per_batch:.6f} seconds")
print(f"Throughput: {throughput_fps:.2f} fps")



# 将模型移动到 CPU
best_model.to('cpu')
input_tensor = preprocess(image).unsqueeze(0)
# 重复多次进行推理，然后计算平均时间
num_runs = 1000
total_time = 0
with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time()
        outputs = best_model(input_tensor)
        predicted_labels = torch.argmax(outputs, dim=1)

        # predicted_labels 是一个包含每个样本的预测标签的张量
        print(predicted_labels)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

average_time = total_time / num_runs * 1000
print(f"Average Inference Time on CPU: {average_time:.6f} ms")




model = nn.Sequential(*list(best_model.children())[:-1])
# 将模型移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 提取特征
features = []
labels = []

model.eval()
with torch.no_grad():
    for images, batch_labels in test_loader:
        images = images.to(device)
        batch_features = model(images)
        features.append(batch_features)
        labels.extend(batch_labels)

# 将特征堆叠为一个张量
features = torch.cat(features, dim=0).squeeze()
# 将标签转换为 NumPy 数组
labels = torch.tensor(labels).numpy()

# 最终的 features 张量可以作为你的图像特征表示
print("Extracted features shape:", features.shape)
print("Extracted features shape:", labels.shape)
# 将特征张量和标签保存为 .mat 文件
mat_data = {
    'features': features.cpu().numpy(),
    'labels': labels
}
save_file_name = 'SavedData/features_and_labels.mat'
scipy.io.savemat(save_file_name, mat_data)
print(f"Features and labels saved to {save_file_name}")
