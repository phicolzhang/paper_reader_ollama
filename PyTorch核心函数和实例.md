# PyTorch 核心函数和实例

本文档提供了PyTorch深度学习框架的核心功能速查表，涵盖从基础张量操作到神经网络构建的所有常用函数和方法。这份文档旨在为PyTorch用户提供快速参考，包含简洁的代码示例和实用的函数说明。

## 基础张量操作

### 张量创建与初始化

PyTorch中的张量是所有操作的基础数据结构。创建张量有多种方式，每种方式都适用于不同的场景。

```python
import torch

# 创建张量
torch.tensor([1, 2, 3])                    # 从列表创建
torch.zeros(3, 4)                         # 创建零张量
torch.ones(2, 3)                          # 创建全1张量  
torch.randn(2, 3)                         # 创建正态分布随机张量
torch.rand(2, 3)                          # 创建均匀分布随机张量
torch.arange(0, 10, 2)                    # 创建等差数列张量
torch.linspace(0, 1, 5)                   # 创建线性间隔张量
torch.eye(3)                              # 创建单位矩阵
torch.empty(2, 3)                         # 创建未初始化张量
```


### 张量属性与形状操作

理解和操作张量的形状是深度学习中的核心技能。PyTorch提供了丰富的形状操作函数来满足各种需求。

```python
# 张量属性
x = torch.randn(2, 3, 4)
x.shape                                    # 张量形状
x.size()                                   # 张量尺寸
x.dtype                                    # 数据类型
x.device                                   # 设备类型
x.requires_grad                            # 梯度计算标志

# 形状操作
x.view(6, 4)                              # 改变形状（共享内存）
x.reshape(6, 4)                           # 改变形状（可能复制）
x.squeeze()                               # 移除大小为1的维度
x.unsqueeze(0)                            # 添加大小为1的维度
x.transpose(0, 1)                         # 交换维度
x.permute(2, 0, 1)                        # 重排维度
x.flatten()                               # 展平张量
```


### 张量运算

张量运算包括基本的数学运算、线性代数运算以及元素级操作，这些是构建复杂模型的基础。

```python
# 基本运算
a + b, torch.add(a, b)                    # 加法
a - b, torch.sub(a, b)                    # 减法
a * b, torch.mul(a, b)                    # 元素级乘法
a / b, torch.div(a, b)                    # 除法
a ** 2, torch.pow(a, 2)                   # 幂运算

# 线性代数
torch.mm(a, b)                            # 矩阵乘法
torch.bmm(a, b)                           # 批量矩阵乘法
torch.matmul(a, b)                        # 通用矩阵乘法
torch.dot(a, b)                           # 向量点积
torch.cross(a, b)                         # 向量叉积

# 统计运算
torch.sum(a)                              # 求和
torch.mean(a)                             # 均值
torch.std(a)                              # 标准差
torch.var(a)                              # 方差
torch.max(a)                              # 最大值
torch.min(a)                              # 最小值
torch.argmax(a)                           # 最大值索引
torch.argmin(a)                           # 最小值索引
```


## 神经网络构建

### 基础网络层

PyTorch的`nn`模块提供了构建神经网络所需的所有基础层，这些层可以组合成复杂的网络架构。

```python
import torch.nn as nn

# 线性层
nn.Linear(10, 5)                          # 全连接层：输入10，输出5
nn.Bilinear(10, 20, 5)                    # 双线性层

# 卷积层
nn.Conv1d(16, 32, 3)      # 1D卷积：16输入通道，32输出通道，核大小3
nn.Conv2d(3, 64, 3)       # 2D卷积：3输入通道，64输出通道，核大小3
nn.Conv3d(1, 8, 3)        # 3D卷积
nn.ConvTranspose2d(64, 32, 4, 2, 1)       # 转置卷积

# 池化层
nn.MaxPool2d(2)                           # 最大池化，核大小2
nn.AvgPool2d(2)                           # 平均池化
nn.AdaptiveMaxPool2d((1, 1))              # 自适应最大池化
nn.AdaptiveAvgPool2d((7, 7))              # 自适应平均池化

# 归一化层
nn.BatchNorm1d(100)                       # 1D批归一化
nn.BatchNorm2d(64)                        # 2D批归一化
nn.LayerNorm([10, 10])                    # 层归一化
nn.GroupNorm(2, 20)                       # 组归一化
```


### 激活函数

激活函数为神经网络引入非线性，是深度学习模型能够学习复杂模式的关键组件。

```python
# 激活函数
nn.ReLU()                                 # ReLU激活
nn.LeakyReLU(0.1)                         # LeakyReLU
nn.ELU()                                  # ELU激活
nn.SELU()                                 # SELU激活
nn.Sigmoid()                              # Sigmoid激活
nn.Tanh()                                 # Tanh激活
nn.Softmax(dim=1)                         # Softmax
nn.LogSoftmax(dim=1)                      # LogSoftmax
nn.GELU()                                 # GELU激活
nn.Swish()                                # Swish激活

# 函数式激活
torch.relu(x)
torch.sigmoid(x)
torch.tanh(x)
torch.softmax(x, dim=1)
```


### 损失函数

损失函数定义了模型预测与真实标签之间的差异，指导模型的训练过程。

```python
# 回归损失
nn.MSELoss()                              # 均方误差损失
nn.L1Loss()                               # L1损失（平均绝对误差）
nn.SmoothL1Loss()                         # 平滑L1损失
nn.HuberLoss()                            # Huber损失

# 分类损失
nn.CrossEntropyLoss()                     # 交叉熵损失
nn.NLLLoss()                              # 负对数似然损失
nn.BCELoss()                              # 二元交叉熵损失
nn.BCEWithLogitsLoss()                    # 带Logits的二元交叉熵
nn.MultiLabelSoftMarginLoss()             # 多标签软间隔损失

# 其他损失
nn.KLDivLoss()                            # KL散度损失
nn.PoissonNLLLoss()                       # 泊松负对数似然损失
nn.CosineEmbeddingLoss()                  # 余弦嵌入损失
```


## 优化器与训练

### 优化器

优化器实现了各种梯度下降算法的变体，用于更新模型参数以最小化损失函数。

```python
import torch.optim as optim

# 基础优化器
optim.SGD(model.parameters(), lr=0.01)                    
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)      
optim.Adam(model.parameters(), lr=0.001)                  
optim.AdamW(model.parameters(), lr=0.001)                 
optim.RMSprop(model.parameters(), lr=0.01)                
optim.Adagrad(model.parameters(), lr=0.01)                
optim.Adadelta(model.parameters())                        

# 学习率调度器
optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
```


### 训练循环

标准的PyTorch训练循环包括前向传播、损失计算、反向传播和参数更新四个基本步骤。

```python
# 基本训练循环
model.train()                             # 设置为训练模式
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()             # 清零梯度
        output = model(data)              # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()                   # 反向传播
        optimizer.step()                  # 更新参数

# 验证循环
model.eval()                              # 设置为评估模式
with torch.no_grad():                     # 禁用梯度计算
    for data, target in val_loader:
        output = model(data)
        loss = criterion(output, target)
```


## 数据处理

### 数据加载

PyTorch的数据加载系统提供了高效的数据预处理和批量加载功能，支持多进程加载和自定义数据集。

```python
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

# 数据加载器
DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 内置数据集
datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform
)
datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)
datasets.ImageFolder(root='./data', transform=transform)

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```


### 数据变换

数据变换用于预处理和数据增强，提高模型的泛化能力和训练效果。

```python
from torchvision import transforms

# 基础变换
transforms.Resize((224, 224))             # 调整大小
transforms.CenterCrop(224)                # 中心裁剪
transforms.RandomCrop(32, padding=4)      # 随机裁剪
transforms.RandomHorizontalFlip()         # 随机水平翻转
transforms.RandomVerticalFlip()           # 随机垂直翻转
transforms.RandomRotation(10)             # 随机旋转

# 颜色变换
transforms.ColorJitter(
    brightness=0.2, contrast=0.2,
    saturation=0.2, hue=0.1
)
transforms.RandomGrayscale(p=0.1)         # 随机转灰度
transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

# 组合变换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])
```


## 模型定义与管理

### 模型定义

PyTorch提供了灵活的模型定义方式，可以通过继承`nn.Module`类来创建自定义模型。

```python
# 简单模型定义
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 使用Sequential
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10)
)

# 使用ModuleList
layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```


### 模型保存与加载

模型的保存和加载是深度学习工作流程中的重要环节，PyTorch提供了多种保存和加载模型的方法。

```python
# 保存和加载整个模型
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 保存和加载状态字典（推荐）
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# 保存训练检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```


## 自动求导与梯度

### 自动求导

PyTorch的自动求导系统是其核心特性之一，能够自动计算复杂函数的梯度。

```python
# 梯度计算
x = torch.randn(5, requires_grad=True) # 创建需要梯度的张量
y = x.pow(2).sum()                     # 前向计算
y.backward()                           # 反向传播
print(x.grad)                          # 查看梯度

# 梯度控制
with torch.no_grad():                  # 禁用梯度计算
    y = model(x)

torch.set_grad_enabled(False)          # 全局禁用梯度
x.detach()                             # 分离张量，停止梯度跟踪
x.requires_grad_(False)                # 就地修改requires_grad属性

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```


### 梯度累积与冻结

在某些情况下，需要对梯度进行特殊处理，如累积梯度或冻结特定参数。

```python
# 梯度累积
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 冻结参数
for param in model.parameters():
    param.requires_grad = False           # 冻结所有参数

# 部分冻结
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False       # 冻结非全连接层
```


## 设备管理与并行计算

### GPU加速

充分利用GPU资源可以显著加速深度学习模型的训练和推理过程。

```python
# 设备检查和设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 将张量移动到GPU
x = x.to(device)
x = x.cuda()                              # 移动到默认GPU

# 将模型移动到GPU
model = model.to(device)
model = model.cuda()

# 多GPU并行
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# CUDA内存管理
torch.cuda.empty_cache()                  # 清空缓存
torch.cuda.memory_summary()               # 内存使用情况
```


### 分布式训练

分布式训练允许在多个GPU或多个节点上并行训练大型模型。

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 创建分布式模型
model = DDP(model, device_ids=[local_rank])

# 分布式数据采样器
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```


## 工具函数与调试

### 模型分析

了解模型的结构和参数对于优化和调试非常重要。

```python
# 模型信息
print(model)                              # 打印模型结构
sum(p.numel() for p in model.parameters() if p.requires_grad)  
                                          # 可训练参数数量

# 参数遍历
for name, param in model.named_parameters():
    print(f'{name}: {param.size()}')

# 模块遍历
for name, module in model.named_modules():
    print(f'{name}: {module}')

# 钩子函数
def hook_function(module, input, output):
    print(f'Layer: {module.__class__.__name__}, 
		Output shape: {output.shape}')

for layer in model.modules():
    layer.register_forward_hook(hook_function)
```


### 性能分析

性能分析工具帮助识别训练过程中的瓶颈并优化代码效率。

```python
# 时间测量
import time
start_time = time.time()
# 训练代码
end_time = time.time()
print(f'Training time: {end_time - start_time:.2f}s')

# CUDA事件计时
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# GPU计算
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)

# Profiler
with torch.profiler.profile(
    activities=[
	torch.profiler.ProfilerActivity.CPU, 
        torch.profiler.ProfilerActivity.CUDA
	]
) as prof:
    # 训练代码
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```
