import os, sys
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import builtins

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入DeepFlows相关模块
from DeepFlows import tensor
from DeepFlows.tensor import Tensor
from DeepFlows.autograd import no_grad
from DeepFlows import nn
from DeepFlows.optim import Adam
from DeepFlows.optim.scheduler import WarmupCosineLR
from DeepFlows.utils import data_loader
from DeepFlows import backend_api

# 导入预训练模型相关模块
from DeepFlows.utils import pretrained_models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, device="cuda"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        self.downsample = downsample
        self.stride = stride
        self.device = device

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            for layer in self.downsample:
                identity = layer(identity)
        out = out + identity
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, img_size=(32, 32), device="cuda"):
        super().__init__()
        self.device = device
        self.in_channels = 64  # 改为64以匹配预训练模型
        # 第一层改为64通道以匹配标准ResNet18结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(64, device=device)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 各层通道数也需要相应调整以匹配预训练模型
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes, device=device)  # 改为512以匹配预训练模型

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            conv = nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False, device=self.device)
            bn = nn.BatchNorm2d(out_channels, device=self.device)
            downsample = [conv, bn]
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, device=self.device))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, device=self.device))
        return layers

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        for block in self.layer1:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        for block in self.layer2:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        for block in self.layer3:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        for block in self.layer4:
            x = block(x)
            if not isinstance(x, Tensor):
                x = Tensor(x)
        x = tensor.mean(x, axis=2)
        x = tensor.mean(x, axis=2)
        x = self.fc(x)
        return x

def ResNet18(num_classes=10, img_size=(32, 32), device="cuda"):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes, img_size, device)

def load_cifar10_data():
    """
    加载CIFAR10数据集
    """
    import pickle
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cifar-10-batches-py')
    
    def load_batch(fname):
        with open(os.path.join(base, fname), 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        x = d['data']
        y = np.array(d['labels'], dtype=np.int32)
        x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        return x, y
    
    xs = []
    ys = []
    for i in range(1, 6):
        x, y = load_batch(f'data_batch_{i}')
        xs.append(x)
        ys.append(y)
    
    x_train = np.ascontiguousarray(np.concatenate(xs, axis=0))
    y_train = np.ascontiguousarray(np.concatenate(ys, axis=0))
    x_test, y_test = load_batch('test_batch')
    x_test = np.ascontiguousarray(x_test)
    y_test = np.ascontiguousarray(y_test)
    
    # 标准化处理
    m = x_train.mean(axis=(0, 2, 3), keepdims=True)
    s = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x_train = (x_train - m) / s
    x_test = (x_test - m) / s
    
    return x_train, y_train, x_test, y_test

def augment_batch(inputs, epoch, num_epochs):
    """
    数据增强函数
    """
    bs, c, h, w = inputs.shape
    pad = 4
    padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='reflect')
    
    # 随机裁剪
    ys = np.random.randint(0, 2 * pad + 1, size=bs)
    xs = np.random.randint(0, 2 * pad + 1, size=bs)
    out = np.empty_like(inputs)
    for i in range(bs):
        out[i] = padded[i, :, ys[i]:ys[i] + h, xs[i]:xs[i] + w]
    
    # 随机水平翻转
    flip_mask = np.random.rand(bs) < 0.5
    out[flip_mask] = out[flip_mask][:, :, :, ::-1]
    
    # 随机擦除
    if epoch < num_epochs - 5 and np.random.rand() < 0.2:
        erase_h = builtins.max(1, int(h * np.random.uniform(0.1, 0.2)))
        erase_w = builtins.max(1, int(w * np.random.uniform(0.1, 0.2)))
        ys_e = np.random.randint(0, h - erase_h + 1, size=bs)
        xs_e = np.random.randint(0, w - erase_w + 1, size=bs)
        for i in range(bs):
            out[i, :, ys_e[i]:ys_e[i]+erase_h, xs_e[i]:xs_e[i]+erase_w] = 0.0
    
    out = np.clip(out, -1.0, 1.0)
    return out

def prepare_model_for_transfer_learning(model_name='resnet18', num_classes=10, device='cuda'):
    """
    准备迁移学习模型
    
    Args:
        model_name: 预训练模型名称
        num_classes: 目标任务类别数
        device: 设备
        
    Returns:
        准备好的迁移学习模型
    """
    # 创建与CIFAR10兼容的ResNet18模型（32x32输入尺寸）
    print(f"正在创建与CIFAR10兼容的{model_name}模型...")
    
    # 1. 先创建一个与预训练模型结构匹配的模型
    # 注意：预训练模型通常是为224x224输入设计的，但我们会在后续修改为适应32x32
    # 首先确保模型结构与预训练权重兼容
    print("加载预训练权重...")
    model = pretrained_models.create_model_with_pretrained_weights(model_name, pretrained_dir=os.path.join(os.path.dirname(__file__), '..', 'DeepFlows', 'pretrained'))
    
    # 2. 替换模型的第一层卷积和最后一层全连接层，使其适应32x32输入和10个类别
    print("调整模型以适应CIFAR10数据集...")
    
    # 修改第一层卷积以适应32x32输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
    model.bn1 = nn.BatchNorm2d(64, device=device)
    model.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # 修改全连接层以适应10个类别
    model.fc = nn.Linear(512, num_classes, device=device)
    
    print("预训练模型准备完成！")
    return model

def train_transfer_learning():
    """
    使用预训练模型进行CIFAR10迁移学习
    """
    # 超参数
    batch_size = 64  # 减小batch size以解决CUDA内存不足问题
    num_epochs = 2  # 减少训练轮数以便快速看到结果
    learning_rate = 0.001
    target_acc = 85.0
    max_train_batches = 10  # 限制训练批次数量
    max_test_batches = 5    # 限制测试批次数量
    max_test_batches = None   # 使用所有测试批次
    
    # 加载数据集
    print("=== 加载CIFAR10数据集 ===")
    x_train, y_train, x_test, y_test = load_cifar10_data()
    num_classes = 10
    
    # 准备数据加载器
    encoder = OneHotEncoder(sparse_output=False)
    all_classes = np.arange(num_classes).reshape(-1, 1)
    encoder.fit(all_classes)
    
    loader = data_loader(x_train, y_train, batch_size, shuffle=True, prefetch_size=0, as_contiguous=True)
    test_loader = data_loader(x_test, y_test, batch_size, shuffle=False, prefetch_size=0, as_contiguous=True)
    
    # 准备迁移学习模型
    print("\n=== 准备迁移学习模型 ===")
    
    try:
        # 尝试加载预训练模型
        model = prepare_model_for_transfer_learning('resnet18', num_classes, device='cuda')
    except Exception as e:
        print(f"加载预训练模型失败: {str(e)}")
        print("将使用随机初始化的模型继续训练...")
        model = ResNet18(num_classes, img_size=(32, 32), device='cuda')
    
    # 设置迁移学习策略：训练整个网络（包括预训练层）
    print("\n=== 设置迁移学习策略 ===")
    
    # 解冻所有层，训练整个网络
    for param in model.parameters():
        param.requires_grad = True
    
    print("解冻所有层，训练整个网络")
    
    # 统计可训练参数
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        param_count = 1
        for dim in param.shape:
            param_count *= dim
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数比例: {((total_params - trainable_params) / total_params * 100):.1f}%")
    
    # 定义损失函数和优化器（只优化可训练参数）
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=5e-4)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, T_max=num_epochs, eta_min=1e-5)
    
    # 训练过程
    print("\n=== 开始迁移学习训练 ===")
    train_losses = []
    test_accuracies = []
    train_batch_losses = []
    test_batch_accuracies = []
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            
            # 数据增强
            inputs = augment_batch(inputs, epoch, num_epochs)
            
            # 标签处理
            eps = 0.05
            labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
            labels_onehot = labels_onehot * (1 - eps) + eps / num_classes
            
            # 转换为张量
            inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels_onehot)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            loss_value = loss.data.numpy().item()
            running_loss += loss_value
            train_batch_losses.append(loss_value)
            
            if batch_idx % 50 == 0 or batch_idx + 1 == min(len(loader.batch_sampler), max_train_batches) if max_train_batches is not None else batch_idx + 1 == len(loader.batch_sampler):
                print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{min(len(loader.batch_sampler), max_train_batches) if max_train_batches is not None else len(loader.batch_sampler)}] 当前Loss: {loss_value:.4f}")
            
            # 内存管理
            outputs.dispose()
            loss.dispose()
            inputs.dispose()
            labels_onehot.dispose()
            del inputs, labels_onehot, outputs, loss
            tensor.Graph.free_graph()
            
            if batch_idx % 50 == 0:
                gc.collect()
        
        # 记录epoch损失
        train_loss = running_loss / len(loader.batch_sampler)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | Time: {time.time()-epoch_start:.2f}s")
        
        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 开始测试...")
        
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                if max_test_batches is not None and batch_idx >= max_test_batches:
                    break
                
                labels_onehot = encoder.transform(labels.reshape(-1, 1)).astype(np.float32)
                inputs, labels_onehot = Tensor(inputs, device=backend_api.Device('cuda')), Tensor(labels_onehot, device=backend_api.Device('cuda'))
                
                outputs = model(inputs)
                
                # 计算准确率
                total += labels_onehot.shape[0]
                _pred = np.argmax(outputs.data.numpy(), 1).reshape(-1, 1)
                _true = np.argmax(labels_onehot.data.numpy(), 1).reshape(-1, 1)
                correct += np.sum(_pred == _true)
                
                if batch_idx % 10 == 0:
                    current_acc = 100 * correct / total
                    test_batch_accuracies.append(current_acc)
                    print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{min(len(test_loader.batch_sampler), max_test_batches) if max_test_batches is not None else len(test_loader.batch_sampler)}] 当前准确率: {current_acc:.2f}%")
                
                # 内存管理
                outputs.dispose()
                inputs.dispose()
                labels_onehot.dispose()
                del inputs, labels_onehot, outputs
                tensor.Graph.free_graph()
                
                if batch_idx % 20 == 0:
                    gc.collect()
        
        # 记录测试准确率
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        
        # 更新学习率
        scheduler.step()
        
        # 检查是否达到目标精度
        if accuracy >= target_acc:
            print(f"达到目标准确率 {target_acc:.2f}% ，提前停止训练")
            break
        
        gc.collect()
    
    # 训练完成
    gc.collect()
    total_time = time.time() - total_start_time
    print(f"\n=== 迁移学习训练完成 ===")
    print(f"总训练时间: {total_time:.2f}s")
    print(f"最终测试准确率: {accuracy:.2f}%")
    print(f"最高测试准确率: {max(test_accuracies):.2f}%")
    print(f"最后一个epoch的损失: {train_loss:.4f}")
    
    # 保存训练结果图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_batch_losses) + 1), train_batch_losses)
    plt.title('训练损失')
    plt.xlabel('批次')
    plt.ylabel('损失')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_batch_accuracies) + 1), test_batch_accuracies, color='orange')
    plt.title('测试准确率')
    plt.xlabel('批次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('resnet_cifar10_transfer_learning.png')
    print("训练图表已保存为 'resnet_cifar10_transfer_learning.png'")
    
    # 打印训练结果摘要
    print("\n=== 训练结果摘要 ===")
    print(f"数据集: CIFAR10")
    print(f"模型架构: ResNet18 (迁移学习)")
    print(f"训练批次大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100 - ((total_params - trainable_params) / total_params * 100):.1f}%)")
    print(f"最终测试准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    train_transfer_learning()