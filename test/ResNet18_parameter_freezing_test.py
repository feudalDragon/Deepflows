
"""
ResNet18 参数冻结测试脚本
演示如何加载预训练权重、冻结部分参数、训练部分参数
"""

import os
import sys
import time
import numpy as np
import gc
import warnings
from sklearn.preprocessing import OneHotEncoder

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from DeepFlows import tensor
from DeepFlows.tensor import Tensor
from DeepFlows.autograd import no_grad
from DeepFlows import nn
from DeepFlows import backend_api
from DeepFlows.optim import Adam
from DeepFlows.optim.scheduler import WarmupCosineLR
from DeepFlows.utils import data_loader, pretrained_models
from ResNet import ResNet18



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


def prepare_model_with_frozen_params(model_name='resnet18', num_classes=10, device='cuda'):
    """
    准备带有参数冻结的模型
    """
    print(f"=== 准备{model_name}模型，冻结部分参数 ===")
    
    # 加载预训练模型
    try:
        # 使用create_model_with_pretrained_weights创建模型并加载权重
        model = pretrained_models.create_model_with_pretrained_weights(
            model_name=model_name,
            pretrained_dir=os.path.join(os.path.dirname(__file__), '..', 'DeepFlows', 'pretrained')
        )
        print(f"成功加载{model_name}预训练权重")
    except Exception as e:
        print(f"加载预训练权重失败: {str(e)}")
        print(f"将创建随机初始化的{model_name}模型")
        model = ResNet18(num_classes, img_size=(32, 32), device=device)
    
    # 修改模型以适应CIFAR10 (32x32输入, 10个类别)
    print("调整模型以适应CIFAR10数据集...")
    
    # 修改第一层卷积以适应32x32输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
    model.bn1 = nn.BatchNorm2d(64, device=device)
    model.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # 修改全连接层以适应10个类别
    model.fc = nn.Linear(512, num_classes, device=device)
    
    return model


def freeze_model_layers(model, freeze_strategy='partial'):
    """
    冻结模型的部分层
    freeze_strategy:
    - 'partial': 冻结前几层卷积和批量归一化层
    - 'all_but_fc': 冻结除了全连接层之外的所有层
    - 'none': 不冻结任何层
    """
    print(f"\n=== 应用参数冻结策略: {freeze_strategy} ===")
    
    if freeze_strategy == 'none':
        print("不冻结任何层，训练整个网络")
        for param in model.parameters():
            param.requires_grad = True
    elif freeze_strategy == 'all_but_fc':
        print("冻结除了全连接层之外的所有层")
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻全连接层
        for param in model.fc.parameters():
            param.requires_grad = True
    elif freeze_strategy == 'partial':
        print("冻结前几层卷积和批量归一化层")
        # 冻结第一层卷积和批量归一化层
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        
        # 冻结前两个残差块组中的所有block
        print("冻结layer1中的所有block...")
        for block in model.layer1:
            for param in block.parameters():
                param.requires_grad = False
        
        print("冻结layer2中的所有block...")
        for block in model.layer2:
            for param in block.parameters():
                param.requires_grad = False
        
        # 解冻后面的层和全连接层
        print("解冻layer3中的所有block...")
        for block in model.layer3:
            for param in block.parameters():
                param.requires_grad = True
        
        print("解冻layer4中的所有block...")
        for block in model.layer4:
            for param in block.parameters():
                param.requires_grad = True
        
        print("解冻全连接层...")
        for param in model.fc.parameters():
            param.requires_grad = True
    
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
    
    return total_params, trainable_params


def augment_batch(batch, epoch, max_epochs):
    """
    简单的数据增强
    """
    if np.random.rand() < 0.5:
        batch = np.fliplr(batch)
    return batch


def train_with_frozen_params(freeze_strategy='partial'):
    """
    使用冻结的模型进行训练
    """
    # 超参数
    batch_size = 64
    num_epochs = 1  # 可以根据需要调整
    learning_rate = 0.001
    max_train_batches = 5  
    max_test_batches = 20 
    
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
    
    # 准备模型
    model = prepare_model_with_frozen_params('resnet18', num_classes, device='cuda')
    
    # 冻结部分参数
    total_params, trainable_params = freeze_model_layers(model, freeze_strategy)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=5e-4)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs=2, T_max=num_epochs, eta_min=1e-5)
    
    # 训练记录
    train_losses = []
    test_accuracies = []
    
    print(f"\n=== 开始训练 (策略: {freeze_strategy}) ===")
    print(f"  训练批次大小: {batch_size}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    
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
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] 训练批次 [{batch_idx+1}/{max_train_batches}] 当前Loss: {loss_value:.4f}")
            
            # 内存管理
            outputs.dispose()
            loss.dispose()
            inputs.dispose()
            labels_onehot.dispose()
            del inputs, labels_onehot, outputs, loss
            tensor.Graph.free_graph()
        
        # 记录epoch损失
        train_loss = running_loss / max_train_batches
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} | 训练时间: {time.time()-epoch_start:.2f}s")
        
        # 更新学习率
        scheduler.step()
        
        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        
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
                    print(f"Epoch [{epoch+1}/{num_epochs}] 测试批次 [{batch_idx+1}/{len(test_loader.batch_sampler)}] 当前准确率: {current_acc:.2f}%")
                
                # 内存管理
                outputs.dispose()
                inputs.dispose()
                labels_onehot.dispose()
                del inputs, labels_onehot, outputs
                tensor.Graph.free_graph()
        
        # 记录测试准确率
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}] 测试准确率: {accuracy:.2f}% | 总时间: {time.time()-total_start_time:.2f}s")
        print("-" * 60)
    
    # 训练完成
    print(f"\n=== 训练完成 (策略: {freeze_strategy}) ===")
    print(f"总训练时间: {time.time()-total_start_time:.2f}s")
    print(f"最终测试准确率: {accuracy:.2f}%")
    
    return train_losses, test_accuracies


def main():
    """
    主函数，测试不同的参数冻结策略
    """
    print("=" * 60)
    print("ResNet18 参数冻结测试")
    print("=" * 60)
    
    # 测试不同的冻结策略
    strategies = ['partial', 'all_but_fc']
    
    for strategy in strategies:
        print(f"\n\n{'='*80}")
        print(f"测试策略: {strategy}")
        print(f"{'='*80}")
        train_losses, test_accuracies = train_with_frozen_params(strategy)
        
        # 打印训练结果
        print(f"\n{'='*40}")
        print(f"策略 {strategy} 结果摘要")
        print(f"{'='*40}")
        print(f"训练损失变化: {[f'{loss:.4f}' for loss in train_losses]}")
        print(f"测试准确率变化: {[f'{acc:.2f}%' for acc in test_accuracies]}")
        print(f"最佳测试准确率: {max(test_accuracies):.2f}%")


if __name__ == "__main__":
    main()