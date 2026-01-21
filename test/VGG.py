import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DeepFlows.tensor import Tensor
from DeepFlows import nn
from DeepFlows import backend_api

class VGG16Model(nn.Module):
    def __init__(self, num_classes=10, device="cuda", img_size=(32, 32)):
        super().__init__()
        self.device = device
        self.img_size = img_size
        
        # VGG16特征提取部分
        # 第一阶段
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1_1 = nn.BatchNorm2d(64, device=device)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1_2 = nn.BatchNorm2d(64, device=device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二阶段
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2_1 = nn.BatchNorm2d(128, device=device)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2_2 = nn.BatchNorm2d(128, device=device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三阶段
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn3_1 = nn.BatchNorm2d(256, device=device)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn3_2 = nn.BatchNorm2d(256, device=device)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn3_3 = nn.BatchNorm2d(256, device=device)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四阶段
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn4_1 = nn.BatchNorm2d(512, device=device)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn4_2 = nn.BatchNorm2d(512, device=device)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn4_3 = nn.BatchNorm2d(512, device=device)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第五阶段
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn5_1 = nn.BatchNorm2d(512, device=device)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn5_2 = nn.BatchNorm2d(512, device=device)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn5_3 = nn.BatchNorm2d(512, device=device)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        # 对于32x32输入，经过5次池化后尺寸变为1x1
        # 对于224x224输入，经过5次池化后尺寸变为7x7
        feature_map_size = img_size[0] // (2 ** 5)
        fc_input_dim = 512 * feature_map_size * feature_map_size
        
        # 分类器部分
        self.fc1 = nn.Linear(fc_input_dim, 4096, device=device)
        self.fc2 = nn.Linear(4096, 4096, device=device)
        self.fc3 = nn.Linear(4096, num_classes, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        
        # 第一阶段
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # 第二阶段
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # 第三阶段
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # 第四阶段
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu(x)
        x = self.pool4(x)
        
        # 第五阶段
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu(x)
        x = self.pool5(x)
        
        # 展平
        x = x.reshape(x.shape[0], -1)
        
        # 分类器
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def VGG16(num_classes=10, device="cuda", img_size=(32, 32)):
    """
    VGG16模型工厂函数
    
    Args:
        num_classes: 分类类别数
        device: 运行设备
        img_size: 输入图像尺寸
        
    Returns:
        VGG16模型实例
    """
    return VGG16Model(num_classes=num_classes, device=device, img_size=img_size)
