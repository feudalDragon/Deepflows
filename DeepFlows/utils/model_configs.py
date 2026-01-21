"""
预训练模型配置文件
定义支持的模型结构和参数映射关系
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

# 支持的预训练模型列表
SUPPORTED_MODELS = [
    'resnet18',
    'resnet50',
    'vgg16',
    'mobilenet_v1'
]

# 预训练模型元数据，包括类别数量、输入尺寸等
MODEL_METADATA = {
    'resnet18': {
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'architecture': 'residual',
        'description': 'ResNet-18模型，包含18个卷积层',
        'block_type': 'basic'
    },

    'resnet50': {
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'architecture': 'residual',
        'description': 'ResNet-50模型，包含50个卷积层',
        'block_type': 'bottleneck'
    },
    'vgg16': {
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'architecture': 'vgg',
        'description': 'VGG-16模型，包含16个卷积/全连接层',
        'channels_sequence': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    },
    'mobilenet_v1': {
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'architecture': 'mobile',
        'description': 'MobileNet V1模型',
        'width_multiplier': 1.0,
        'depth_multiplier': 1
    }
}


# 模型默认路径配置
MODEL_DEFAULT_PATHS = {
    # 默认预训练模型存储目录
    'pretrained_dir': 'DeepFlows/pretrained',
    # 下载缓存目录
    'download_cache': 'DeepFlows/cache',
    # 模型元数据JSON文件扩展名
    'metadata_extension': '.json',
    # 模型权重文件扩展名
    'weight_extension': '.pkl'
}

# ResNet模型块结构定义
RESNET_BLOCK_CONFIG = {
    # ResNet-18层配置
    'resnet18': {
        'block_type': 'basic',
        'layers': [2, 2, 2, 2]  # 每层重复次数
    },
    # ResNet-50层配置
    'resnet50': {
        'block_type': 'bottleneck',
        'layers': [3, 4, 6, 3]
    }
}

# VGG模型结构配置
VGG_CONFIG = {
    'vgg16': {
        # 每一层的配置：[输出通道数, 卷积核大小, 步长, 是否有池化]
        'features': [
            (64, 3, 1, False),
            (64, 3, 1, True),  # True表示后面接池化层
            (128, 3, 1, False),
            (128, 3, 1, True),
            (256, 3, 1, False),
            (256, 3, 1, False),
            (256, 3, 1, True),
            (512, 3, 1, False),
            (512, 3, 1, False),
            (512, 3, 1, True),
            (512, 3, 1, False),
            (512, 3, 1, False),
            (512, 3, 1, True)
        ],
        # 全连接层配置
        'classifier': [4096, 4096, 1000]
    }
}

# MobileNet V1模型结构配置
MOBILENET_V1_CONFIG = {
    # 每一层的配置：[输出通道数, 步长]
    'conv_layers': [
        (32, 2),
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1)
    ]
}

# 预训练模型标准化参数
IMAGENET_STATS = {
    # ImageNet数据集的均值和标准差
    'mean': [0.485, 0.456, 0.406],  # RGB通道
    'std': [0.229, 0.224, 0.225]    # RGB通道
}

def get_model_metadata(model_name: str) -> Optional[Dict]:
    """
    获取指定模型的元数据
    
    Args:
        model_name: 模型名称
        
    Returns:
        包含模型元数据的字典，如果模型不存在则返回None
    """
    return MODEL_METADATA.get(model_name, None)



def is_model_supported(model_name: str) -> bool:
    """
    检查指定模型是否支持
    
    Args:
        model_name: 模型名称
        
    Returns:
        布尔值，表示模型是否支持
    """
    return model_name in SUPPORTED_MODELS

def get_model_config(model_name: str) -> Optional[Dict]:
    """
    获取指定模型的结构配置
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型配置字典，如果模型不存在则返回None
    """
    # 检查是哪种架构的模型
    if model_name.startswith('resnet'):
        return RESNET_BLOCK_CONFIG.get(model_name, None)
    elif model_name.startswith('vgg'):
        return VGG_CONFIG.get(model_name, None)
    elif model_name == 'mobilenet_v1':
        return MOBILENET_V1_CONFIG
    else:
        return None

def get_normalization_stats():  
    """
    获取ImageNet预训练模型使用的标准化参数
    
    Returns:
        包含均值和标准差的字典
    """
    return IMAGENET_STATS
