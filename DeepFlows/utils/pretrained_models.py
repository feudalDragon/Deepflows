import os
import sys
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预训练模型库的存储路径
DEFAULT_PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pretrained')

# 支持的模型类型
SUPPORTED_MODELS = {
    'resnet18',
    'resnet50',
    'mobilenet_v1',
    'vgg16'
}

# 模型与源框架的映射
MODEL_TO_SOURCE = {
    'resnet18': 'pytorch',  # 默认使用PyTorch的预训练模型
    'resnet50': 'pytorch',
    'mobilenet_v1': 'pytorch',
    'vgg16': 'pytorch'
}

# 映射关系：PyTorch参数名 -> DeepFlows参数名
PARAM_MAPPING = {
    'resnet18': {},
    'resnet50': {},
    'mobilenet_v1': {},
    'vgg16': {}
}

# 项目根目录路径（预先生成）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..', '..')


def load_model_param_mapping(model_name: str) -> None:
    """
    按需加载指定模型的参数映射表
    
    Args:
        model_name: 模型名称，必须是SUPPORTED_MODELS中的一个
    """
    # 转换为小写以处理大小写问题
    model_name = model_name.lower()
    
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {SUPPORTED_MODELS}")
    
    # 如果映射表已经加载，则直接返回
    if PARAM_MAPPING.get(model_name, {}):
        logger.info(f"{model_name}的参数映射表已加载")
        return
    
    try:
        # 动态构建映射表文件路径，支持不同的命名格式
        # 将模型名称统一转换为映射文件的命名格式（如mobilenet_v1 -> mobilenetv1）
        mapping_file_name = model_name.replace('_', '') + '_complete_mapping.json'
        mapping_path = os.path.join(PROJECT_ROOT, 'model_param_mappings', mapping_file_name)
        
        # 加载映射表文件
        with open(mapping_path, 'r', encoding='utf-8') as f:
            PARAM_MAPPING[model_name] = json.load(f)
        
        logger.info(f"加载{model_name}完整参数映射表成功，包含{len(PARAM_MAPPING[model_name])}个参数映射")
        
    except FileNotFoundError as e:
        logger.warning(f"无法找到{model_name}的参数映射表文件: {e}")
        logger.info(f"{model_name}将使用默认的空参数映射表")
    except json.JSONDecodeError as e:
        logger.error(f"解析{model_name}的参数映射表文件失败: {e}")
        logger.info(f"{model_name}将使用默认的空参数映射表")
    except Exception as e:
        logger.error(f"加载{model_name}的参数映射表时发生未知错误: {e}")
        logger.info(f"{model_name}将使用默认的空参数映射表")

def ensure_pretrained_dir_exists(pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> None:
    """
    确保预训练模型目录存在
    
    Args:
        pretrained_dir: 预训练模型存储目录
    """
    os.makedirs(pretrained_dir, exist_ok=True)

def get_pretrained_model_path(model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> str:
    """
    获取预训练模型的文件路径
    
    Args:
        model_name: 模型名称
        pretrained_dir: 预训练模型存储目录
        
    Returns:
        模型参数文件的路径
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {SUPPORTED_MODELS}")
    
    ensure_pretrained_dir_exists(pretrained_dir)
    return os.path.join(pretrained_dir, f"{model_name}.pkl")

def get_model_info_path(model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> str:
    """
    获取模型信息文件的路径
    
    Args:
        model_name: 模型名称
        pretrained_dir: 预训练模型存储目录
        
    Returns:
        模型信息文件的路径
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {SUPPORTED_MODELS}")
    
    ensure_pretrained_dir_exists(pretrained_dir)
    return os.path.join(pretrained_dir, f"{model_name}.json")

def save_model_metadata(model_name: str, metadata: Dict[str, Any], 
                       pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> None:
    """
    保存模型元数据信息
    
    Args:
        model_name: 模型名称
        metadata: 元数据字典
        pretrained_dir: 预训练模型存储目录
    """
    info_path = get_model_info_path(model_name, pretrained_dir)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_model_metadata(model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> Dict[str, Any]:
    """
    加载模型元数据信息
    
    Args:
        model_name: 模型名称
        pretrained_dir: 预训练模型存储目录
        
    Returns:
        元数据字典
    """
    info_path = get_model_info_path(model_name, pretrained_dir)
    if not os.path.exists(info_path):
        return {}
    
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def download_pretrained_model(model_name: str, source: str = None,
                             pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> str:
    """
    从源框架下载预训练模型
    
    Args:
        model_name: 模型名称
        source: 源框架，如 'pytorch' 或 'paddle'
        pretrained_dir: 预训练模型存储目录
        
    Returns:
        下载的模型路径
    """
    # 转换为小写以处理大小写问题
    model_name = model_name.lower()
    
    if model_name not in SUPPORTED_MODELS:
        logger.error(f"不支持的模型: {model_name}. 支持的模型: {SUPPORTED_MODELS}")
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {SUPPORTED_MODELS}")
    
    if source is None:
        source = MODEL_TO_SOURCE.get(model_name, 'pytorch')
        logger.info(f"使用默认源框架 {source} 下载 {model_name}")
    else:
        logger.info(f"使用指定源框架 {source} 下载 {model_name}")
    
    # 确保预训练目录存在
    ensure_pretrained_dir_exists(pretrained_dir)
    
    logger.info(f"开始从{source}下载{model_name}的预训练模型...")
    
    try:
        # 实际实现中，这里需要调用PyTorch/Paddle的API下载模型
        # 并保存到临时文件
        
        # 临时返回路径
        temp_path = os.path.join(pretrained_dir, f"{model_name}_{source}_temp.pth")
        
        # 实际PyTorch模型下载逻辑
        if source == 'pytorch':
            logger.info(f"使用torchvision下载PyTorch预训练模型 {model_name}...")
            import torch
            import torchvision.models as models
            
            # 将模型名称转换为torchvision兼容的格式
            torch_model_name = model_name.replace('_', '')
            
            try:
                # 获取模型构造函数
                model_func = getattr(models, torch_model_name)
                # 创建模型实例并下载预训练权重
                torch_model = model_func(pretrained=True)
                # 保存模型权重
                torch.save(torch_model.state_dict(), temp_path)
                logger.info(f"成功下载并保存 {model_name} 到: {temp_path}")
            except AttributeError:
                logger.error(f"torchvision中不存在模型: {torch_model_name}")
                raise Exception(f"torchvision中不存在模型: {torch_model_name}")
        else:
            logger.error(f"暂不支持源框架: {source}")
            raise NotImplementedError(f"暂不支持源框架: {source}")
        
        logger.info(f"{model_name} 下载完成")
        
        # 保存元数据
        metadata = {
            'model_name': model_name,
            'source': source,
            'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_checkpoint': f'PyTorch官方预训练模型: {model_name}',
            'download_path': temp_path
        }
        save_model_metadata(model_name, metadata, pretrained_dir)
        logger.info(f"已保存 {model_name} 的元数据")
        
        return temp_path
    
    except Exception as e:
        logger.error(f"下载 {model_name} 时发生错误: {str(e)}")
        # 清理临时文件
        temp_path = os.path.join(pretrained_dir, f"{model_name}_{source}_temp.pth")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"已清理临时文件: {temp_path}")
        raise Exception(f"下载 {model_name} 失败: {str(e)}")

def convert_weights_to_deepflows(model_name: str, source_weights_path: str,
                               pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> str:
    """
    将源框架的权重转换为DeepFlows格式
    
    Args:
        model_name: 模型名称
        source_weights_path: 源框架权重文件路径
        pretrained_dir: DeepFlows格式权重保存目录
        
    Returns:
        转换后的权重文件路径
    """
    logger.info(f"将{model_name}的权重从源框架格式转换为DeepFlows格式...")
    
    # 1. 加载参数映射表
    load_model_param_mapping(model_name)
    param_mapping = PARAM_MAPPING.get(model_name, {})
    
    # 2. 加载源框架权重
    logger.info(f"加载源框架权重文件: {source_weights_path}")
    import torch
    torch_weights = torch.load(source_weights_path, map_location='cpu')
    logger.info(f"成功加载源框架权重，包含{len(torch_weights)}个参数")
    
    # 3. 根据映射关系转换权重
    deepflows_weights = {}
    converted_count = 0
    skipped_count = 0
    
    for src_param_name, src_param_tensor in torch_weights.items():
        if src_param_name in param_mapping:
            # 应用参数映射
            dst_param_name = param_mapping[src_param_name]
            # 将PyTorch张量转换为numpy数组
            dst_param_value = src_param_tensor.numpy()
            deepflows_weights[dst_param_name] = dst_param_value
            converted_count += 1
        else:
            skipped_count += 1
    
    logger.info(f"权重转换统计: {converted_count}个参数成功转换，{skipped_count}个参数跳过")
    
    # 4. 保存转换后的权重
    # 使用.pkl格式保存
    deepflows_path = get_pretrained_model_path(model_name, pretrained_dir)
    logger.info(f"保存转换后的权重到: {deepflows_path}")
    
    import pickle
    with open(deepflows_path, 'wb') as f:
        pickle.dump(deepflows_weights, f)
    logger.info(f"权重转换完成，保存到: {deepflows_path}")
    
    return deepflows_path

def get_pretrained_weights(model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR,
                          force_download: bool = False) -> Dict[str, np.ndarray]:
    """
    获取预训练模型权重
    
    Args:
        model_name: 模型名称
        pretrained_dir: 预训练模型存储目录
        force_download: 是否强制重新下载
        
    Returns:
        模型权重字典
    """
    # 获取DeepFlows格式的权重文件路径（.pkl格式）
    model_path = get_pretrained_model_path(model_name, pretrained_dir)
    
    # 如果权重文件不存在或强制下载，则下载并转换
    if force_download or not os.path.exists(model_path):
        temp_path = download_pretrained_model(model_name, pretrained_dir=pretrained_dir)
        convert_weights_to_deepflows(model_name, temp_path, pretrained_dir)
    
    # 加载DeepFlows格式的权重
    logger.info(f"加载DeepFlows格式权重文件: {model_path}")
    
    import pickle
    with open(model_path, 'rb') as f:
        weights_dict = pickle.load(f)
    
    logger.info(f"成功加载DeepFlows格式权重，包含{len(weights_dict)}个参数")
    
    return weights_dict

def load_pretrained_model(model, model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR,
                         force_download: bool = False) -> None:
    """
    加载预训练权重到模型
    
    Args:
        model: DeepFlows模型实例
        model_name: 预训练模型名称
        pretrained_dir: 预训练模型存储目录
        force_download: 是否强制重新下载
    """
    # 获取DeepFlows格式的权重文件路径（.pkl格式）
    model_path = get_pretrained_model_path(model_name, pretrained_dir)
    
    # 如果权重文件不存在或强制下载，则下载并转换
    if force_download or not os.path.exists(model_path):
        temp_path = download_pretrained_model(model_name, pretrained_dir=pretrained_dir)
        convert_weights_to_deepflows(model_name, temp_path, pretrained_dir)
    
    try:
        # 直接加载权重到模型
        logger.info(f"加载预训练模型{model_name}到DeepFlows模型...")
        weights_dict = get_pretrained_weights(model_name, pretrained_dir)
        
        # 检查模型是否有load_weights方法
        if hasattr(model, 'load_weights'):
            model.load_weights(weights_dict)
            logger.info(f"成功使用模型的load_weights方法加载预训练权重")
        else:
            # 否则尝试直接设置权重属性
            logger.info(f"模型没有load_weights方法，尝试直接设置权重属性")
            model.weights = weights_dict
    except Exception as e:
        logger.error(f"加载预训练权重到模型时发生错误: {str(e)}")
        raise Exception(f"加载预训练权重到模型失败: {str(e)}")
    
def list_available_pretrained_models(only_downloaded: bool = False, pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> List[str]:
    """
    列出可用的预训练模型
    
    Args:
        only_downloaded: 是否只返回已下载的模型（默认为False，返回所有支持的模型）
        pretrained_dir: 预训练模型存储目录
        
    Returns:
        可用模型名称列表
    """
    if not only_downloaded:
        return list(SUPPORTED_MODELS)
    
    # 返回已下载的模型
    available_models = []
    for model_name in SUPPORTED_MODELS:
        if is_pretrained_available(model_name, pretrained_dir):
            available_models.append(model_name)
    return available_models


def is_pretrained_available(model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR) -> bool:
    """
    检查指定模型的预训练权重是否已下载
    
    Args:
        model_name: 模型名称
        pretrained_dir: 预训练模型存储目录
        
    Returns:
        是否已下载
    """
    if model_name not in SUPPORTED_MODELS:
        return False
    
    # 检查.pkl格式的文件（实际保存的格式）
    model_path = get_pretrained_model_path(model_name, pretrained_dir)
    return os.path.exists(model_path)


def create_model_with_pretrained_weights(model_name: str, pretrained_dir: str = DEFAULT_PRETRAINED_DIR,
                                       force_download: bool = False) -> Any:
    """
    创建模型实例并加载预训练权重
    
    Args:
        model_name: 模型名称
        pretrained_dir: 预训练模型存储目录
        force_download: 是否强制重新下载
        
    Returns:
        加载了预训练权重的DeepFlows模型实例
    """
    logger.info(f"创建{model_name}模型实例...")
    
    # 确保test目录在sys.path中（只需要添加一次）
    test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test')
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    
    # 模型注册表：模型名称 -> (模块名, 类名, 构造参数)
    model_registry = {
        'resnet18': ('ResNet', 'ResNet18', {}),
        'resnet50': ('ResNet', 'ResNet50', {}),
        'mobilenet_v1': ('MobileNet', 'MobileNetV1', {}),
        'vgg16': ('VGG', 'VGG16', {'img_size': (224, 224)})
    }
    
    if model_name not in model_registry:
        raise ValueError(f"暂不支持自动创建{model_name}模型，请手动创建模型实例")
    
    module_name, class_name, kwargs = model_registry[model_name]
    model = None
    
    try:
        # 动态导入模块
        module = __import__(module_name)
        # 获取模型类
        model_class = getattr(module, class_name)
        # 创建模型实例
        model = model_class(**kwargs)
    except ImportError as e:
        logger.error(f"无法导入{module_name}模块，请确保test/{module_name}.py存在")
        raise ImportError(f"Failed to import {module_name} module: {str(e)}")
    except AttributeError:
        # 特殊处理ResNet50的情况
        if model_name == 'resnet50':
            try:
                from ResNet import ResNet, ResidualBlock
                # 创建一个近似ResNet50结构的模型
                model = ResNet(ResidualBlock, [3, 4, 6, 3])
            except Exception as e:
                logger.error(f"无法创建ResNet50模型: {str(e)}")
                raise ImportError(f"Failed to create ResNet50 model: {str(e)}")
        else:
            logger.error(f"{module_name}模块中没有{class_name}类")
            raise ImportError(f"Failed to find {class_name} in {module_name} module")
    except Exception as e:
        logger.error(f"创建{model_name}模型实例时发生错误: {str(e)}")
        raise
    
    # 加载预训练权重
    load_pretrained_model(model, model_name, pretrained_dir, force_download)
    
    logger.info(f"{model_name}模型创建并加载预训练权重成功")
    return model


def list_available_models() -> List[str]:
    """
    列出所有可用的模型（为了兼容示例脚本）
    
    Returns:
        可用模型名称列表
    """
    return list_available_pretrained_models(only_downloaded=True)







