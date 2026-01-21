import os
import sys
import json
import numpy as np
import argparse
from typing import Dict, Any, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 从pretrained_models模块导入参数映射和按需加载函数
from DeepFlows.utils.pretrained_models import PARAM_MAPPING, load_model_param_mapping

# 模型参数映射，直接使用从pretrained_models导入的映射
MODEL_PARAM_MAPPINGS = {}


def convert_pytorch_to_deepflows(pytorch_checkpoint_path: str, 
                              deepflows_save_path: str, 
                              model_name: str) -> Dict[str, Any]:
    """
    将PyTorch预训练模型转换为DeepFlows格式
    
    Args:
        pytorch_checkpoint_path: PyTorch模型权重文件路径
        deepflows_save_path: DeepFlows模型权重保存路径
        model_name: 模型名称
        
    Returns:
        转换后的模型元数据
    """
    import torch
    
    logger.info(f"加载PyTorch模型权重: {pytorch_checkpoint_path}")
    
    # 按需加载当前模型的参数映射表
    load_model_param_mapping(model_name)
    logger.info(f"{model_name}映射表包含 {len(PARAM_MAPPING[model_name])} 个参数映射")
    
    # 加载PyTorch模型权重
    checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu')
    
    # 如果checkpoint是字典且包含state_dict键，则使用state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 创建DeepFlows格式的权重字典
    deepflows_weights = {'model_parameters': {}}
    
    # 获取当前模型的参数映射
    param_mapping = PARAM_MAPPING.get(model_name, {})
    
    # 映射并转换权重
    mapped_count = 0
    skipped_count = 0
    
    for pytorch_name, pytorch_param in state_dict.items():
        # 查找映射的DeepFlows参数名
        deepflows_name = param_mapping.get(pytorch_name, None)
        
        if deepflows_name is not None:
            # 将PyTorch张量转换为NumPy数组
            np_array = pytorch_param.cpu().numpy()
            deepflows_weights['model_parameters'][deepflows_name] = np_array
            mapped_count += 1
        else:
            # 记录未映射的参数
            logger.debug(f"跳过未映射的参数: {pytorch_name}")
            skipped_count += 1
    
    logger.info(f"参数映射完成: 成功映射 {mapped_count} 个参数, 跳过 {skipped_count} 个参数")
    
    # 保存转换后的权重
    import pickle
    os.makedirs(os.path.dirname(os.path.abspath(deepflows_save_path)), exist_ok=True)
    
    with open(deepflows_save_path, 'wb') as f:
        pickle.dump(deepflows_weights, f)
    
    logger.info(f"转换后的模型已保存到: {deepflows_save_path}")
    
    # 创建元数据
    metadata = {
        'model_name': model_name,
        'source': 'pytorch',
        'source_path': pytorch_checkpoint_path,
        'conversion_time': str(np.datetime64('now')),
        'mapped_parameters': mapped_count,
        'skipped_parameters': skipped_count
    }
    
    return metadata


def download_pytorch_pretrained_model(model_name: str, save_dir: str) -> str:
    """
    下载PyTorch预训练模型
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
        
    Returns:
        下载的模型权重文件路径
    """
    import torch
    import torchvision.models as models
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_pytorch.pth")
    
    # 尝试导入权重类，适配不同版本的PyTorch
    try:
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG16_Weights, MobileNet_V2_Weights
        use_new_api = True
    except ImportError:
        use_new_api = False
    
    # 根据模型名称选择对应的PyTorch模型和权重
    if use_new_api:
        # 使用新版API（PyTorch 1.13+）
        if model_name == 'resnet18':
            logger.info(f"下载PyTorch预训练模型: {model_name}")
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            torch.save(model.state_dict(), save_path)
        elif model_name == 'resnet50':
            logger.info(f"下载PyTorch预训练模型: {model_name}")
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            torch.save(model.state_dict(), save_path)
        elif model_name == 'vgg16':
            logger.info(f"下载PyTorch预训练模型: {model_name}")
            model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            torch.save(model.state_dict(), save_path)
        elif model_name == 'mobilenet_v1':
            logger.info(f"下载PyTorch预训练模型: {model_name} (使用MobileNetV2替代)")
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            torch.save(model.state_dict(), save_path)
        else:
            raise ValueError(f"不支持自动下载的模型: {model_name}")
    else:
        # 使用旧版API
        model_dict = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'vgg16': models.vgg16,
            'mobilenet_v1': models.mobilenet_v2,  # 使用V2替代V1
        }
        
        if model_name in model_dict:
            logger.info(f"下载PyTorch预训练模型: {model_name}")
            model = model_dict[model_name](pretrained=True)
            torch.save(model.state_dict(), save_path)
        else:
            raise ValueError(f"不支持自动下载的模型: {model_name}")
    
    logger.info(f"PyTorch模型已保存到: {save_path}")
    return save_path

def main():
    """
    命令行入口，用于转换预训练模型
    """
    parser = argparse.ArgumentParser(description='将PyTorch/Paddle预训练模型转换为DeepFlows格式')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='模型名称，支持: resnet18, resnet50, mobilenet_v1, vgg16')
    parser.add_argument('--source', type=str, default='pytorch', 
                       choices=['pytorch', 'paddle', 'auto'], 
                       help='源框架，默认为pytorch')
    parser.add_argument('--source_path', type=str, default=None, 
                       help='源框架模型权重文件路径，默认为自动下载')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='DeepFlows模型权重保存目录，默认为DeepFlows/pretrained')
    parser.add_argument('--metadata_output', type=str, default=None, 
                       help='元数据输出文件路径，默认为不保存')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置源模型路径
    if args.source_path is None:
        if args.source == 'pytorch':
            source_path = download_pytorch_pretrained_model(args.model_name, output_dir)
        else:
            raise ValueError(f"自动下载仅支持PyTorch模型，请提供{args.source}模型的路径")
    else:
        source_path = args.source_path
    
    # 设置输出路径
    output_path = os.path.join(output_dir, f"{args.model_name}.pkl")
    
    # 根据源框架进行转换
    if args.source == 'pytorch':
        metadata = convert_pytorch_to_deepflows(source_path, output_path, args.model_name)
    else:
        raise ValueError(f"不支持的源框架: {args.source}")
    
    # 保存元数据
    if args.metadata_output is not None:
        with open(args.metadata_output, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"元数据已保存到: {args.metadata_output}")
    
    # 同时保存元数据到模型目录
    metadata_path = os.path.join(output_dir, f"{args.model_name}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"元数据已保存到: {metadata_path}")

if __name__ == '__main__':
    main()
