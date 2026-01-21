import sys
import os

# 将父目录加入路径以便导入DeepFlows
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import DeepFlows

from DeepFlows.utils.pretrained_models import (
    download_pretrained_model,
    convert_weights_to_deepflows,
    get_pretrained_weights,
    list_available_pretrained_models,
    is_pretrained_available,
    DEFAULT_PRETRAINED_DIR
)

def test_pretrained_models():
    """测试预训练模型功能"""
    print("=== 测试预训练模型功能 ===")
    
    # 测试1：列出可用模型
    print("\n1. 列出所有可用的预训练模型:")
    available_models = list_available_pretrained_models()
    print(f"可用模型: {available_models}")
    
    # 选择一个模型进行测试
    test_model = 'resnet18'
    
    # 测试2：下载预训练模型
    print(f"\n2. 下载预训练模型 {test_model}:")
    try:
        temp_path = download_pretrained_model(test_model)
        print(f"成功下载到临时路径: {temp_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        return
    
    # 测试3：转换权重格式
    print(f"\n3. 将 {test_model} 权重转换为DeepFlows格式:")
    try:
        deepflows_path = convert_weights_to_deepflows(test_model, temp_path)
        print(f"成功转换并保存到: {deepflows_path}")
    except Exception as e:
        print(f"转换失败: {e}")
        return
    
    # 测试4：检查模型是否可用
    print(f"\n4. 检查 {test_model} 是否可用:")
    is_available = is_pretrained_available(test_model)
    print(f"{test_model} 可用: {is_available}")
    
    # 测试5：获取预训练权重
    print(f"\n5. 获取 {test_model} 预训练权重:")
    try:
        weights = get_pretrained_weights(test_model)
        print(f"成功加载权重，包含 {len(weights)} 个参数")
        # 打印前5个参数名称
        print("前5个参数名称:", list(weights.keys())[:5])
    except Exception as e:
        print(f"加载权重失败: {e}")
        return
    
    # 测试6：列出已下载的模型
    print("\n6. 列出已下载的预训练模型:")
    downloaded_models = list_available_pretrained_models(only_downloaded=True)
    print(f"已下载模型: {downloaded_models}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_pretrained_models()