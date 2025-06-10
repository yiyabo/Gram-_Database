#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试生成环境配置
验证所有依赖和文件是否正确配置
"""

import os
import sys
import importlib

def test_file_exists(filepath, description):
    """测试文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} 未找到: {filepath}")
        return False

def test_import(module_name, description):
    """测试模块导入"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} 导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔍 检查生成环境配置...")
    print("=" * 50)
    
    all_good = True
    
    # 检查Python库
    print("\n📚 检查Python库:")
    libraries = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("Bio", "Biopython"),
        ("peptides", "Peptides"),
        ("transformers", "Transformers")
    ]
    
    for module, desc in libraries:
        if not test_import(module, desc):
            all_good = False
    
    # 检查必要文件
    print("\n📁 检查必要文件:")
    files = [
        ("enhanced_architecture/output/checkpoints/best.pt", "生成模型检查点"),
        ("model/hybrid_classifier_best_tuned.keras", "预测模型"),
        ("model/hybrid_model_scaler.pkl", "特征标准化器"),
        ("data/Gram+-.fasta", "现有数据库"),
        ("enhanced_architecture/config/model_config.py", "模型配置"),
        ("enhanced_architecture/esm2_auxiliary_encoder.py", "ESM2编码器"),
        ("enhanced_architecture/diffusion_models/d3pm_diffusion.py", "扩散模型"),
        ("enhanced_architecture/data_loader.py", "数据加载器")
    ]
    
    for filepath, desc in files:
        if not test_file_exists(filepath, desc):
            all_good = False
    
    # 检查GPU/CPU
    print("\n💻 检查计算设备:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU不可用，将使用CPU (生成速度较慢)")
    except:
        print("❌ 无法检查GPU状态")
        all_good = False
    
    # 检查内存
    print("\n💾 检查系统资源:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ 可用内存: {memory.available / (1024**3):.1f} GB")
        if memory.available < 4 * (1024**3):  # 小于4GB
            print("⚠️  内存可能不足，建议至少4GB")
    except ImportError:
        print("⚠️  无法检查内存 (需要安装psutil)")
    
    # 测试简单导入
    print("\n🧪 测试项目模块导入:")
    sys.path.append(os.path.join(os.getcwd(), 'enhanced_architecture'))
    
    modules_to_test = [
        ("config.model_config", "模型配置"),
        ("data_loader", "数据加载器")
    ]
    
    for module, desc in modules_to_test:
        if not test_import(module, desc):
            all_good = False
    
    # 最终结果
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 所有检查通过！环境配置正确")
        print("💡 可以运行: python generate_high_quality_sequences.py")
        return 0
    else:
        print("❌ 存在配置问题，请修复后再运行")
        print("💡 常见解决方案:")
        print("   - 安装缺失的Python库: pip install -r requirements.txt")
        print("   - 检查模型文件路径是否正确")
        print("   - 确保在项目根目录运行")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)