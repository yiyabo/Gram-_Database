#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务器端ESM-2模型部署脚本
解压本地下载的模型包并设置正确的路径
"""

import os
import sys
import tarfile
import shutil
from pathlib import Path

def setup_esm2_models():
    """在服务器上设置ESM-2模型"""
    
    package_name = "esm2_models_package.tar.gz"
    
    # 检查压缩包是否存在
    if not os.path.exists(package_name):
        print(f"❌ 未找到模型包: {package_name}")
        print(f"请确保已将 {package_name} 上传到当前目录")
        return False
    
    print(f"📦 找到模型包: {package_name}")
    
    # 解压模型包
    print(f"🔄 正在解压模型包...")
    try:
        with tarfile.open(package_name, "r:gz") as tar:
            tar.extractall(".")
        print(f"✅ 模型包解压完成")
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False
    
    # 检查解压后的目录
    local_models_dir = "./local_esm2_models"
    if not os.path.exists(local_models_dir):
        print(f"❌ 解压后未找到模型目录: {local_models_dir}")
        return False
    
    # 列出可用的模型
    print(f"\n📋 可用的模型:")
    model_dirs = []
    for item in os.listdir(local_models_dir):
        item_path = os.path.join(local_models_dir, item)
        if os.path.isdir(item_path):
            model_dirs.append(item)
            print(f"  📁 {item}")
    
    if not model_dirs:
        print(f"❌ 未找到任何模型目录")
        return False
    
    # 创建符号链接或移动到标准位置
    target_dir = "./esm2_models"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    shutil.move(local_models_dir, target_dir)
    print(f"📁 模型已移动到: {target_dir}")
    
    # 创建模型路径映射文件
    create_model_config(target_dir, model_dirs)
    
    print(f"\n✅ ESM-2模型设置完成!")
    print(f"📍 模型位置: {os.path.abspath(target_dir)}")
    
    return True

def create_model_config(models_dir, model_dirs):
    """创建模型配置文件"""
    config_content = f"""# ESM-2模型本地路径配置
# 由setup_esm2_server.py自动生成

import os

# 模型根目录
ESM2_MODELS_ROOT = "{os.path.abspath(models_dir)}"

# 模型路径映射
MODEL_PATHS = {
"""
    
    for model_dir in model_dirs:
        original_name = model_dir.replace("_", "/")
        local_path = os.path.join(models_dir, model_dir)
        config_content += f'    "{original_name}": "{os.path.abspath(local_path)}",\n'
    
    config_content += """}

def get_model_path(model_name):
    \"\"\"获取模型的本地路径\"\"\"
    return MODEL_PATHS.get(model_name, model_name)

def is_local_model(model_name):
    \"\"\"检查是否为本地模型\"\"\"
    return model_name in MODEL_PATHS
"""
    
    config_file = "esm2_model_config.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"📝 模型配置文件已创建: {config_file}")

def test_model_loading():
    """测试模型加载"""
    print(f"\n🧪 测试模型加载...")
    
    try:
        # 导入配置
        sys.path.insert(0, '.')
        import esm2_model_config
        
        # 测试加载第一个可用模型
        if esm2_model_config.MODEL_PATHS:
            test_model = list(esm2_model_config.MODEL_PATHS.keys())[0]
            test_path = esm2_model_config.get_model_path(test_model)
            
            print(f"  🔍 测试模型: {test_model}")
            print(f"  📁 模型路径: {test_path}")
            
            # 检查必要文件是否存在
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            missing_files = []
            
            for file_name in required_files:
                file_path = os.path.join(test_path, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"  ⚠️  缺少文件: {missing_files}")
            else:
                print(f"  ✅ 模型文件完整")
                
                # 尝试加载tokenizer
                from transformers import EsmTokenizer
                tokenizer = EsmTokenizer.from_pretrained(test_path, local_files_only=True)
                print(f"  ✅ Tokenizer加载成功")
                
                return True
                
    except Exception as e:
        print(f"  ❌ 模型加载测试失败: {e}")
        return False
    
    return False

if __name__ == "__main__":
    print("🚀 开始设置ESM-2模型...")
    
    if setup_esm2_models():
        if test_model_loading():
            print(f"\n🎉 ESM-2模型部署成功!")
            print(f"\n📋 使用说明:")
            print(f"1. 在代码中导入: import esm2_model_config")
            print(f"2. 获取模型路径: model_path = esm2_model_config.get_model_path('facebook/esm2_t33_650M_UR50D')")
            print(f"3. 加载模型: EsmModel.from_pretrained(model_path, local_files_only=True)")
        else:
            print(f"\n⚠️  模型部署完成，但加载测试失败，请检查模型文件")
    else:
        print(f"\n❌ ESM-2模型部署失败")