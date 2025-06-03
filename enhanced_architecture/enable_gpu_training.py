#!/usr/bin/env python3
"""
启用GPU训练的配置脚本
"""

import torch
import os

def check_gpu_availability():
    """检查GPU可用性"""
    print("🔍 检查GPU可用性...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"✅ CUDA可用")
        print(f"  • GPU数量: {gpu_count}")
        print(f"  • 当前GPU: {gpu_name}")
        print(f"  • 显存大小: {gpu_memory:.1f} GB")
        
        return True, "cuda"
    
    # 检查MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"✅ MPS可用 (Apple Silicon)")
        print(f"  • 设备: Apple Silicon GPU")
        return True, "mps"
    
    else:
        print("❌ 没有可用的GPU，将使用CPU训练")
        return False, "cpu"

def enable_gpu_training():
    """启用GPU训练"""
    gpu_available, device_type = check_gpu_availability()
    
    if not gpu_available:
        print("⚠️ 没有可用的GPU，保持CPU训练配置")
        return False
    
    # 读取主训练器文件
    trainer_file = "main_trainer.py"
    
    if not os.path.exists(trainer_file):
        print(f"❌ 找不到文件: {trainer_file}")
        return False
    
    with open(trainer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经启用GPU
    if 'torch.device("cuda"' in content or 'torch.device("mps"' in content:
        print("✅ GPU训练已经启用")
        return True
    
    # 替换设备配置
    old_device_config = '''        # 强制使用CPU以避免MPS兼容性问题
        self.device = torch.device("cpu")
        # 如果在服务器上，可以改为：
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'''
    
    if device_type == "cuda":
        new_device_config = '''        # 自动选择最佳设备 (GPU优先)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 使用GPU训练: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU不可用，使用CPU训练")'''
    
    elif device_type == "mps":
        new_device_config = '''        # 自动选择最佳设备 (Apple Silicon GPU优先)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 使用Apple Silicon GPU训练")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 使用CUDA GPU训练: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU不可用，使用CPU训练")'''
    
    # 执行替换
    if old_device_config in content:
        new_content = content.replace(old_device_config, new_device_config)
        
        # 备份原文件
        backup_file = f"{trainer_file}.backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📁 原文件已备份到: {backup_file}")
        
        # 写入新配置
        with open(trainer_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 已启用{device_type.upper()}训练")
        print("💡 如需恢复CPU训练，请运行: python disable_gpu_training.py")
        
        return True
    else:
        print("⚠️ 未找到预期的设备配置代码，请手动修改")
        return False

def disable_gpu_training():
    """禁用GPU训练，恢复CPU配置"""
    trainer_file = "main_trainer.py"
    backup_file = f"{trainer_file}.backup"
    
    if os.path.exists(backup_file):
        with open(backup_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(trainer_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        print("✅ 已恢复CPU训练配置")
        os.remove(backup_file)
        print(f"🗑️ 已删除备份文件: {backup_file}")
        return True
    else:
        print("❌ 找不到备份文件，无法自动恢复")
        return False

def get_recommended_batch_size(device_type, gpu_memory_gb=None):
    """根据设备类型推荐批次大小"""
    if device_type == "cpu":
        return 8, "CPU训练建议使用较小批次"
    
    elif device_type == "mps":
        return 16, "Apple Silicon GPU建议批次大小"
    
    elif device_type == "cuda":
        if gpu_memory_gb is None:
            return 32, "CUDA GPU默认批次大小"
        elif gpu_memory_gb >= 24:
            return 64, "大显存GPU可使用较大批次"
        elif gpu_memory_gb >= 12:
            return 32, "中等显存GPU推荐批次"
        elif gpu_memory_gb >= 8:
            return 16, "较小显存GPU建议批次"
        else:
            return 8, "显存不足，使用小批次"
    
    return 16, "默认批次大小"

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU训练配置工具")
    parser.add_argument("--enable", action="store_true", help="启用GPU训练")
    parser.add_argument("--disable", action="store_true", help="禁用GPU训练")
    parser.add_argument("--check", action="store_true", help="仅检查GPU可用性")
    
    args = parser.parse_args()
    
    if args.disable:
        disable_gpu_training()
    elif args.enable:
        enable_gpu_training()
    elif args.check:
        gpu_available, device_type = check_gpu_availability()
        if gpu_available:
            if device_type == "cuda":
                current_device = torch.cuda.current_device()
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                batch_size, reason = get_recommended_batch_size(device_type, gpu_memory)
            else:
                batch_size, reason = get_recommended_batch_size(device_type)
            
            print(f"\n💡 推荐配置:")
            print(f"  • 推荐批次大小: {batch_size} ({reason})")
            print(f"  • 设备类型: {device_type}")
    else:
        # 默认行为：检查并询问是否启用
        gpu_available, device_type = check_gpu_availability()
        
        if gpu_available:
            response = input(f"\n🤔 检测到{device_type.upper()}可用，是否启用GPU训练? (y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                enable_gpu_training()
            else:
                print("保持当前CPU训练配置")
        else:
            print("将继续使用CPU训练")

if __name__ == "__main__":
    main()
