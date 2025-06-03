#!/usr/bin/env python3
"""
完整训练启动脚本 - 提供更友好的训练管理界面
"""

import os
import sys
import time
import argparse
from datetime import datetime
import subprocess

def print_banner():
    """打印启动横幅"""
    print("=" * 80)
    print("🚀 增强型抗菌肽生成模型 - 完整训练系统")
    print("=" * 80)
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查依赖...")
    
    required_packages = [
        'torch', 'transformers', 'numpy', 'pandas', 
        'scikit-learn', 'tqdm', 'tensorboard'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def check_system_status():
    """检查系统状态"""
    print("\n🧪 检查系统状态...")
    
    try:
        # 运行系统测试
        result = subprocess.run([
            sys.executable, "start_training.py", "--test-only", "--config", "production"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 系统测试通过")
            return True
        else:
            print("❌ 系统测试失败")
            print("错误输出:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ 系统测试超时")
        return False
    except Exception as e:
        print(f"❌ 系统测试出错: {e}")
        return False

def estimate_training_time(config_name):
    """估算训练时间"""
    estimates = {
        'quick_test': "10-30分钟",
        'default': "2-6小时", 
        'production': "6-24小时"
    }
    return estimates.get(config_name, "未知")

def show_config_info(config_name):
    """显示配置信息"""
    print(f"\n📋 训练配置: {config_name}")
    
    try:
        from config.model_config import get_config
        config = get_config(config_name)
        
        print(f"  • 训练轮数: {config.training.num_epochs}")
        print(f"  • 批次大小: {config.data.batch_size}")
        print(f"  • 学习率: {config.training.learning_rate}")
        print(f"  • 扩散模型维度: {config.diffusion.hidden_dim}")
        print(f"  • 扩散模型层数: {config.diffusion.num_layers}")
        print(f"  • 预计训练时间: {estimate_training_time(config_name)}")
        
        if config.training.use_wandb:
            print("  • WandB监控: 启用")
        if config.training.use_mixed_precision:
            print("  • 混合精度: 启用")
            
    except Exception as e:
        print(f"  ⚠️ 无法加载配置详情: {e}")

def setup_monitoring():
    """设置监控"""
    print("\n📊 设置训练监控...")
    
    # 创建输出目录
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("output/checkpoints", exist_ok=True)
    os.makedirs("output/tensorboard", exist_ok=True)
    
    print("  ✓ 输出目录已创建")
    print("  💡 训练开始后可以运行以下命令查看监控:")
    print("     tensorboard --logdir output/tensorboard")
    print("     然后在浏览器打开 http://localhost:6006")

def confirm_training(config_name):
    """确认开始训练"""
    print(f"\n🎯 准备开始 {config_name} 配置的完整训练")
    print(f"⏱️ 预计训练时间: {estimate_training_time(config_name)}")
    print("\n⚠️ 注意事项:")
    print("  • 训练过程中请保持终端开启")
    print("  • 可以使用 Ctrl+C 安全中断训练")
    print("  • 训练会自动保存检查点，可以随时恢复")
    print("  • 建议在训练期间监控系统资源使用情况")
    
    while True:
        response = input("\n🤔 确认开始训练吗? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            return True
        elif response in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y 或 n")

def start_training(config_name, resume_path=None):
    """开始训练"""
    print(f"\n🚀 开始训练 (配置: {config_name})")
    print("=" * 50)
    
    # 构建训练命令
    cmd = [sys.executable, "start_training.py", "--config", config_name]
    if resume_path:
        cmd.extend(["--resume", resume_path])
    
    try:
        # 启动训练
        subprocess.run(cmd)
        print("\n✅ 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        print("💾 检查点已保存，可以使用 --resume 参数恢复训练")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整训练启动脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default="production",
        choices=["quick_test", "default", "production"],
        help="训练配置"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="跳过系统检查，直接开始训练"
    )
    parser.add_argument(
        "--auto-confirm", 
        action="store_true",
        help="自动确认，不询问用户"
    )
    
    args = parser.parse_args()
    
    # 打印启动信息
    print_banner()
    
    # 系统检查
    if not args.skip_checks:
        if not check_dependencies():
            return 1
            
        if not check_system_status():
            print("\n❌ 系统检查失败，请解决问题后重试")
            return 1
    
    # 显示配置信息
    show_config_info(args.config)
    
    # 设置监控
    setup_monitoring()
    
    # 确认训练
    if not args.auto_confirm:
        if not confirm_training(args.config):
            print("🚫 训练已取消")
            return 0
    
    # 开始训练
    success = start_training(args.config, args.resume)
    
    if success:
        print("\n🎉 训练流程完成!")
        print("📁 检查 output/ 目录查看训练结果")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
