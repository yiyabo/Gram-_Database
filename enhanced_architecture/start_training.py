#!/usr/bin/env python3
"""
启动增强型抗菌肽生成模型训练
"""

import sys
import os
import argparse

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强型抗菌肽生成模型训练")
    parser.add_argument(
        "--config",
        type=str,
        default="quick_test",
        choices=["default", "quick_test", "production", "dual_4090"],
        help="配置名称"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="从检查点恢复训练的路径"
    )
    parser.add_argument(
        "--test-only", 
        action="store_true",
        help="仅测试系统，不开始训练"
    )
    
    args = parser.parse_args()
    
    try:
        # 导入必要模块
        print("正在导入模块...")
        from main_trainer import EnhancedAMPTrainer
        
        # 测试模式
        if args.test_only:
            print("🧪 运行系统测试模式...")
            
            # 检查数据文件
            required_files = [
                "main_training_sequences.txt",
                "positive_sequences.txt", 
                "negative_sequences.txt"
            ]
            
            print("\n检查数据文件:")
            for file in required_files:
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        lines = len([l for l in f if l.strip()])
                    print(f"  ✓ {file}: {lines} 个序列")
                else:
                    print(f"  ❌ {file}: 文件不存在")
                    return 1
            
            # 测试配置
            from config.model_config import get_config
            config = get_config(args.config)
            print(f"\n✓ 配置 '{args.config}' 加载成功")
            print(f"  - 训练轮数: {config.training.num_epochs}")
            print(f"  - 批次大小: {config.data.batch_size}")
            print(f"  - 学习率: {config.training.learning_rate}")
            
            print("\n🎉 系统测试通过! 可以开始训练。")
            print("💡 使用以下命令开始训练:")
            print(f"   python3 start_training.py --config {args.config}")
            return 0
        
        # 正常训练模式
        print(f"🚀 开始训练 (配置: {args.config})")
        
        # 创建训练器
        trainer = EnhancedAMPTrainer(config_name=args.config)
        
        # 如果指定了恢复路径，加载检查点
        if args.resume:
            print(f"📂 从检查点恢复: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        trainer.train()
        
        print("✅ 训练完成!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
