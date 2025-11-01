#!/usr/bin/env python3
"""
测试所有组件的导入和基本功能
"""

def test_imports():
    """测试所有核心组件的导入"""
    try:
        print("开始测试组件导入...")
        
        # 测试配置
        from gram_predictor.config.model_config import get_config
        config = get_config('quick_test')
        print("✓ 配置系统正常")
        
        # 测试数据加载器
        from gram_predictor.data_loader import AntimicrobialPeptideDataset, ContrastiveAMPDataset, sequence_to_tokens
        test_seq = "KRWWKWWRR"
        tokens = sequence_to_tokens(test_seq, 20)
        print("✓ 数据加载器正常")
        
        # 测试ESM-2编码器
        from gram_predictor.esm2_auxiliary_encoder import ESM2AuxiliaryEncoder, ContrastiveLoss
        print("✓ ESM-2编码器正常")
        
        # 测试扩散模型
        from gram_predictor.diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler
        print("✓ D3PM扩散模型正常")
        
        # 测试评估器
        from enhanced_architecture.evaluation.evaluator import ModelEvaluator, EvaluationMetrics
        print("✓ 评估器正常")
        
        # 测试主训练器
        from enhanced_architecture.main_trainer import EnhancedAMPTrainer
        print("✓ 主训练器正常")
        
        print("\n🎉 所有核心组件导入和基本测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_files():
    """测试数据文件是否存在"""
    import os
    
    required_files = [
        "main_training_sequences.txt",
        "positive_sequences.txt", 
        "negative_sequences.txt"
    ]
    
    print("\n检查数据文件...")
    for file in required_files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                lines = len(f.readlines())
            print(f"✓ {file}: {lines} 行")
        else:
            print(f"❌ {file}: 文件不存在")

if __name__ == "__main__":
    success = test_imports()
    test_data_files()
    
    if success:
        print("\n✨ 系统就绪，可以开始训练!")
    else:
        print("\n⚠️  存在问题，请检查依赖项")
