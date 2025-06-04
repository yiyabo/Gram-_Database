#!/usr/bin/env python3
"""
快速诊断生成问题
"""

import torch
import time
from data_loader import sequence_to_tokens, tokens_to_sequence, AMINO_ACID_VOCAB
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet

def quick_generation_test():
    """快速测试生成功能"""
    print("🔍 快速生成测试开始...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用设备: {device}")
    
    # 创建模型
    vocab_size = len(AMINO_ACID_VOCAB)
    scheduler = D3PMScheduler(num_timesteps=50)  # 减少timesteps加速测试
    
    unet = D3PMUNet(
        vocab_size=vocab_size,
        hidden_dim=256,  # 减小hidden_dim
        num_layers=4,    # 减少层数
        num_heads=8,
        max_seq_len=30   # 正确的参数名
    )
    
    diffusion = D3PMDiffusion(
        model=unet,
        scheduler=scheduler,
        vocab_size=vocab_size
    ).to(device)
    
    print("✓ 模型创建完成")
    
    # 测试1: 简单采样
    print("\n🧪 测试1: 简单采样...")
    start_time = time.time()
    
    try:
        generated_tokens = diffusion.sample(
            batch_size=1,
            seq_len=20,  # 更短的序列
            device=device
        )
        
        sequence = tokens_to_sequence(generated_tokens[0])
        elapsed = time.time() - start_time
        
        print(f"✓ 简单采样成功 ({elapsed:.2f}s)")
        print(f"  生成序列: {sequence}")
        
    except Exception as e:
        print(f"❌ 简单采样失败: {e}")
        return False
    
    # 测试2: 多样性采样（如果存在）
    print("\n🧪 测试2: 检查多样性采样方法...")
    
    if hasattr(diffusion, 'diverse_sample'):
        print("✓ 找到 diverse_sample 方法")
        
        try:
            start_time = time.time()
            generated_tokens = diffusion.diverse_sample(
                batch_size=1,
                seq_len=20,
                diversity_strength=0.3,
                temperature=1.0
            )
            
            sequence = tokens_to_sequence(generated_tokens[0])
            elapsed = time.time() - start_time
            
            print(f"✓ 多样性采样成功 ({elapsed:.2f}s)")
            print(f"  生成序列: {sequence}")
            
        except Exception as e:
            print(f"❌ 多样性采样失败: {e}")
            print(f"  错误详情: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("❌ 未找到 diverse_sample 方法")
        return False
    
    # 测试3: 批量生成
    print("\n🧪 测试3: 批量生成...")
    
    try:
        start_time = time.time()
        generated_tokens = diffusion.sample(
            batch_size=5,
            seq_len=15,
            device=device
        )
        
        sequences = [tokens_to_sequence(tokens) for tokens in generated_tokens]
        elapsed = time.time() - start_time
        
        print(f"✓ 批量生成成功 ({elapsed:.2f}s)")
        print(f"  生成 {len(sequences)} 条序列")
        for i, seq in enumerate(sequences):
            print(f"    {i+1}: {seq}")
            
    except Exception as e:
        print(f"❌ 批量生成失败: {e}")
        return False
    
    print("\n✅ 所有测试通过")
    return True

def test_generation_speed():
    """测试生成速度"""
    print("\n⏱️ 生成速度测试...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建简化模型
    vocab_size = len(AMINO_ACID_VOCAB)
    scheduler = D3PMScheduler(num_timesteps=20)  # 非常少的steps
    
    unet = D3PMUNet(
        vocab_size=vocab_size,
        hidden_dim=128,  # 很小的hidden_dim
        num_layers=2,    # 很少的层数
        num_heads=4,
        max_seq_len=30   # 正确的参数名
    )
    
    diffusion = D3PMDiffusion(
        model=unet,
        scheduler=scheduler,
        vocab_size=vocab_size
    ).to(device)
    
    # 测试不同batch_size的速度
    for batch_size in [1, 5, 10]:
        start_time = time.time()
        
        try:
            generated_tokens = diffusion.sample(
                batch_size=batch_size,
                seq_len=20,
                device=device
            )
            
            elapsed = time.time() - start_time
            per_sequence = elapsed / batch_size
            
            print(f"  Batch {batch_size}: {elapsed:.2f}s 总计, {per_sequence:.2f}s/序列")
            
        except Exception as e:
            print(f"  Batch {batch_size}: 失败 - {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 生成功能诊断测试")
    print("=" * 60)
    
    # 运行快速测试
    success = quick_generation_test()
    
    if success:
        # 运行速度测试
        test_generation_speed()
    else:
        print("\n❌ 基础测试失败，需要修复模型")
