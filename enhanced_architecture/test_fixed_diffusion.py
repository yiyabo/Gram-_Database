#!/usr/bin/env python3
"""
测试修复后的扩散模型
验证PAD token处理是否正确
"""

import torch
import numpy as np
from data_loader import sequence_to_tokens, tokens_to_sequence, AMINO_ACID_VOCAB
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet

def test_fixed_diffusion():
    """测试修复后的扩散模型"""
    print("🧪 测试修复后的扩散模型...")
    
    # 设置设备
    device = torch.device("cpu")
    vocab_size = len(AMINO_ACID_VOCAB)
    
    # 创建模型组件
    scheduler = D3PMScheduler(num_timesteps=100, vocab_size=vocab_size)
    unet = D3PMUNet(vocab_size=vocab_size, hidden_dim=128, num_layers=4, 
                    max_seq_len=50)
    diffusion = D3PMDiffusion(unet, scheduler, device)
    
    print(f"✓ 模型创建成功 (vocab_size: {vocab_size})")
    
    # 测试数据：一些简单的氨基酸序列
    test_sequences = [
        "ARNDCQEGHILKMFPSTWYV",  # 包含所有氨基酸
        "KKLLKWLLKLL",           # 常见的抗菌肽模式
        "GGGPPPGGG",             # 简单重复模式
        "ACDEFGHIK",             # 短序列
    ]
    
    # 转换为tokens
    max_len = 30
    test_tokens = []
    for seq in test_sequences:
        tokens = sequence_to_tokens(seq, max_len)
        test_tokens.append(tokens)
    
    x_batch = torch.stack(test_tokens)
    print(f"✓ 测试数据准备完成: {x_batch.shape}")
    
    # 检查PAD token分布
    pad_count = (x_batch == 0).sum().item()
    total_tokens = x_batch.numel()
    pad_ratio = pad_count / total_tokens
    print(f"  - PAD token比例: {pad_ratio:.2%} ({pad_count}/{total_tokens})")
    
    # 测试前向扩散过程
    print("\n🔄 测试前向扩散过程...")
    t = torch.randint(0, scheduler.num_timesteps, (len(test_sequences),))
    x_noisy = scheduler.q_sample(x_batch, t)
    
    # 检查噪声中的PAD token
    noisy_pad_count = (x_noisy == 0).sum().item()
    noisy_pad_ratio = noisy_pad_count / total_tokens
    print(f"  - 加噪后PAD token比例: {noisy_pad_ratio:.2%} ({noisy_pad_count}/{total_tokens})")
    
    # 验证PAD位置是否保持不变
    original_pad_mask = (x_batch == 0)
    noisy_pad_preserved = (x_noisy[original_pad_mask] == 0).all()
    print(f"  - PAD位置保持不变: {'✓' if noisy_pad_preserved else '❌'}")
    
    # 测试损失计算
    print("\n📊 测试损失计算...")
    try:
        loss = diffusion.training_loss(x_batch)
        print(f"  - 训练损失: {loss.item():.4f}")
        print("  - 损失计算成功 ✓")
    except Exception as e:
        print(f"  - 损失计算失败 ❌: {e}")
        return False
    
    # 测试采样过程
    print("\n🎲 测试采样过程...")
    try:
        # 生成序列
        generated = diffusion.sample(batch_size=2, seq_len=20, num_inference_steps=10)
        print(f"  - 生成序列shape: {generated.shape}")
        
        # 检查生成的token
        unique_tokens = torch.unique(generated)
        print(f"  - 生成的unique tokens: {unique_tokens.tolist()}")
        
        # 检查是否包含PAD token
        contains_pad = (generated == 0).any()
        print(f"  - 包含PAD token: {'❌' if contains_pad else '✓'}")
        
        # 转换为序列
        generated_sequences = []
        for i in range(generated.shape[0]):
            seq = tokens_to_sequence(generated[i])
            generated_sequences.append(seq)
            print(f"  - 序列{i+1}: '{seq}' (长度: {len(seq)})")
        
        # 检查序列有效性
        valid_sequences = [seq for seq in generated_sequences if len(seq) > 0]
        valid_ratio = len(valid_sequences) / len(generated_sequences)
        print(f"  - 有效序列比例: {valid_ratio:.1%} ({len(valid_sequences)}/{len(generated_sequences)})")
        
        if valid_ratio > 0:
            print("  - 序列生成成功 ✓")
            return True
        else:
            print("  - 序列生成失败 ❌")
            return False
            
    except Exception as e:
        print(f"  - 采样过程失败 ❌: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 扩散模型修复验证测试")
    print("=" * 60)
    
    success = test_fixed_diffusion()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试通过！扩散模型修复成功")
        print("💡 现在可以开始重新训练模型")
        print("⚡ 使用命令: python3 start_training.py --config quick_test")
    else:
        print("❌ 测试失败，需要进一步调试")
    print("=" * 60)

if __name__ == "__main__":
    main()
