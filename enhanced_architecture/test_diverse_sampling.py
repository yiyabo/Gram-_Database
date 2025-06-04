#!/usr/bin/env python3
"""
测试多样性扩散采样
验证甘氨酸过度生成问题是否得到解决
"""

import torch
import numpy as np
from collections import Counter
from data_loader import sequence_to_tokens, tokens_to_sequence, AMINO_ACID_VOCAB, VOCAB_TO_AA
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet

def analyze_sequences(sequences):
    """分析序列的氨基酸分布"""
    all_aa = ""
    for seq in sequences:
        all_aa += seq
    
    aa_counts = Counter(all_aa)
    total_aa = len(all_aa)
    
    print(f"总氨基酸数: {total_aa}")
    print("氨基酸分布:")
    for aa, count in sorted(aa_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_aa * 100
        print(f"  {aa}: {count:4d} ({percentage:5.1f}%)")
    
    return aa_counts, total_aa

def test_diverse_generation():
    """测试多样性生成"""
    print("=" * 60)
    print("🧪 测试多样性扩散采样")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cpu")
    vocab_size = len(AMINO_ACID_VOCAB)
    
    # 创建模型组件
    scheduler = D3PMScheduler(num_timesteps=100, vocab_size=vocab_size)
    unet = D3PMUNet(vocab_size=vocab_size, hidden_dim=128, num_layers=4, 
                    max_seq_len=50)
    diffusion = D3PMDiffusion(unet, scheduler, device)
    
    print(f"✓ 模型创建成功")
    
    # 参数设置
    batch_size = 5
    seq_len = 30
    num_samples = 20  # 生成更多样本进行统计
    
    print(f"\n📊 生成参数:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 总样本数: {num_samples}")
    
    # 1. 测试标准采样
    print(f"\n🎲 1. 标准采样结果:")
    standard_sequences = []
    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        generated = diffusion.sample(
            batch_size=current_batch, 
            seq_len=seq_len, 
            num_inference_steps=20,
            temperature=1.0
        )
        
        for j in range(current_batch):
            seq = tokens_to_sequence(generated[j])
            standard_sequences.append(seq)
    
    print(f"标准采样生成了 {len(standard_sequences)} 个序列")
    analyze_sequences(standard_sequences)
    
    # 2. 测试多样性采样
    print(f"\n🌈 2. 多样性采样结果:")
    diverse_sequences = []
    for i in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - i)
        generated = diffusion.diverse_sample(
            batch_size=current_batch, 
            seq_len=seq_len, 
            num_inference_steps=20,
            diversity_strength=0.5,
            temperature=1.2
        )
        
        for j in range(current_batch):
            seq = tokens_to_sequence(generated[j])
            diverse_sequences.append(seq)
    
    print(f"多样性采样生成了 {len(diverse_sequences)} 个序列")
    diverse_counts, diverse_total = analyze_sequences(diverse_sequences)
    
    # 3. 对比分析
    print(f"\n📈 3. 对比分析:")
    
    # 计算甘氨酸比例
    standard_g_count = sum(seq.count('G') for seq in standard_sequences)
    standard_total = sum(len(seq) for seq in standard_sequences)
    standard_g_ratio = standard_g_count / standard_total if standard_total > 0 else 0
    
    diverse_g_count = diverse_counts.get('G', 0)
    diverse_g_ratio = diverse_g_count / diverse_total if diverse_total > 0 else 0
    
    print(f"甘氨酸(G)比例:")
    print(f"  标准采样: {standard_g_ratio:.1%}")
    print(f"  多样性采样: {diverse_g_ratio:.1%}")
    print(f"  改善比例: {((standard_g_ratio - diverse_g_ratio) / standard_g_ratio * 100):.1f}%")
    
    # 计算氨基酸多样性（使用的不同氨基酸种类）
    standard_diversity = len(set(''.join(standard_sequences)))
    diverse_diversity = len(set(''.join(diverse_sequences)))
    
    print(f"\n氨基酸多样性:")
    print(f"  标准采样: {standard_diversity} 种氨基酸")
    print(f"  多样性采样: {diverse_diversity} 种氨基酸")
    
    # 4. 展示示例序列
    print(f"\n🔍 4. 序列示例对比:")
    print(f"\n标准采样示例:")
    for i, seq in enumerate(standard_sequences[:5]):
        print(f"  {i+1}. {seq}")
    
    print(f"\n多样性采样示例:")
    for i, seq in enumerate(diverse_sequences[:5]):
        print(f"  {i+1}. {seq}")
    
    # 5. 评估生成质量
    print(f"\n🎯 5. 生成质量评估:")
    
    # 检查是否有过短的序列
    standard_short = sum(1 for seq in standard_sequences if len(seq) < 10)
    diverse_short = sum(1 for seq in diverse_sequences if len(seq) < 10)
    
    print(f"过短序列(< 10aa):")
    print(f"  标准采样: {standard_short}/{len(standard_sequences)} ({standard_short/len(standard_sequences)*100:.1f}%)")
    print(f"  多样性采样: {diverse_short}/{len(diverse_sequences)} ({diverse_short/len(diverse_sequences)*100:.1f}%)")
    
    # 检查重复序列
    standard_unique = len(set(standard_sequences))
    diverse_unique = len(set(diverse_sequences))
    
    print(f"\n序列唯一性:")
    print(f"  标准采样: {standard_unique}/{len(standard_sequences)} 唯一 ({standard_unique/len(standard_sequences)*100:.1f}%)")
    print(f"  多样性采样: {diverse_unique}/{len(diverse_sequences)} 唯一 ({diverse_unique/len(diverse_sequences)*100:.1f}%)")
    
    print(f"\n" + "=" * 60)
    if diverse_g_ratio < standard_g_ratio * 0.8:  # 如果甘氨酸比例减少20%以上
        print("🎉 多样性采样成功！甘氨酸过度生成问题得到改善")
        if diverse_diversity > standard_diversity:
            print("🌟 氨基酸多样性也得到提升")
    else:
        print("⚠️  多样性采样效果有限，可能需要调整参数")
    print("=" * 60)

if __name__ == "__main__":
    test_diverse_generation()
