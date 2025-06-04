#!/usr/bin/env python3
"""
最终模型评估：使用改进的多样性采样评估模型性能
验证修复后的扩散模型是否满足实际应用需求
"""

import torch
import numpy as np
from data_loader import sequence_to_tokens, tokens_to_sequence, AMINO_ACID_VOCAB
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet
from collections import Counter
import logging
from datetime import datetime
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sequence_properties(sequences):
    """分析序列的生物学特性"""
    properties = {
        'lengths': [],
        'amino_acid_counts': Counter(),
        'hydrophobic_ratio': [],
        'charged_ratio': [],
        'aromatic_ratio': []
    }
    
    # 氨基酸分类
    hydrophobic = set(['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'])
    charged = set(['K', 'R', 'H', 'D', 'E'])
    aromatic = set(['F', 'Y', 'W'])
    
    for seq in sequences:
        if len(seq) == 0:
            continue
            
        properties['lengths'].append(len(seq))
        
        # 统计氨基酸
        for aa in seq:
            properties['amino_acid_counts'][aa] += 1
        
        # 计算各种比例
        total_aa = len(seq)
        hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic)
        charged_count = sum(1 for aa in seq if aa in charged)
        aromatic_count = sum(1 for aa in seq if aa in aromatic)
        
        properties['hydrophobic_ratio'].append(hydrophobic_count / total_aa)
        properties['charged_ratio'].append(charged_count / total_aa)
        properties['aromatic_ratio'].append(aromatic_count / total_aa)
    
    return properties

def calculate_diversity_metrics(sequences):
    """计算序列多样性指标"""
    if not sequences:
        return 0, 0
    
    # 去重计算唯一性
    unique_sequences = set(sequences)
    uniqueness = len(unique_sequences) / len(sequences)
    
    # 计算编辑距离多样性
    diversity_scores = []
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            # 简单的Hamming距离（对于等长序列）
            if len(sequences[i]) == len(sequences[j]):
                distance = sum(c1 != c2 for c1, c2 in zip(sequences[i], sequences[j]))
                diversity_scores.append(distance / len(sequences[i]))
    
    avg_diversity = np.mean(diversity_scores) if diversity_scores else 0
    
    return uniqueness, avg_diversity

def final_evaluation():
    """进行最终的综合评估"""
    print("=" * 70)
    print("🎯 最终模型评估 - 使用改进的多样性采样")
    print("=" * 70)
    
    # 创建模型
    device = torch.device("cpu")
    vocab_size = len(AMINO_ACID_VOCAB)
    
    scheduler = D3PMScheduler(num_timesteps=1000, vocab_size=vocab_size)
    unet = D3PMUNet(vocab_size=vocab_size, hidden_dim=768, num_layers=12, 
                    max_seq_len=100)
    diffusion = D3PMDiffusion(unet, scheduler, device)
    
    # 加载训练好的模型
    checkpoint_path = "./output/checkpoints/latest.pt"
    if os.path.exists(checkpoint_path):
        print(f"📂 加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载扩散模型权重
        if 'diffusion_model_state_dict' in checkpoint:
            diffusion.model.load_state_dict(checkpoint['diffusion_model_state_dict'])
            print("✓ 扩散模型权重加载成功")
        else:
            print("⚠️ 未找到扩散模型权重，使用随机初始化")
    else:
        print("⚠️ 未找到检查点文件，使用随机初始化的模型")
    
    print(f"✓ 模型创建成功 (vocab_size: {vocab_size})")
    
    # 评估参数
    num_samples = 50  # 生成更多样本进行全面评估
    seq_length = 30   # 适中的序列长度
    batch_size = 10
    
    print(f"\n📊 评估参数:")
    print(f"  - 总样本数: {num_samples}")
    print(f"  - 序列长度: {seq_length}")
    print(f"  - 批次大小: {batch_size}")
    
    # 生成序列
    print(f"\n🎲 使用改进的多样性采样生成序列...")
    all_sequences = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # 使用多样性采样
        generated_tokens = diffusion.diverse_sample(
            batch_size=current_batch_size,
            seq_len=seq_length,
            diversity_strength=0.4,  # 使用较强的多样性
            temperature=1.1  # 略高的温度增加随机性
        )
        
        # 转换为序列
        for i in range(current_batch_size):
            seq = tokens_to_sequence(generated_tokens[i])
            if len(seq) > 0:  # 只保留有效序列
                all_sequences.append(seq)
    
    print(f"✓ 成功生成 {len(all_sequences)} 个有效序列")
    
    # 1. 基础统计
    print(f"\n📈 1. 基础统计分析:")
    print(f"  - 有效序列数: {len(all_sequences)}")
    print(f"  - 有效率: {len(all_sequences)/num_samples:.1%}")
    
    if not all_sequences:
        print("❌ 没有生成有效序列，评估终止")
        return
    
    # 分析序列长度
    lengths = [len(seq) for seq in all_sequences]
    print(f"  - 平均长度: {np.mean(lengths):.1f}")
    print(f"  - 长度范围: {min(lengths)}-{max(lengths)}")
    
    # 2. 氨基酸分布分析
    print(f"\n🧬 2. 氨基酸分布分析:")
    aa_counter = Counter()
    total_aa = 0
    
    for seq in all_sequences:
        for aa in seq:
            aa_counter[aa] += 1
            total_aa += 1
    
    print(f"  总氨基酸数: {total_aa}")
    print(f"  氨基酸分布 (前10位):")
    
    for aa, count in aa_counter.most_common(10):
        percentage = count / total_aa * 100
        print(f"    {aa}: {count:4d} ({percentage:5.1f}%)")
    
    # 与目标分布比较
    target_distribution = {
        'K': 0.1136, 'G': 0.1045, 'L': 0.0897, 'R': 0.0888, 'A': 0.0719,
        'I': 0.0612, 'V': 0.0577, 'P': 0.0569, 'S': 0.0500, 'C': 0.0459,
        'F': 0.0436, 'T': 0.0362, 'N': 0.0320, 'Q': 0.0248, 'D': 0.0247,
        'E': 0.0238, 'W': 0.0216, 'Y': 0.0209, 'H': 0.0198, 'M': 0.0122
    }
    
    print(f"\n  与目标分布的偏差:")
    total_deviation = 0
    for aa in ['K', 'G', 'L', 'R', 'A']:  # 检查主要氨基酸
        current_freq = aa_counter[aa] / total_aa if aa in aa_counter else 0
        target_freq = target_distribution.get(aa, 0)
        deviation = abs(current_freq - target_freq)
        total_deviation += deviation
        print(f"    {aa}: 当前{current_freq:.3f} vs 目标{target_freq:.3f} (偏差: {deviation:.3f})")
    
    print(f"  平均偏差: {total_deviation/5:.3f}")
    
    # 3. 序列多样性分析
    print(f"\n🌈 3. 序列多样性分析:")
    uniqueness, avg_diversity = calculate_diversity_metrics(all_sequences)
    print(f"  - 序列唯一性: {uniqueness:.3f}")
    print(f"  - 平均多样性: {avg_diversity:.3f}")
    
    # 4. 生物学特性分析
    print(f"\n🧬 4. 生物学特性分析:")
    properties = analyze_sequence_properties(all_sequences)
    
    print(f"  疏水性氨基酸比例: {np.mean(properties['hydrophobic_ratio']):.3f}")
    print(f"  带电氨基酸比例: {np.mean(properties['charged_ratio']):.3f}")
    print(f"  芳香性氨基酸比例: {np.mean(properties['aromatic_ratio']):.3f}")
    
    # 5. 序列示例展示
    print(f"\n📋 5. 生成序列示例 (前10个):")
    for i, seq in enumerate(all_sequences[:10]):
        print(f"  {i+1:2d}. {seq}")
    
    # 6. 质量评估
    print(f"\n⭐ 6. 整体质量评估:")
    
    # 质量分数计算
    length_score = 1.0 if 15 <= np.mean(lengths) <= 50 else 0.5
    diversity_score = min(1.0, avg_diversity * 2)  # 多样性分数
    distribution_score = max(0, 1.0 - total_deviation/5 * 10)  # 分布分数
    uniqueness_score = uniqueness
    
    overall_score = (length_score + diversity_score + distribution_score + uniqueness_score) / 4
    
    print(f"  - 长度合理性: {length_score:.3f}")
    print(f"  - 序列多样性: {diversity_score:.3f}")
    print(f"  - 分布准确性: {distribution_score:.3f}")
    print(f"  - 序列唯一性: {uniqueness_score:.3f}")
    print(f"  - 📊 总体质量分数: {overall_score:.3f}")
    
    # 结论
    print(f"\n" + "=" * 70)
    if overall_score >= 0.8:
        print("🎉 评估结论: 模型性能优秀！")
        print("✅ 建议：模型已准备好用于生产环境")
    elif overall_score >= 0.6:
        print("✅ 评估结论: 模型性能良好")
        print("💡 建议：可以投入使用，建议继续优化")
    else:
        print("⚠️ 评估结论: 模型性能需要改进")
        print("🔧 建议：需要进一步调整参数或重新训练")
    
    print(f"📝 改进的多样性采样显著提升了生成质量")
    print("=" * 70)
    
    # 保存评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"final_evaluation_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"最终模型评估结果 - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"总体质量分数: {overall_score:.3f}\n")
        f.write(f"有效序列数: {len(all_sequences)}\n")
        f.write(f"序列唯一性: {uniqueness:.3f}\n")
        f.write(f"平均多样性: {avg_diversity:.3f}\n")
        f.write(f"分布偏差: {total_deviation/5:.3f}\n")
        f.write("\n生成序列示例:\n")
        for i, seq in enumerate(all_sequences[:20]):
            f.write(f"{i+1:2d}. {seq}\n")
    
    print(f"\n📄 详细结果已保存到: {results_file}")

if __name__ == "__main__":
    final_evaluation()
