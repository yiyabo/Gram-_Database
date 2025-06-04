#!/usr/bin/env python3
"""
分析训练数据的氨基酸分布
诊断模型生成序列质量问题
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data_loader import AMINO_ACID_VOCAB, VOCAB_TO_AA, sequence_to_tokens, tokens_to_sequence

def analyze_amino_acid_distribution():
    """分析训练数据中的氨基酸分布"""
    print("🔍 分析训练数据的氨基酸分布...")
    
    # 读取训练数据
    sequences = []
    
    # 读取主要训练序列
    if os.path.exists("main_training_sequences.txt"):
        with open("main_training_sequences.txt", 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:
                    sequences.append(seq)
        print(f"✓ 加载主训练序列: {len(sequences)} 条")
    
    # 读取正样本序列
    if os.path.exists("positive_sequences.txt"):
        with open("positive_sequences.txt", 'r') as f:
            for line in f:
                seq = line.strip()
                if seq:
                    sequences.append(seq)
        print(f"✓ 加载正样本序列: 总计 {len(sequences)} 条")
    
    if not sequences:
        print("❌ 未找到训练数据文件")
        return
    
    # 统计氨基酸分布
    aa_counter = Counter()
    total_length = 0
    
    for seq in sequences:
        for aa in seq.upper():
            if aa in AMINO_ACID_VOCAB and aa != 'PAD':
                aa_counter[aa] += 1
                total_length += 1
    
    print(f"\n📊 氨基酸分布统计 (总计: {total_length} 个氨基酸)")
    print("=" * 50)
    
    # 按频率排序
    sorted_aas = sorted(aa_counter.items(), key=lambda x: x[1], reverse=True)
    
    for aa, count in sorted_aas:
        percentage = (count / total_length) * 100
        print(f"{aa}: {count:6d} ({percentage:5.2f}%)")
    
    # 检查是否有极端偏差
    most_common_aa, most_count = sorted_aas[0]
    most_percentage = (most_count / total_length) * 100
    
    print(f"\n⚠️  最高频氨基酸: {most_common_aa} ({most_percentage:.2f}%)")
    
    if most_percentage > 15:
        print(f"🚨 警告: {most_common_aa} 氨基酸占比过高 ({most_percentage:.2f}%)，可能导致生成偏差")
    
    # 分析序列长度分布
    lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(lengths)
    print(f"\n📏 序列长度统计:")
    print(f"   平均长度: {avg_length:.1f}")
    print(f"   最短: {min(lengths)}")
    print(f"   最长: {max(lengths)}")
    
    # 生成可视化
    plot_amino_acid_distribution(sorted_aas, total_length)
    
    return aa_counter, sequences

def plot_amino_acid_distribution(sorted_aas, total_length):
    """绘制氨基酸分布图"""
    aas = [item[0] for item in sorted_aas]
    counts = [item[1] for item in sorted_aas]
    percentages = [(count / total_length) * 100 for count in counts]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(aas, percentages)
    
    # 标出超过平均值的氨基酸
    avg_percentage = 100 / 20  # 理论平均值 (20种氨基酸)
    for i, (aa, percentage) in enumerate(zip(aas, percentages)):
        if percentage > avg_percentage * 1.5:  # 超过平均值1.5倍
            bars[i].set_color('red')
        elif percentage > avg_percentage:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('lightblue')
    
    plt.title('训练数据中的氨基酸分布')
    plt.xlabel('氨基酸')
    plt.ylabel('占比 (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=avg_percentage, color='black', linestyle='--', alpha=0.5, label='理论平均值')
    plt.legend()
    plt.tight_layout()
    plt.savefig('amino_acid_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ 氨基酸分布图已保存: amino_acid_distribution.png")

def analyze_generated_sequences():
    """分析生成序列的问题"""
    print("\n🔍 分析生成序列的问题...")
    
    # 从评估结果中读取生成的序列
    generated_sequences = [
        "DVGGGGGGRGYGGGGGGKGGGGGGGGGGGGLGGGGLGGKGGLGGGGLGGG",
        "DVGLGGGGGGKGGGGGGGGGGLGGGLGGGGLGGGGLGGGGGLGGGGLLGG",
        "DGYKGGGGRGYGGGGGGGGGGYGGGLGGGGLGGGGGGGGGGLGGGGLGGG",
        "DVGYGGGGGLYGGGGGGGGGGGGGGLDGGGGGGGGLGGGGGGGGGGLGGG",
        "DVYGGGGGRYGGGGGGGGGGGYGGGGGGGGLGGGGLGGGGGGGGGGGGGG",
        "DVYGGGGGGLGGGGGGGGGGGLGGGGGGGGLGGGGLGGGGGLGGGGLGGG",
        "DGYGGGGGGYGGGGRGGGGGGLGGGGGGGGGGGGGGGGGGGLGGGGLGGG",
        "DVYVGGGGGGYGGGGGGGGGGLGGGGGGGGLGGGGLGGGGGLGGGGLGGG",
        "DVRGGGGGGLGGGGLYGGGGGLGGGGGGGGGGGGGLGGGGGLGGGGLGGG",
        "DGYGDGGGGGYGGGGGGGGGGYGGGGGGGGGGGGGLGLGKGGGGGGLGGG"
    ]
    
    # 统计生成序列的氨基酸分布
    gen_aa_counter = Counter()
    gen_total = 0
    
    for seq in generated_sequences:
        for aa in seq:
            gen_aa_counter[aa] += 1
            gen_total += 1
    
    print(f"生成序列氨基酸分布 (总计: {gen_total} 个氨基酸):")
    print("=" * 40)
    
    sorted_gen_aas = sorted(gen_aa_counter.items(), key=lambda x: x[1], reverse=True)
    for aa, count in sorted_gen_aas:
        percentage = (count / gen_total) * 100
        print(f"{aa}: {count:4d} ({percentage:5.2f}%)")
    
    # 检查G的占比
    g_percentage = (gen_aa_counter.get('G', 0) / gen_total) * 100
    print(f"\n🚨 甘氨酸(G)占比: {g_percentage:.1f}%")
    
    if g_percentage > 30:
        print("⚠️  甘氨酸占比过高，这是主要问题！")

def main():
    """主函数"""
    import os
    
    print("=" * 60)
    print("🔬 训练数据与生成序列质量分析")
    print("=" * 60)
    
    # 分析训练数据
    aa_counter, sequences = analyze_amino_acid_distribution()
    
    # 分析生成序列
    analyze_generated_sequences()
    
    print("\n" + "=" * 60)
    print("🎯 问题诊断与建议")
    print("=" * 60)
    print("1. 检查训练数据是否有氨基酸分布偏差")
    print("2. 考虑在损失函数中加入多样性正则化")
    print("3. 调整扩散模型的采样策略")
    print("4. 可能需要重新平衡训练数据")

if __name__ == "__main__":
    main()
