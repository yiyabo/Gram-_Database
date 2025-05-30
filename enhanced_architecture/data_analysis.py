#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集分析脚本
分析Gram-和Gram+-数据集的关系，为对比学习做准备
"""

import os
import pandas as pd
from collections import Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_fasta(file_path):
    """解析FASTA文件"""
    sequences = []
    headers = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        current_seq = ""
        current_header = ""
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    headers.append(current_header)
                current_header = line[1:]  # 去掉>
                current_seq = ""
            else:
                current_seq += line
        
        # 添加最后一个序列
        if current_seq:
            sequences.append(current_seq)
            headers.append(current_header)
    
    return headers, sequences

def analyze_datasets():
    """分析数据集"""
    # 数据路径
    gram_neg_path = "/Users/apple/AIBD/Gram-_Database/data/Gram-.fasta"        # 只抗阴性菌
    gram_both_path = "/Users/apple/AIBD/Gram-_Database/data/Gram+-.fasta"      # 既抗阳性又抗阴性
    gram_pos_path = "/Users/apple/AIBD/Gram-_Database/data/Gram+.fasta"        # 只抗阳性菌
    
    logger.info("开始分析三个数据集...")
    
    # 解析数据
    headers_neg, seqs_neg = parse_fasta(gram_neg_path)
    headers_both, seqs_both = parse_fasta(gram_both_path)
    headers_pos, seqs_pos = parse_fasta(gram_pos_path)
    
    logger.info(f"Gram- (只抗阴性菌): {len(seqs_neg)} 条序列")
    logger.info(f"Gram+- (既抗阳性又抗阴性): {len(seqs_both)} 条序列")
    logger.info(f"Gram+ (只抗阳性菌): {len(seqs_pos)} 条序列")
    
    # 转换为集合便于分析
    set_neg = set(seqs_neg)
    set_both = set(seqs_both)
    set_pos = set(seqs_pos)
    
    # 分析重叠情况
    overlap_neg_both = set_neg.intersection(set_both)
    overlap_pos_both = set_pos.intersection(set_both)
    overlap_neg_pos = set_neg.intersection(set_pos)
    
    logger.info(f"\n=== 数据集重叠分析 ===")
    logger.info(f"只抗阴性菌序列: {len(set_neg)}")
    logger.info(f"广谱抗菌序列: {len(set_both)}")
    logger.info(f"只抗阳性菌序列: {len(set_pos)}")
    logger.info(f"阴性-广谱重叠: {len(overlap_neg_both)} ({len(overlap_neg_both)/len(set_neg)*100:.1f}%)")
    logger.info(f"阳性-广谱重叠: {len(overlap_pos_both)} ({len(overlap_pos_both)/len(set_pos)*100:.1f}%)")
    logger.info(f"阴性-阳性重叠: {len(overlap_neg_pos)} (应该为0)")
    
    # 对比学习数据集构建
    # 正样本：所有抗阴性菌序列 (Gram- + Gram+-)
    positive_seqs = list(set_neg.union(set_both))
    
    # 负样本：只抗阳性菌序列 (Gram+)
    negative_seqs = list(set_pos)
    
    logger.info(f"\n=== 对比学习数据集 ===")
    logger.info(f"正样本（抗阴性菌）: {len(positive_seqs)} 条")
    logger.info(f"  - 只抗阴性: {len(set_neg)} 条")
    logger.info(f"  - 广谱抗菌: {len(set_both)} 条")
    logger.info(f"负样本（只抗阳性菌）: {len(negative_seqs)} 条")
    logger.info(f"正负比例: 1:{len(negative_seqs)/len(positive_seqs):.2f}")
    
    # 序列长度统计
    len_neg = [len(seq) for seq in seqs_neg]
    len_both = [len(seq) for seq in seqs_both]
    len_pos = [len(seq) for seq in seqs_pos]
    
    logger.info(f"\n=== 序列长度统计 ===")
    logger.info(f"只抗阴性菌平均长度: {sum(len_neg)/len(len_neg):.1f} ± {(sum([(x-sum(len_neg)/len(len_neg))**2 for x in len_neg])/len(len_neg))**0.5:.1f}")
    logger.info(f"广谱抗菌平均长度: {sum(len_both)/len(len_both):.1f} ± {(sum([(x-sum(len_both)/len(len_both))**2 for x in len_both])/len(len_both))**0.5:.1f}")
    logger.info(f"只抗阳性菌平均长度: {sum(len_pos)/len(len_pos):.1f} ± {(sum([(x-sum(len_pos)/len(len_pos))**2 for x in len_pos])/len(len_pos))**0.5:.1f}")
    
    # 氨基酸组成分析
    def get_aa_composition(sequences):
        all_aas = ''.join(sequences)
        total = len(all_aas)
        composition = Counter(all_aas)
        return {aa: count/total for aa, count in composition.items()}
    
    comp_pos = get_aa_composition(positive_seqs)  # 抗阴性菌
    comp_neg = get_aa_composition(negative_seqs)  # 只抗阳性菌
    
    logger.info(f"\n=== 氨基酸组成差异 (Top 5) ===")
    logger.info("抗阴性菌序列 vs 只抗阳性菌序列的组成差异:")
    
    aa_diff = {}
    for aa in comp_pos:
        if aa in comp_neg:
            aa_diff[aa] = comp_pos[aa] - comp_neg[aa]
    
    # 显示差异最大的氨基酸
    sorted_diff = sorted(aa_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for aa, diff in sorted_diff:
        logger.info(f"{aa}: 抗阴性{comp_pos[aa]:.3f} vs 只抗阳性{comp_neg.get(aa, 0):.3f} (差异: {diff:+.3f})")
    
    # 保存分析结果
    results = {
        'gram_neg_only': list(set_neg),          # 只抗阴性菌
        'gram_both': list(set_both),             # 广谱抗菌
        'gram_pos_only': list(set_pos),          # 只抗阳性菌
        'positive_sequences': positive_seqs,      # 对比学习正样本
        'negative_sequences': negative_seqs,      # 对比学习负样本
        'stats': {
            'gram_neg_count': len(set_neg),
            'gram_both_count': len(set_both),
            'gram_pos_count': len(set_pos),
            'positive_count': len(positive_seqs),
            'negative_count': len(negative_seqs),
            'pos_neg_ratio': len(negative_seqs)/len(positive_seqs)
        }
    }
    
    return results

def create_training_datasets(results):
    """创建训练数据集"""
    logger.info("\n=== 创建训练数据集 ===")
    
    # 正样本：所有抗革兰氏阴性菌序列 (Gram- + Gram+-)
    positive_seqs = results['positive_sequences']
    
    # 负样本：只抗阳性菌序列 (Gram+)
    negative_seqs = results['negative_sequences']
    
    # 主训练集：只用Gram-序列进行diffusion训练
    main_training_seqs = results['gram_neg_only']
    
    logger.info(f"主训练集（只抗阴性菌）: {len(main_training_seqs)} 条")
    logger.info(f"正样本（所有抗阴性菌）: {len(positive_seqs)} 条")
    logger.info(f"负样本（只抗阳性菌）: {len(negative_seqs)} 条")
    logger.info(f"正负比例: 1:{len(negative_seqs)/len(positive_seqs):.2f}")
    
    # 保存训练集
    with open('/Users/apple/AIBD/Gram-_Database/enhanced_architecture/main_training_sequences.txt', 'w') as f:
        for seq in main_training_seqs:
            f.write(seq + '\n')
    
    with open('/Users/apple/AIBD/Gram-_Database/enhanced_architecture/positive_sequences.txt', 'w') as f:
        for seq in positive_seqs:
            f.write(seq + '\n')
    
    with open('/Users/apple/AIBD/Gram-_Database/enhanced_architecture/negative_sequences.txt', 'w') as f:
        for seq in negative_seqs:
            f.write(seq + '\n')
    
    # 额外保存三个原始数据集
    with open('/Users/apple/AIBD/Gram-_Database/enhanced_architecture/gram_neg_only.txt', 'w') as f:
        for seq in results['gram_neg_only']:
            f.write(seq + '\n')
    
    with open('/Users/apple/AIBD/Gram-_Database/enhanced_architecture/gram_both.txt', 'w') as f:
        for seq in results['gram_both']:
            f.write(seq + '\n')
    
    with open('/Users/apple/AIBD/Gram-_Database/enhanced_architecture/gram_pos_only.txt', 'w') as f:
        for seq in results['gram_pos_only']:
            f.write(seq + '\n')
    
    logger.info("训练数据集已保存到 enhanced_architecture/ 目录")
    logger.info("文件列表:")
    logger.info("  - main_training_sequences.txt: 主训练集（只抗阴性菌）")
    logger.info("  - positive_sequences.txt: 对比学习正样本（所有抗阴性菌）")
    logger.info("  - negative_sequences.txt: 对比学习负样本（只抗阳性菌）")
    
    return main_training_seqs, positive_seqs, negative_seqs

if __name__ == "__main__":
    # 分析数据集
    results = analyze_datasets()
    
    # 创建训练数据集
    main_seqs, positive_seqs, negative_seqs = create_training_datasets(results)
    
    logger.info("\n✅ 数据分析完成！")
    logger.info("\n=== 推荐的训练策略 ===")
    logger.info(f"🎯 主路径(Diffusion): 使用{len(main_seqs)}条只抗阴性菌序列")
    logger.info(f"🔄 辅助路径(ESM-2): 对比学习")
    logger.info(f"   ├─ 正样本: {len(positive_seqs)}条抗阴性菌序列")
    logger.info(f"   └─ 负样本: {len(negative_seqs)}条只抗阳性菌序列")
    logger.info(f"📊 数据构成:")
    logger.info(f"   ├─ 只抗阴性菌: {results['stats']['gram_neg_count']}条")
    logger.info(f"   ├─ 广谱抗菌: {results['stats']['gram_both_count']}条")
    logger.info(f"   └─ 只抗阳性菌: {results['stats']['gram_pos_count']}条")
