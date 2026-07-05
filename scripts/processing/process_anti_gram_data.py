#!/usr/bin/env python3
"""
处理Anti-Gram+数据集，生成peptide_features.csv文件
用于训练抗革兰氏阴性菌分类器
"""

import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import molecular_weight
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸字典
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def calculate_aliphatic_index(sequence):
    """手动计算脂肪族指数"""
    # 脂肪族氨基酸：A, I, L, V
    aliphatic_counts = {'A': 0, 'I': 0, 'L': 0, 'V': 0}
    total_length = len(sequence)

    if total_length == 0:
        return 0.0

    for aa in sequence:
        if aa in aliphatic_counts:
            aliphatic_counts[aa] += 1

    # 脂肪族指数计算公式
    aliphatic_index = (aliphatic_counts['A'] + 2.9 * aliphatic_counts['V'] +
                      3.9 * (aliphatic_counts['I'] + aliphatic_counts['L'])) / total_length * 100

    return aliphatic_index

def calculate_hydrophilicity(sequence):
    """手动计算亲水性指数"""
    # Hopp & Woods 亲水性标度
    hydrophilicity_scale = {
        'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
        'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
        'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
        'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
    }

    if len(sequence) == 0:
        return 0.0

    total_score = sum(hydrophilicity_scale.get(aa, 0.0) for aa in sequence)
    return total_score / len(sequence)

def calculate_peptide_features(sequence):
    """计算肽段的28维特征"""
    try:
        # 清理序列，只保留标准氨基酸
        clean_seq = ''.join([aa for aa in sequence.upper() if aa in AMINO_ACIDS])
        if len(clean_seq) == 0:
            return None

        # 使用BioPython计算理化性质
        analysis = ProteinAnalysis(clean_seq)

        # 基本特征 (8维)
        features = {
            'Length': len(clean_seq),
            'Charge': analysis.charge_at_pH(7.0),
            'Hydrophobicity': analysis.gravy(),  # GRAVY值
            'Hydrophobic_Moment': 0.0,  # 简化处理，设为0
            'Instability_Index': analysis.instability_index(),
            'Isoelectric_Point': analysis.isoelectric_point(),
            'Aliphatic_Index': calculate_aliphatic_index(clean_seq),  # 手动计算
            'Hydrophilicity': calculate_hydrophilicity(clean_seq),  # 手动计算
        }

        # 氨基酸组成 (20维)
        aa_percent = analysis.get_amino_acids_percent()
        for aa in AMINO_ACIDS:
            features[f'AA_{aa}'] = aa_percent.get(aa, 0.0)

        return features

    except Exception as e:
        logger.warning(f"计算特征失败，序列: {sequence[:50]}..., 错误: {e}")
        return None

def process_fasta_file(fasta_path, label, source_name):
    """处理单个FASTA文件"""
    logger.info(f"处理文件: {fasta_path}")
    
    sequences_data = []
    
    try:
        with open(fasta_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequence = str(record.seq)
                
                # 计算特征
                features = calculate_peptide_features(sequence)
                if features is None:
                    continue
                
                # 添加基本信息
                row_data = {
                    'ID': record.id,
                    'Sequence': sequence,
                    'Label': label,
                    'Source': source_name
                }
                
                # 添加特征
                row_data.update(features)
                sequences_data.append(row_data)
                
                if len(sequences_data) % 1000 == 0:
                    logger.info(f"已处理 {len(sequences_data)} 个序列...")
    
    except Exception as e:
        logger.error(f"处理文件 {fasta_path} 时出错: {e}")
        return []
    
    logger.info(f"文件 {fasta_path} 处理完成，共 {len(sequences_data)} 个有效序列")
    return sequences_data

def main():
    """主函数"""
    logger.info("开始处理Anti-Gram+数据集...")
    
    # 数据文件路径
    data_dir = './Anti-Gram+'
    
    # 定义文件和标签映射
    file_configs = [
        {
            'path': os.path.join(data_dir, 'anti_gram-_neg_cdhit90_train80.fasta'),
            'label': 0,  # 抗革兰氏阴性菌肽
            'source': 'anti_gram_neg_train'
        },
        {
            'path': os.path.join(data_dir, 'anti_gram-_neg_cdhit90_test20.fasta'),
            'label': 0,  # 抗革兰氏阴性菌肽
            'source': 'anti_gram_neg_test'
        },
        {
            'path': os.path.join(data_dir, 'anti_gram-_pos_cdhit90_train80.fasta'),
            'label': 1,  # 抗革兰氏阳性菌肽
            'source': 'anti_gram_pos_train'
        },
        {
            'path': os.path.join(data_dir, 'anti_gram-_pos_cdhit90_test20.fasta'),
            'label': 1,  # 抗革兰氏阳性菌肽
            'source': 'anti_gram_pos_test'
        }
    ]
    
    all_data = []
    
    # 处理每个文件
    for config in file_configs:
        if os.path.exists(config['path']):
            file_data = process_fasta_file(config['path'], config['label'], config['source'])
            all_data.extend(file_data)
        else:
            logger.warning(f"文件不存在: {config['path']}")
    
    if not all_data:
        logger.error("没有处理到任何数据！")
        return
    
    # 转换为DataFrame
    logger.info("转换为DataFrame...")
    df = pd.DataFrame(all_data)
    
    # 检查数据
    logger.info(f"数据集统计:")
    logger.info(f"总样本数: {len(df)}")
    logger.info(f"标签分布:")
    logger.info(df['Label'].value_counts())
    logger.info(f"来源分布:")
    logger.info(df['Source'].value_counts())
    
    # 检查序列长度
    df['seq_length'] = df['Sequence'].str.len()
    logger.info(f"序列长度统计:")
    logger.info(f"最短: {df['seq_length'].min()}")
    logger.info(f"最长: {df['seq_length'].max()}")
    logger.info(f"平均: {df['seq_length'].mean():.1f}")
    logger.info(f"中位数: {df['seq_length'].median():.1f}")
    
    # 检查超过128长度的序列比例
    long_seqs = (df['seq_length'] > 128).sum()
    logger.info(f"超过128长度的序列: {long_seqs} ({long_seqs/len(df)*100:.1f}%)")
    
    # 删除临时列
    df = df.drop('seq_length', axis=1)
    
    # 保存到CSV
    output_path = './data/peptide_features.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"保存到: {output_path}")
    df.to_csv(output_path, index=False)
    
    # 保存特征名称
    feature_names = [col for col in df.columns if col not in ['ID', 'Sequence', 'Label', 'Source']]
    feature_names_path = './data/feature_names.txt'
    
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    logger.info(f"特征名称保存到: {feature_names_path}")
    logger.info(f"特征数量: {len(feature_names)}")
    
    logger.info("✅ 数据处理完成！")
    logger.info(f"现在可以运行: python hybrid_classifier.py")

if __name__ == '__main__':
    main()
