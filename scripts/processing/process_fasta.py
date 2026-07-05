#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理FASTA文件并提取特征用于双曲空间模型训练
"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from peptides import Peptide
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features_from_fasta(fasta_file):
    """从FASTA文件中提取肽段特征
    
    Args:
        fasta_file: FASTA文件路径
        
    Returns:
        pandas.DataFrame: 包含特征的数据框
    """
    logger.info(f"从 {fasta_file} 提取特征...")
    records = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        peptide_id = record.id
        sequence = str(record.seq)
        
        # 跳过无效序列
        if not sequence or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
            continue
            
        try:
            # 使用peptides库计算特征
            pep = Peptide(sequence)
            
            # 基本特征
            length = len(sequence)
            charge = pep.charge(pH=7.4)
            hydrophobicity = pep.hydrophobicity(scale="Eisenberg")
            hydrophobic_moment = pep.hydrophobic_moment(window=11) or 0
            
            # 额外特征
            instability_index = pep.instability_index()
            isoelectric_point = pep.isoelectric_point()
            aliphatic_index = pep.aliphatic_index()
            hydrophilicity = pep.hydrophobicity(scale="HoppWoods")
            
            # 氨基酸组成
            aa_counts = {aa: sequence.count(aa)/length for aa in 'ACDEFGHIKLMNPQRSTVWY'}
            
            # 创建特征字典
            feature_dict = {
                'ID': peptide_id,
                'Sequence': sequence,
                'Length': length,
                'Charge': charge,
                'Hydrophobicity': hydrophobicity,
                'Hydrophobic_Moment': hydrophobic_moment,
                'Instability_Index': instability_index,
                'Isoelectric_Point': isoelectric_point,
                'Aliphatic_Index': aliphatic_index,
                'Hydrophilicity': hydrophilicity
            }
            
            # 添加氨基酸组成特征
            feature_dict.update({f'AA_{aa}': count for aa, count in aa_counts.items()})
            
            records.append(feature_dict)
            
        except Exception as e:
            logger.warning(f"处理序列 {peptide_id} 时出错: {e}")
    
    # 创建数据框
    df = pd.DataFrame(records)
    logger.info(f"成功提取 {len(df)} 条序列的特征")
    return df

def prepare_data_for_model(df, test_size=0.2, random_state=42):
    """准备模型训练数据
    
    Args:
        df: 特征数据框
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # 为了演示，我们将根据电荷和疏水性创建一个二分类标签
    # 实际应用中，您可能有真实的标签或使用其他方法
    logger.info("准备模型训练数据...")
    
    # 原始序列作为正例，标签为1
    df['Label'] = 1
    logger.info(f"原始序列数量: {len(df)}，全部标记为正例（Label=1）")
    
    # 生成负例 - 通过随机打乱氧基酸序列
    import random
    negative_samples = []
    
    for _, row in df.iterrows():
        # 复制原始行数据
        neg_row = row.copy()
        
        # 获取原始序列并随机打乱
        orig_seq = row['Sequence']
        seq_list = list(orig_seq)
        random.shuffle(seq_list)
        shuffled_seq = ''.join(seq_list)
        
        # 重新计算打乱序列的特征
        try:
            pep = Peptide(shuffled_seq)
            
            # 基本特征
            neg_row['Sequence'] = shuffled_seq
            neg_row['Charge'] = pep.charge(pH=7.4)
            neg_row['Hydrophobicity'] = pep.hydrophobicity(scale="Eisenberg")
            neg_row['Hydrophobic_Moment'] = pep.hydrophobic_moment(window=11) or 0
            neg_row['Instability_Index'] = pep.instability_index()
            neg_row['Isoelectric_Point'] = pep.isoelectric_point()
            neg_row['Aliphatic_Index'] = pep.aliphatic_index()
            neg_row['Hydrophilicity'] = pep.hydrophobicity(scale="HoppWoods")
            
            # 氧基酸组成特征保持不变，因为我们只是打乱了序列顺序
            
            # 标记为负例
            neg_row['Label'] = 0
            neg_row['ID'] = f"NEG_{row['ID']}"
            
            negative_samples.append(neg_row)
        except Exception as e:
            logger.warning(f"为序列 {row['ID']} 生成负例时出错: {e}")
    
    # 将负例添加到数据集
    neg_df = pd.DataFrame(negative_samples)
    logger.info(f"生成负例数量: {len(neg_df)}")
    
    # 合并正例和负例
    df = pd.concat([df, neg_df], ignore_index=True)
    logger.info(f"合并后数据集大小: {len(df)}, 正例比例: {len(df[df['Label']==1])/len(df):.2f}")
    
    # 打乱数据集
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 选择特征列
    feature_cols = [col for col in df.columns if col not in ['ID', 'Sequence', 'Label']]
    
    # 处理缺失值
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # 分割数据
    X = df[feature_cols].values
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"数据准备完成。训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
    logger.info(f"正例比例 - 训练集: {np.mean(y_train):.2f}, 测试集: {np.mean(y_test):.2f}")
    
    # 保存处理后的数据
    os.makedirs('./data', exist_ok=True)
    np.save('./data/X_train.npy', X_train_scaled)
    np.save('./data/X_test.npy', X_test_scaled)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
    
    # 保存特征名称以便解释
    with open('./data/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

if __name__ == "__main__":
    # 处理FASTA文件
    fasta_file = "./data/Gram.fasta"
    df = extract_features_from_fasta(fasta_file)
    
    # 保存原始特征数据
    df.to_csv('./data/peptide_features.csv', index=False)
    logger.info("原始特征已保存到 ./data/peptide_features.csv")
    
    # 准备模型数据
    X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_model(df)
    
    logger.info("数据处理完成，可以运行 enhanced_hyperbolic.py 进行模型训练")
