#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理革兰氏分类数据，为抗革兰氏阴性菌分类任务准备数据集
"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from peptides import Peptide
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import random

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

def extract_features_from_uniport(uniport_file, max_samples=2000):
    """从Uniport文件中提取肽段特征
    
    Args:
        uniport_file: Uniport文件路径
        max_samples: 最大样本数
        
    Returns:
        pandas.DataFrame: 包含特征的数据框
    """
    logger.info(f"从 {uniport_file} 提取特征...")
    
    # 读取Uniport文件
    try:
        uniport_df = pd.read_csv(uniport_file, sep='\t')
        logger.info(f"Uniport文件包含 {len(uniport_df)} 条序列")
    except Exception as e:
        logger.error(f"读取Uniport文件出错: {e}")
        return pd.DataFrame()
    
    # 随机采样，避免处理过多数据
    if len(uniport_df) > max_samples:
        uniport_df = uniport_df.sample(max_samples, random_state=42)
        logger.info(f"从Uniport文件中随机采样 {max_samples} 条序列")
    
    records = []
    
    for _, row in uniport_df.iterrows():
        try:
            entry = row['Entry']
            sequence = row['Sequence']
            
            # 跳过过短或无效序列
            if len(sequence) < 5 or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
                continue
                
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
                'ID': entry,
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
            logger.warning(f"处理Uniport序列 {entry if 'entry' in locals() else 'unknown'} 时出错: {e}")
    
    # 创建数据框
    df = pd.DataFrame(records)
    logger.info(f"成功提取 {len(df)} 条Uniport序列的特征")
    return df

def prepare_gram_negative_classification_data(gram_neg_file, gram_pos_file, gram_both_file, uniport_file=None):
    """准备抗革兰氏阴性菌分类数据
    
    Args:
        gram_neg_file: 仅抗阴性肽段文件
        gram_pos_file: 仅抗阳性肽段文件
        gram_both_file: 同时抗阳阴性肽段文件
        uniport_file: 与革兰氏无关的肽段文件
        
    Returns:
        pandas.DataFrame: 处理后的数据集
    """
    # 加载各类数据
    gram_neg_df = extract_features_from_fasta(gram_neg_file)
    gram_pos_df = extract_features_from_fasta(gram_pos_file)
    gram_both_df = extract_features_from_fasta(gram_both_file)
    
    # 标记数据类别
    gram_neg_df['Label'] = 1  # 抗阴性为正例
    gram_pos_df['Label'] = 0  # 仅抗阳性为负例
    gram_both_df['Label'] = 1  # 同时抗阳阴性也标记为正例
    
    # 添加来源标记，便于后续分析
    gram_neg_df['Source'] = 'gram_neg_only'
    gram_pos_df['Source'] = 'gram_pos_only'
    gram_both_df['Source'] = 'gram_both'
    
    logger.info(f"仅抗阴性: {len(gram_neg_df)} 条，仅抗阳性: {len(gram_pos_df)} 条，同时抗阳阴性: {len(gram_both_df)} 条")
    
    # 计算正例和负例数量
    pos_examples = len(gram_neg_df) + len(gram_both_df)
    neg_examples = len(gram_pos_df)
    
    logger.info(f"初始正例数量: {pos_examples}，初始负例数量: {neg_examples}")
    
    # 如果有uniport数据，添加到负例中
    if uniport_file and os.path.exists(uniport_file):
        # 计算需要的uniport样本数
        # 设置一个最大比例，避免负例过多
        target_ratio = 1.0  # 正例:负例 = 1:1
        needed_neg = max(0, int(pos_examples * target_ratio) - neg_examples)
        
        logger.info(f"为了平衡数据集，将从uniport添加 {needed_neg} 条负例")
        
        if needed_neg > 0:
            uniport_df = extract_features_from_uniport(uniport_file, max_samples=needed_neg)
            if len(uniport_df) > 0:
                uniport_df['Label'] = 0  # 非抗菌肽为负例
                uniport_df['Source'] = 'uniport'
                
                # 合并数据
                all_data = pd.concat([gram_neg_df, gram_pos_df, gram_both_df, uniport_df])
                logger.info(f"添加了 {len(uniport_df)} 条uniport负例")
            else:
                all_data = pd.concat([gram_neg_df, gram_pos_df, gram_both_df])
        else:
            all_data = pd.concat([gram_neg_df, gram_pos_df, gram_both_df])
    else:
        all_data = pd.concat([gram_neg_df, gram_pos_df, gram_both_df])
    
    # 打乱数据
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算最终的正例和负例比例
    pos_ratio = len(all_data[all_data['Label'] == 1]) / len(all_data)
    logger.info(f"最终数据集大小: {len(all_data)}，正例比例: {pos_ratio:.2f}")
    
    return all_data

def train_test_split_and_save(df, test_size=0.2, random_state=42):
    """分割数据并保存为训练和测试集
    
    Args:
        df: 数据框
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # 选择特征列
    feature_cols = [col for col in df.columns if col not in ['ID', 'Sequence', 'Label', 'Source']]
    
    # 处理缺失值
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # 分割数据
    X = df[feature_cols].values
    y = df['Label'].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 创建保存目录
    os.makedirs('./data', exist_ok=True)
    
    # 保存数据
    np.save('./data/X_train.npy', X_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
    
    # 保存特征名称，便于后续分析
    with open('./data/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # 保存完整特征数据，便于后续分析
    df.to_csv('./data/peptide_features.csv', index=False)
    
    logger.info(f"数据已保存: X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"正例比例: 训练集 {np.mean(y_train):.2f}, 测试集 {np.mean(y_test):.2f}")
    
    return X_train, X_test, y_train, y_test, feature_cols

if __name__ == "__main__":
    # 数据文件路径
    gram_neg_file = './data/Gram-.fasta'
    gram_pos_file = './data/Gram+.fasta'
    gram_both_file = './data/Gram+-.fasta'
    uniport_file = './data/uniport'
    
    # 准备革兰氏阴性菌分类数据
    df = prepare_gram_negative_classification_data(
        gram_neg_file=gram_neg_file,
        gram_pos_file=gram_pos_file,
        gram_both_file=gram_both_file,
        uniport_file=uniport_file
    )
    
    # 分割数据并保存
    X_train, X_test, y_train, y_test, feature_names = train_test_split_and_save(df)
    
    logger.info("数据处理完成，可以运行enhanced_hyperbolic.py进行模型训练")
