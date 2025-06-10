#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gram-Negative Bacteria Prediction Service
Provides a web interface to predict if peptide sequences are active against Gram-negative bacteria.
Uses a hybrid Keras model (LSTM+MLP).
"""

import os
import sys
import uuid
import json
import tempfile
import tensorflow as tf
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import statistics
from Bio import SeqIO
from peptides import Peptide 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from collections import Counter
import pickle
import traceback # For detailed error logging
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 导入生成服务
from generation_service import get_generation_service

print("!!!!!!!!!! DEBUG: app.py IS LOADING FRESHLY (Hybrid Model Version) !!!!!!!!!!")

# 处理代理设置问题
import os
proxy_backup = {}
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    if proxy_var in os.environ:
        proxy_backup[proxy_var] = os.environ[proxy_var]
        # 临时禁用代理以避免Hugging Face连接问题
        del os.environ[proxy_var]
        print(f"临时禁用代理: {proxy_var}")

# 创建图表保存目录
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# --- Constants and functions copied/adapted from hybrid_classifier.py ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)

MAX_SEQUENCE_LENGTH = 32 
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

def tokenize_and_pad_sequences_app(sequences, vocab_dict, max_len, pad_token_id, unk_token_id):
    """Converts a list of raw amino acid sequence strings to tokenized and padded integer ID sequences."""
    tokenized_sequences = []
    for seq_str in sequences:
        seq_str = str(seq_str).upper()
        tokens = [vocab_dict.get(aa, unk_token_id) for aa in seq_str]
        tokenized_sequences.append(tokens)
    
    padded_sequences = pad_sequences(
        tokenized_sequences,
        maxlen=max_len,
        dtype='int32',
        padding='post',
        truncating='post',
        value=pad_token_id
    )
    return padded_sequences
# --- End of copied/adapted section ---


def load_keras_model(model_path):
    """Loads a trained Keras model."""
    print(f"Loading Keras model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"Error: Keras model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Unknown error occurred while loading Keras model: {e}")
        raise

def load_app_scaler(scaler_path):
    """Loads the StandardScaler for global features."""
    print(f"Attempting to load StandardScaler from: {scaler_path}")
    if not os.path.exists(scaler_path):
        error_msg = f"Error: Required StandardScaler file '{scaler_path}' not found. Cannot proceed with predictions."
        print(error_msg)
        raise FileNotFoundError(error_msg)
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Successfully loaded StandardScaler from {scaler_path}.")
        return scaler
    except Exception as e:
        print(f"Error loading StandardScaler from {scaler_path}: {e}")
        raise RuntimeError(f"Could not load StandardScaler file '{scaler_path}': {e}")

def extract_features_from_fasta(fasta_file):
    """Extracts peptide features and raw sequences from a FASTA file."""
    print(f"Extracting features from {fasta_file}...")
    records_data = [] 
    
    feature_names_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'feature_names.txt')
    expected_feature_order = []
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f_names:
            expected_feature_order = [line.strip() for line in f_names if line.strip()]
        if len(expected_feature_order) != 28:
            print(f"Warning: Number of features in feature_names.txt ({len(expected_feature_order)}) is not 28. Will use default order.")
            expected_feature_order = [] 
    else:
        print(f"Warning: {feature_names_path} not found. Will use default feature order.")

    if not expected_feature_order: 
        expected_feature_order = [
            'Length', 'Charge', 'Hydrophobicity', 'Hydrophobic_Moment',
            'Instability_Index', 'Isoelectric_Point', 'Aliphatic_Index',
            'Hydrophilicity', 'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G',
            'AA_H', 'AA_I', 'AA_K', 'AA_L', 'AA_M', 'AA_N', 'AA_P', 'AA_Q',
            'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y'
        ]

    for record in SeqIO.parse(fasta_file, "fasta"):
        peptide_id = record.id
        sequence = str(record.seq).upper()
        
        if not sequence or any(aa not in AMINO_ACIDS for aa in sequence):
            print(f"Skipping invalid sequence {peptide_id}: {sequence}")
            continue
            
        try:
            pep = Peptide(sequence)
            length = len(sequence)
            
            feature_values_dict = {
                'ID': peptide_id,
                'Sequence': sequence,
                'Length': float(length),
                'Charge': float(pep.charge(pH=7.4)),
                'Hydrophobicity': float(pep.hydrophobicity(scale="Eisenberg")),
                'Hydrophobic_Moment': float(pep.hydrophobic_moment(window=min(11,length)) or 0.0),
                'Instability_Index': float(pep.instability_index()),
                'Isoelectric_Point': float(pep.isoelectric_point()),
                'Aliphatic_Index': float(pep.aliphatic_index()),
                'Hydrophilicity': float(pep.hydrophobicity(scale="HoppWoods"))
            }
            aa_comp = {f'AA_{aa}': sequence.count(aa) / length for aa in AMINO_ACIDS}
            feature_values_dict.update(aa_comp)
            
            ordered_feature_dict = {'ID': peptide_id, 'Sequence': sequence}
            for fname in expected_feature_order:
                ordered_feature_dict[fname] = feature_values_dict.get(fname, 0.0)
            
            records_data.append(ordered_feature_dict)
            
        except Exception as e:
            print(f"Error processing sequence {peptide_id}: {e}")
    
    df = pd.DataFrame(records_data)
    if not df.empty:
        cols_to_keep = ['ID', 'Sequence'] + [col for col in expected_feature_order if col in df.columns]
        df = df[cols_to_keep]

    print(f"Successfully extracted features and raw sequences for {len(df)} sequences.")
    return df

def predict_with_hybrid_model_app(k_model, scaler, sequence_ids_list, sequence_strings_list, global_features_df, threshold=0.8):
    """Makes predictions using the loaded Keras hybrid model and scaler."""
    print("!!!!!!!!!! DEBUG: predict_with_hybrid_model_app FUNCTION HAS BEEN CALLED !!!!!!!!!!")
    if not sequence_ids_list:
        print("No sequences to predict.")
        return pd.DataFrame(columns=['ID', 'Sequence', 'Probability', 'Prediction', 'Label'])

    print(f"Processing {len(sequence_strings_list)} sequences for LSTM branch...")
    tokenized_padded_sequences = tokenize_and_pad_sequences_app(
        sequence_strings_list, VOCAB_DICT, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID, UNK_TOKEN_ID
    )
    print(f"Sequence data processed, shape: {tokenized_padded_sequences.shape}")

    feature_column_names = [col for col in global_features_df.columns if col not in ['ID', 'Sequence']]
    global_features_np = global_features_df[feature_column_names].values.astype(np.float32)
    print(f"Global features extracted, shape: {global_features_np.shape}")
    
    scaled_global_features_np = global_features_np
    if scaler:
        try:
            if global_features_np.shape[1] != scaler.n_features_in_:
                 print(f"Error: Number of features for scaler ({global_features_np.shape[1]}) does not match scaler's expected features ({scaler.n_features_in_}).")
                 error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "Feature dimension error"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
                 return pd.DataFrame(error_results)
            scaled_global_features_np = scaler.transform(global_features_np)
            print("Global features successfully standardized.")
        except Exception as e:
            print(f"Error applying scaler: {e}. Using non-standardized features.")
    else:
        print("Warning: Scaler not provided. Using raw global feature values for prediction.")

    print(f"Predicting for {len(sequence_ids_list)} sequences...")
    if tokenized_padded_sequences.shape[0] == 0 or scaled_global_features_np.shape[0] == 0:
        print("Warning: Processed sequence data or global feature data is empty. Cannot predict.")
        return pd.DataFrame(columns=['ID', 'Sequence', 'Probability', 'Prediction', 'Label'])
    
    if tokenized_padded_sequences.shape[0] != scaled_global_features_np.shape[0]:
        print(f"Error: Mismatch in sample count between sequence data ({tokenized_padded_sequences.shape[0]}) and global feature data ({scaled_global_features_np.shape[0]}).")
        error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "Data mismatch"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
        return pd.DataFrame(error_results)

    try:
        logits = k_model.predict([tokenized_padded_sequences, scaled_global_features_np])
        probabilities = tf.sigmoid(logits).numpy().flatten()
        predictions = (probabilities >= threshold).astype(int)
    except Exception as e:
        print(f"Error during Keras model prediction: {e}")
        error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "Prediction failed"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
        return pd.DataFrame(error_results)

    results_list = []
    for i, seq_id in enumerate(sequence_ids_list):
        results_list.append({
            'ID': seq_id,
            'Sequence': sequence_strings_list[i],
            'Probability': float(probabilities[i]),
            'Prediction': int(predictions[i]),
            'Label': "Anti-Gram-Negative" if predictions[i] == 1 else "Non-Anti-Gram-Negative"
        })
    results_df = pd.DataFrame(results_list)
    
    if not results_df.empty:
        prediction_counts = Counter(results_df['Label'])
        total = len(results_df)
        print("Prediction summary:")
        for label, count in prediction_counts.items():
            percentage = count / total * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
    return results_df

def calculate_box_plot_stats(data):
    """计算箱线图统计数据（最小值、Q1、中位数、Q3、最大值）"""
    if len(data) == 0:
        return {'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0}
    
    # 过滤掉无效值
    valid_data = [x for x in data if not (np.isnan(x) or np.isinf(x))]
    if len(valid_data) == 0:
        return {'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0}
    
    # 排序数据
    sorted_data = sorted(valid_data)
    
    # 计算统计值
    min_val = float(sorted_data[0])
    max_val = float(sorted_data[-1])
    median = float(statistics.median(sorted_data))
    
    # 计算四分位数
    q1 = float(np.percentile(sorted_data, 25))
    q3 = float(np.percentile(sorted_data, 75))
    
    return {
        'min': min_val,
        'q1': q1,
        'median': median,
        'q3': q3,
        'max': max_val
    }

def generate_box_plot_data(results_with_features_df):
    """为所有特征生成箱线图数据"""
    # 获取所有数值特征列（排除ID、Sequence、Probability、Prediction、Label）
    feature_columns = [col for col in results_with_features_df.columns
                      if col not in ['ID', 'Sequence', 'Probability', 'Prediction', 'Label']]
    
    box_plot_data = []
    for feature_name in feature_columns:
        feature_values = results_with_features_df[feature_name].values
        stats = calculate_box_plot_stats(feature_values)
        
        box_plot_data.append({
            'feature': feature_name,
            'stats': stats
        })
    
    return box_plot_data

def save_charts_to_files(results_with_features_df):
    """保存各种图表到文件（覆盖模式）"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 分离positive和negative数据
        positive_data = results_with_features_df[results_with_features_df['Prediction'] == 1]
        negative_data = results_with_features_df[results_with_features_df['Prediction'] == 0]
        
        if len(positive_data) == 0 and len(negative_data) == 0:
            print("Warning: No valid data for chart generation")
            return
        
        # 1. 特征对比图 (模拟前端的Z-score标准化柱状图)
        save_feature_comparison_chart(positive_data, negative_data)
        
        # 2. 雷达图
        save_radar_chart(positive_data, negative_data)
        
        # 3. 散点图
        save_scatter_plot(positive_data, negative_data)
        
        # 4. 氨基酸组成图
        save_aa_composition_chart(positive_data, negative_data)
        
        # 5. 概率分布直方图
        save_probability_histogram(results_with_features_df)
        
        # 6. 滑动窗口分析图
        save_sliding_window_chart(results_with_features_df)
        
        # 7. 降维可视化图 (PCA + t-SNE)
        save_dimensionality_reduction_charts(results_with_features_df)
        
        print(f"Charts saved to {CHARTS_DIR} (overwrite mode)")
        
    except Exception as e:
        print(f"Error saving charts: {e}")
        traceback.print_exc()

def save_feature_comparison_chart(positive_data, negative_data):
    """保存特征对比图"""
    try:
        features = ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment', 'Instability_Index', 'Isoelectric_Point', 'Aliphatic_Index']
        feature_labels = {
            'Charge': '电荷',
            'Hydrophobicity': '疏水性',
            'Hydrophobic_Moment': '疏水力矩',
            'Instability_Index': '不稳定指数',
            'Isoelectric_Point': '等电点',
            'Aliphatic_Index': '脂肪族指数'
        }
        
        # 计算标准化数据（与前端一致）
        pos_means = []
        neg_means = []
        labels = []
        
        for feature in features:
            if feature in positive_data.columns and feature in negative_data.columns:
                # 获取所有数据用于标准化
                all_values = pd.concat([positive_data[feature], negative_data[feature]]).dropna()
                if len(all_values) > 0:
                    global_mean = all_values.mean()
                    global_std = all_values.std()
                    
                    if global_std > 0:
                        pos_norm = (positive_data[feature].mean() - global_mean) / global_std
                        neg_norm = (negative_data[feature].mean() - global_mean) / global_std
                        
                        pos_means.append(pos_norm)
                        neg_means.append(neg_norm)
                        labels.append(feature_labels.get(feature, feature))
        
        if len(pos_means) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pos_means, width, label=f'抗菌肽 (n={len(positive_data)})',
                          color='#28a745', alpha=0.8)
            bars2 = ax.bar(x + width/2, neg_means, width, label=f'非抗菌肽 (n={len(negative_data)})',
                          color='#6c757d', alpha=0.8)
            
            ax.set_xlabel('特征类型', fontsize=14, fontweight='bold')
            ax.set_ylabel('Z-score标准化值', fontsize=14, fontweight='bold')
            ax.set_title('特征对比分析 (Z-score标准化)', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, 'feature_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    except Exception as e:
        print(f"Error saving feature comparison chart: {e}")

def save_radar_chart(positive_data, negative_data):
    """保存雷达图"""
    try:
        features = ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment', 'Instability_Index', 'Aliphatic_Index']
        
        # 计算标准化数据（与前端雷达图算法一致）
        pos_values = []
        neg_values = []
        valid_features = []
        
        for feature in features:
            if feature in positive_data.columns and feature in negative_data.columns:
                pos_mean = positive_data[feature].mean()
                neg_mean = negative_data[feature].mean()
                
                if pd.notna(pos_mean) and pd.notna(neg_mean):
                    # 使用相对差异标准化（与前端一致）
                    max_val = max(abs(pos_mean), abs(neg_mean))
                    diff = abs(pos_mean - neg_mean)
                    
                    if max_val > 1e-10 and diff > 1e-10:
                        center = 0.5
                        scale = 0.3
                        
                        if pos_mean > neg_mean:
                            ratio = min(diff / abs(pos_mean), 1.0)
                            pos_norm = center + scale * ratio
                            neg_norm = center - scale * ratio
                        else:
                            ratio = min(diff / abs(neg_mean), 1.0)
                            pos_norm = center - scale * ratio
                            neg_norm = center + scale * ratio
                    else:
                        pos_norm = neg_norm = 0.5
                    
                    pos_values.append(pos_norm)
                    neg_values.append(neg_norm)
                    valid_features.append(feature)
        
        if len(valid_features) > 0:
            # 创建雷达图
            angles = np.linspace(0, 2 * np.pi, len(valid_features), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            pos_values += pos_values[:1]  # 闭合
            neg_values += neg_values[:1]  # 闭合
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            ax.plot(angles, pos_values, 'o-', linewidth=2, label=f'抗菌肽 (n={len(positive_data)})', 
                   color='#28a745')
            ax.fill(angles, pos_values, alpha=0.25, color='#28a745')
            
            ax.plot(angles, neg_values, 'o-', linewidth=2, label=f'非抗菌肽 (n={len(negative_data)})', 
                   color='#6c757d')
            ax.fill(angles, neg_values, alpha=0.25, color='#6c757d')
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(valid_features)
            ax.set_ylim(0, 1)
            ax.set_title('特征雷达图对比', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, 'radar_chart.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    except Exception as e:
        print(f"Error saving radar chart: {e}")

def save_scatter_plot(positive_data, negative_data):
    """保存散点图"""
    try:
        if 'Hydrophobicity' in positive_data.columns and 'Charge' in positive_data.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制数据点
            ax.scatter(positive_data['Hydrophobicity'], positive_data['Charge'], 
                      alpha=0.7, c='#28a745', label=f'抗菌肽 (n={len(positive_data)})', s=50)
            ax.scatter(negative_data['Hydrophobicity'], negative_data['Charge'], 
                      alpha=0.7, c='#6c757d', label=f'非抗菌肽 (n={len(negative_data)})', s=50)
            
            ax.set_xlabel('疏水性', fontsize=14, fontweight='bold')
            ax.set_ylabel('电荷', fontsize=14, fontweight='bold')
            ax.set_title('疏水性 vs 电荷散点图', fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, 'scatter_plot.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    except Exception as e:
        print(f"Error saving scatter plot: {e}")

def save_aa_composition_chart(positive_data, negative_data):
    """保存氨基酸组成图"""
    try:
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        pos_freqs = []
        neg_freqs = []
        
        for aa in amino_acids:
            aa_col = f'AA_{aa}'
            if aa_col in positive_data.columns:
                pos_freq = positive_data[aa_col].mean() * 100  # 转换为百分比
                neg_freq = negative_data[aa_col].mean() * 100
                pos_freqs.append(pos_freq)
                neg_freqs.append(neg_freq)
            else:
                pos_freqs.append(0)
                neg_freqs.append(0)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(amino_acids))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pos_freqs, width, label=f'抗菌肽 (n={len(positive_data)})',
                      color='#28a745', alpha=0.8)
        bars2 = ax.bar(x + width/2, neg_freqs, width, label=f'非抗菌肽 (n={len(negative_data)})',
                      color='#6c757d', alpha=0.8)
        
        ax.set_xlabel('氨基酸', fontsize=14, fontweight='bold')
        ax.set_ylabel('频率 (%)', fontsize=14, fontweight='bold')
        ax.set_title('氨基酸组成对比', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(amino_acids)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, 'aa_composition.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"Error saving AA composition chart: {e}")

def save_probability_histogram(results_df):
    """保存概率分布直方图"""
    try:
        probabilities = results_df['Probability'].dropna()
        
        if len(probabilities) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(probabilities, bins=20, alpha=0.7, color='#007bff', edgecolor='black')
            ax.set_xlabel('预测概率', fontsize=14, fontweight='bold')
            ax.set_ylabel('序列数量', fontsize=14, fontweight='bold')
            ax.set_title('预测概率分布', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, 'probability_histogram.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
    except Exception as e:
        print(f"Error saving probability histogram: {e}")

def save_sliding_window_chart(results_df):
    """保存滑动窗口分析图"""
    try:
        # 选择一些代表性序列进行滑动窗口分析
        positive_data = results_df[results_df['Prediction'] == 1]
        negative_data = results_df[results_df['Prediction'] == 0]
        
        # 选择前几个序列作为代表
        sample_sequences = []
        if len(positive_data) > 0:
            sample_sequences.extend(positive_data.head(2).to_dict('records'))
        if len(negative_data) > 0:
            sample_sequences.extend(negative_data.head(2).to_dict('records'))
        
        if len(sample_sequences) == 0:
            print("No sequences available for sliding window analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, seq_data in enumerate(sample_sequences[:4]):
            if i >= 4:
                break
                
            sequence = seq_data['Sequence']
            seq_id = seq_data['ID']
            prediction = seq_data['Prediction']
            label = "抗菌肽" if prediction == 1 else "非抗菌肽"
            
            # 计算滑动窗口特征
            window_data = calculate_sliding_window_features(sequence, window_size=3)
            
            if len(window_data) > 0:
                positions = [w['position'] for w in window_data]
                hydrophobicity = [w['hydrophobicity'] for w in window_data]
                charge = [w['charge'] for w in window_data]
                
                ax = axes[i]
                ax.plot(positions, hydrophobicity, 'b-', label='疏水性', linewidth=2)
                ax2 = ax.twinx()
                ax2.plot(positions, charge, 'r-', label='电荷', linewidth=2)
                
                ax.set_xlabel('序列位置')
                ax.set_ylabel('疏水性', color='b')
                ax2.set_ylabel('电荷', color='r')
                ax.set_title(f'{seq_id} ({label})\n{sequence[:20]}...')
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(sample_sequences), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, 'sliding_window_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"Error saving sliding window chart: {e}")

def save_dimensionality_reduction_charts(results_df):
    """保存降维可视化图 (PCA + t-SNE)"""
    try:
        # 获取特征数据
        feature_columns = [col for col in results_df.columns 
                          if col not in ['ID', 'Sequence', 'Probability', 'Prediction', 'Label']]
        
        if len(feature_columns) == 0:
            print("No feature columns found for dimensionality reduction")
            return
        
        # 准备数据
        features_df = results_df[feature_columns].fillna(0)
        labels = results_df['Prediction'].values
        
        if len(features_df) < 2:
            print("Not enough samples for dimensionality reduction")
            return
        
        # 标准化特征
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA
        try:
            n_components = min(2, features_scaled.shape[1])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(features_scaled)
            
            positive_mask = labels == 1
            negative_mask = labels == 0
            
            ax1.scatter(pca_result[positive_mask, 0], pca_result[positive_mask, 1], 
                       c='#28a745', alpha=0.7, label=f'抗菌肽 (n={sum(positive_mask)})', s=50)
            ax1.scatter(pca_result[negative_mask, 0], pca_result[negative_mask, 1], 
                       c='#6c757d', alpha=0.7, label=f'非抗菌肽 (n={sum(negative_mask)})', s=50)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] if n_components > 1 else 0:.1%} variance)')
            ax1.set_title('PCA降维分析')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
        except Exception as pca_error:
            print(f"PCA error: {pca_error}")
            ax1.text(0.5, 0.5, 'PCA分析失败', transform=ax1.transAxes, ha='center')
        
        # t-SNE
        try:
            if len(features_scaled) > 30:  # t-SNE需要足够的样本
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_scaled)-1))
                tsne_result = tsne.fit_transform(features_scaled)
                
                ax2.scatter(tsne_result[positive_mask, 0], tsne_result[positive_mask, 1], 
                           c='#28a745', alpha=0.7, label=f'抗菌肽 (n={sum(positive_mask)})', s=50)
                ax2.scatter(tsne_result[negative_mask, 0], tsne_result[negative_mask, 1], 
                           c='#6c757d', alpha=0.7, label=f'非抗菌肽 (n={sum(negative_mask)})', s=50)
                
                ax2.set_xlabel('t-SNE维度1')
                ax2.set_ylabel('t-SNE维度2')
                ax2.set_title('t-SNE降维分析')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 't-SNE需要更多样本\n(至少30个)', transform=ax2.transAxes, ha='center')
                ax2.set_title('t-SNE降维分析')
            
        except Exception as tsne_error:
            print(f"t-SNE error: {tsne_error}")
            ax2.text(0.5, 0.5, 't-SNE分析失败', transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, 'dimensionality_reduction.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"Error saving dimensionality reduction charts: {e}")

def calculate_sliding_window_features(sequence, window_size=3):
    """计算序列滑动窗口特征"""
    from peptides import Peptide
    
    if len(sequence) < window_size:
        return []
    
    window_data = []
    for i in range(len(sequence) - window_size + 1):
        window_seq = sequence[i:i + window_size]
        try:
            pep = Peptide(window_seq)
            window_features = {
                'position': i + 1,  # 1-based position
                'sequence': window_seq,
                'hydrophobicity': float(pep.hydrophobicity(scale="Eisenberg")),
                'charge': float(pep.charge(pH=7.4)),
                'hydrophobic_moment': float(pep.hydrophobic_moment(window=min(11, len(window_seq))) or 0.0)
            }
            window_data.append(window_features)
        except Exception as e:
            # 如果计算失败，使用默认值
            window_data.append({
                'position': i + 1,
                'sequence': window_seq,
                'hydrophobicity': 0.0,
                'charge': 0.0,
                'hydrophobic_moment': 0.0
            })
    
    return window_data

def generate_sliding_window_data(results_with_features_df, window_size=3):
    """为预测结果生成滑动窗口分析数据"""
    # 只处理前10个序列以避免计算量过大
    sample_sequences = results_with_features_df.head(10)
    
    sliding_window_data = {
        'positive_samples': [],
        'negative_samples': [],
        'window_size': window_size
    }
    
    for _, row in sample_sequences.iterrows():
        sequence = row['Sequence']
        prediction = row['Prediction']
        
        window_features = calculate_sliding_window_features(sequence, window_size)
        
        sample_data = {
            'id': row['ID'],
            'sequence': sequence,
            'prediction': prediction,
            'windows': window_features
        }
        
        if prediction == 1:
            sliding_window_data['positive_samples'].append(sample_data)
        else:
            sliding_window_data['negative_samples'].append(sample_data)
    
    return sliding_window_data

def generate_dimensionality_reduction_data(results_with_features_df):
    """生成PCA和t-SNE降维可视化数据"""
    global global_feature_scaler_app
    
    # 获取数值特征列（排除ID、Sequence、Probability、Prediction、Label）
    feature_columns = [col for col in results_with_features_df.columns
                      if col not in ['ID', 'Sequence', 'Probability', 'Prediction', 'Label']]
    
    if len(feature_columns) == 0:
        return {'pca': [], 'tsne': []}
    
    # 提取特征数据
    features_data = results_with_features_df[feature_columns].values.astype(np.float32)
    
    # 检查样本数量
    n_samples = features_data.shape[0]
    if n_samples < 2:
        return {'pca': [], 'tsne': []}
    
    # 标准化特征数据
    try:
        if global_feature_scaler_app:
            scaled_features = global_feature_scaler_app.transform(features_data)
        else:
            # 如果没有预训练的scaler，使用临时scaler
            temp_scaler = StandardScaler()
            scaled_features = temp_scaler.fit_transform(features_data)
    except Exception as e:
        print(f"Error scaling features for dimensionality reduction: {e}")
        # 使用原始特征
        scaled_features = features_data
    
    dimensionality_data = {'pca': [], 'tsne': []}
    
    try:
        # 执行PCA降维
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(scaled_features)
        
        # 格式化PCA结果
        for i, row in results_with_features_df.iterrows():
            dimensionality_data['pca'].append({
                'id': row['ID'],
                'sequence': row['Sequence'],
                'prediction': int(row['Prediction']),
                'probability': float(row['Probability']),
                'label': row['Label'],
                'x': float(pca_result[i, 0]),
                'y': float(pca_result[i, 1])
            })
        
        print(f"PCA completed: explained variance ratio = {pca.explained_variance_ratio_}")
        
    except Exception as e:
        print(f"Error performing PCA: {e}")
    
    try:
        # 执行t-SNE降维
        # 动态调整perplexity以避免错误
        perplexity = min(30, max(5, n_samples // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                   n_iter=1000, learning_rate='auto')
        tsne_result = tsne.fit_transform(scaled_features)
        
        # 格式化t-SNE结果
        for i, row in results_with_features_df.iterrows():
            dimensionality_data['tsne'].append({
                'id': row['ID'],
                'sequence': row['Sequence'],
                'prediction': int(row['Prediction']),
                'probability': float(row['Probability']),
                'label': row['Label'],
                'x': float(tsne_result[i, 0]),
                'y': float(tsne_result[i, 1])
            })
        
        print(f"t-SNE completed with perplexity={perplexity}")
        
    except Exception as e:
        print(f"Error performing t-SNE: {e}")
    
    return dimensionality_data

# Initialize Flask application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Global variables
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(APP_ROOT_DIR)
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, 'model', 'hybrid_classifier_best_tuned.keras')
SCALER_PATH_APP = os.path.join(PROJECT_ROOT_DIR, 'model', 'hybrid_model_scaler.pkl')

keras_model_global = None 
global_feature_scaler_app = None 

def load_app_dependencies():
    """Loads the Keras model and scaler required by the app."""
    global keras_model_global, global_feature_scaler_app
    if keras_model_global is None:
        keras_model_global = load_keras_model(KERAS_MODEL_PATH)
    if global_feature_scaler_app is None:
        global_feature_scaler_app = load_app_scaler(SCALER_PATH_APP)
    print("Keras model and Scaler loaded for the app!")

@app.before_request
def before_first_request_func():
    # Only load models for prediction endpoints, not for database endpoint
    if request.endpoint in ['predict_sequence_api', 'generate_sequences_route']:
        if keras_model_global is None or global_feature_scaler_app is None:
            load_app_dependencies()

@app.route('/')
@app.route('/predict')
@app.route('/generate')
@app.route('/database')
@app.route('/about')
def index():
    """Serves the main layout, letting the frontend handle routing."""
    return render_template('layout.html')

# --- API Routes for Content Fragments ---

@app.route('/content/predict')
def content_predict():
    """Serves the content for the prediction page."""
    return render_template('fragments/predict.html')

@app.route('/content/generate')
def content_generate():
    """Serves the content for the generation page."""
    return render_template('fragments/generate.html')

@app.route('/content/database')
def content_database():
    """Serves the content for the database page."""
    return render_template('fragments/database.html')

@app.route('/content/about')
def content_about():
    """Serves the content for the about page."""
    return render_template('fragments/about.html')

# --- API Routes for Core Logic ---

@app.route('/api/predict', methods=['POST'])
def predict_sequence_api(): # Renamed to clarify it's an API endpoint
    global keras_model_global, global_feature_scaler_app 
    try:
        temp_fasta = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.fasta")
        
        if 'fasta_file' in request.files and request.files['fasta_file'].filename:
            file = request.files['fasta_file']
            if not file.filename.endswith(('.fasta', '.fa', '.txt')):
                return jsonify({'error': 'Please upload a FASTA format file (.fasta, .fa, .txt).'}), 400
            file.save(temp_fasta)
        elif 'fasta_text' in request.form and request.form['fasta_text'].strip():
            fasta_text = request.form['fasta_text']
            if '>' in fasta_text:
                with open(temp_fasta, 'w') as f: f.write(fasta_text)
            else:
                with open(temp_fasta, 'w') as f:
                    lines = fasta_text.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip(): f.write(f">Seq_{i+1}\n{line.strip()}\n")
        else:
            return jsonify({'error': 'Please upload a FASTA file or input sequence data.'}), 400
        
        if not os.path.exists(temp_fasta) or os.path.getsize(temp_fasta) == 0:
            return jsonify({'error': 'Failed to create temporary FASTA file or file is empty.'}), 400
            
        sequence_count = sum(1 for _ in SeqIO.parse(temp_fasta, "fasta"))
        if sequence_count == 0:
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': 'No valid sequences detected.'}), 400
        
        MAX_SEQUENCES_ALLOWED = 20000 
        if sequence_count > MAX_SEQUENCES_ALLOWED: 
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': f'Too many sequences ({sequence_count}). Please limit to {MAX_SEQUENCES_ALLOWED} or less.'}), 400
        
        all_data_df = extract_features_from_fasta(temp_fasta) 
        
        if all_data_df.empty:
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': 'Could not extract valid features from the input sequences.'}), 400
        
        sequence_ids_list = all_data_df['ID'].tolist()
        sequence_strings_list = all_data_df['Sequence'].tolist()
        global_feature_cols = [col for col in all_data_df.columns if col not in ['ID', 'Sequence']]
        global_features_input_df = all_data_df[global_feature_cols]

        results_df = predict_with_hybrid_model_app(
            k_model=keras_model_global, 
            scaler=global_feature_scaler_app, 
            sequence_ids_list=sequence_ids_list,
            sequence_strings_list=sequence_strings_list,
            global_features_df=global_features_input_df,
            threshold=0.8 
        )
        
        if os.path.exists(temp_fasta):
            try:
                os.remove(temp_fasta)
            except Exception as e_rm:
                print(f"Error deleting temporary FASTA file {temp_fasta}: {e_rm}")

        if results_df.empty and not sequence_ids_list:
             return jsonify({'error': 'No valid sequences for prediction.'}), 400
        if results_df.empty and sequence_ids_list: # Check if prediction failed for existing sequences
             # Check if all predictions are -1 (error state)
            if all(results_df['Prediction'] == -1):
                # Try to get a more specific error message if available from the Label column
                specific_error = "An error occurred during prediction"
                if 'Label' in results_df.columns and results_df['Label'].nunique() == 1:
                    specific_error = results_df['Label'].iloc[0] # e.g., "Feature dimension error"
                return jsonify({'error': f'{specific_error}, failed to generate valid results.'}), 500
            # If not all failed, proceed to format results

        # Merge prediction results back with the original features
        results_with_features_df = pd.merge(results_df, all_data_df, on=['ID', 'Sequence'])

        results_for_json = []
        for _, row in results_with_features_df.iterrows():
            # Extract all feature columns dynamically, excluding ID/Sequence which are already there
            features = {col: row[col] for col in all_data_df.columns if col not in ['ID', 'Sequence']}
            
            results_for_json.append({
                'id': row['ID'],
                'sequence': row['Sequence'],
                'probability': float(row['Probability']) if pd.notna(row['Probability']) else -1.0,
                'prediction': int(row['Prediction']) if pd.notna(row['Prediction']) else -1,
                'label': row['Label'],
                'features': features # Nest all physicochemical features here
            })
        
        valid_results_count = len(results_for_json) - sum(1 for r in results_for_json if r['prediction'] == -1)
        stats = {
            'total': len(results_for_json),
            'positive': sum(1 for r in results_for_json if r['prediction'] == 1),
            'negative': sum(1 for r in results_for_json if r['prediction'] == 0),
            'failed': sum(1 for r in results_for_json if r['prediction'] == -1),
            'avg_probability': float(np.mean([r['probability'] for r in results_for_json if r['probability'] != -1.0])) if any(r['probability'] != -1.0 for r in results_for_json) else 0.0,
            'positive_percentage': round(sum(1 for r in results_for_json if r['prediction'] == 1) / valid_results_count * 100, 1) if valid_results_count > 0 else 0
        }
        
        # 生成箱线图数据
        box_plot_data = generate_box_plot_data(results_with_features_df)
        
        # 生成滑动窗口分析数据
        sliding_window_data = generate_sliding_window_data(results_with_features_df)
        
        # 生成降维可视化数据
        dimensionality_reduction_data = generate_dimensionality_reduction_data(results_with_features_df)
        
        # 保存图表到文件
        try:
            save_charts_to_files(results_with_features_df)
        except Exception as chart_error:
            print(f"Warning: Failed to save charts: {chart_error}")
            # 不影响主要功能，只是记录警告
        
        return jsonify({
            'success': True,
            'results': results_for_json,
            'stats': stats,
            'box_plot_data': box_plot_data,
            'sliding_window_data': sliding_window_data,
            'dimensionality_reduction_data': dimensionality_reduction_data
        })
    
    except Exception as e:
        print(f"Critical error in prediction route (/predict): {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred while processing your request. Please try again later or contact an administrator.'}), 500

@app.route('/export', methods=['POST'])
def export_results():
    """Export prediction results"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export.'}), 400
        
        df = pd.DataFrame(results)
        
        if format_type == 'csv':
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'gram_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        elif format_type == 'fasta':
            output = BytesIO()
            for result in results:
                if result['prediction'] == 1:
                    output.write(f">{result['id']} | Probability={result['probability']:.4f}\n".encode('utf-8'))
                    output.write(f"{result['sequence']}\n".encode('utf-8'))
            output.seek(0)
            return send_file(
                output,
                mimetype='text/plain',
                as_attachment=True,
                download_name=f'gram_positive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.fasta'
            )
        else:
            return jsonify({'error': 'Unsupported export format.'}), 400
    
    except Exception as e:
        print(f"Export error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error exporting results: {str(e)}'}), 500

@app.route('/example')
def get_example():
    """Get example data"""
    example_fasta = """>AP00001
GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV
>AP00002
YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY
>AP00004
NLCERASLTWTGNCGNTGHCDTQCRNWESAKHGACHKRGNWKCFCYFDC
>AP00005
VFIDILDKVENAIHNAAQVGIGFAKPFEKLINPK
>AP00006
GNNRPVYIPQPRPPHPRI"""
    return jsonify({'fasta': example_fasta})

@app.route('/api/generate_sequences', methods=['POST'])
def generate_sequences_route():
    """API route for generating antimicrobial peptide sequences."""
    try:
        # Get generation service
        gen_service = get_generation_service()
        
        # Load models if not already loaded
        if not gen_service.is_loaded:
            try:
                success = gen_service.load_models()
                if not success:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to load generation models. Please check the model files and network connectivity.'
                    }), 503
            except Exception as load_error:
                return jsonify({
                    'success': False,
                    'error': f'Model loading error: {str(load_error)}'
                }), 503
        
        # Parse request parameters
        data = request.json
        num_sequences = int(data.get('num_sequences', 5))
        seq_length = int(data.get('seq_length', 40))
        sampling_method = data.get('sampling_method', 'diverse')
        temperature = float(data.get('temperature', 1.0))
        reference_sequences = data.get('reference_sequences', [])
        
        # Parameter validation
        if num_sequences < 1 or num_sequences > 50:
            return jsonify({'error': 'Number of sequences must be between 1 and 50.'}), 400
        
        if seq_length < 10 or seq_length > 100:
            return jsonify({'error': 'Sequence length must be between 10 and 100.'}), 400
        
        if sampling_method not in ['basic', 'diverse', 'top_k', 'nucleus']:
            return jsonify({'error': 'Invalid sampling method.'}), 400
        
        if temperature < 0.1 or temperature > 3.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 3.0.'}), 400
        
        # Generate sequences
        result = gen_service.generate_sequences(
            num_sequences=num_sequences,
            seq_length=seq_length,
            sampling_method=sampling_method,
            temperature=temperature,
            reference_sequences=reference_sequences if reference_sequences else None,
            k=int(data.get('k', 10)),  # top_k parameter
            p=float(data.get('p', 0.9)),  # nucleus parameter
            diversity_strength=float(data.get('diversity_strength', 0.3))  # diversity parameter
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'sequences': result['sequences'],
                'parameters': result['parameters'],
                'model_info': gen_service.get_model_info()
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
    
    except Exception as e:
        print(f"Error during sequence generation: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error during sequence generation: {str(e)}'}), 500

@app.route('/api/database')
def get_database():
    """Get database sequences from Gram+-.fasta file."""
    print("DEBUG: /api/database endpoint called")
    try:
        # 构建FASTA文件路径
        fasta_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'Gram+-.fasta')
        
        if not os.path.exists(fasta_path):
            return jsonify({
                'success': False,
                'error': f'Database file not found: {fasta_path}'
            }), 404
        
        sequences = []
        total_length = 0
        
        # 读取FASTA文件
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence_str = str(record.seq).upper()
            sequence_length = len(sequence_str)
            
            sequences.append({
                'id': record.id,
                'sequence': sequence_str,
                'length': sequence_length,
                'description': record.description
            })
            total_length += sequence_length
        
        # 计算统计信息
        stats = {
            'total': len(sequences),
            'avg_length': total_length / len(sequences) if sequences else 0,
            'min_length': min(seq['length'] for seq in sequences) if sequences else 0,
            'max_length': max(seq['length'] for seq in sequences) if sequences else 0
        }
        
        return jsonify({
            'success': True,
            'sequences': sequences,
            'stats': stats,
            'file_info': {
                'path': fasta_path,
                'description': 'Antimicrobial peptide sequences with activity against Gram-negative bacteria'
            }
        })
        
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error loading database: {str(e)}'
        }), 500

@app.route('/model_status')
def model_status():
    """Get model status."""
    try:
        gen_service = get_generation_service()
        return jsonify(gen_service.get_model_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
