#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用训练好的混合模型（LSTM+MLP）对新的肽序列进行抗革兰氏阴性菌活性预测。
此脚本结合了序列的原始氨基酸信息和全局特征进行预测。
"""

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO
from peptides import Peptide

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 数据预处理常量 ---
# 词汇表: <PAD>:0, <UNK>:1, A:2 ... Y:21
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)

MAX_SEQUENCE_LENGTH = 128
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

def parse_fasta_for_prediction(fasta_file_path):
    """从FASTA文件中解析序列用于预测"""
    sequences_data = []
    try:
        for record in SeqIO.parse(fasta_file_path, "fasta"):
            peptide_id = record.id
            sequence = str(record.seq).upper()
            if not sequence or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
                logger.debug(f"跳过无效序列 ID: {peptide_id}, Sequence: {sequence[:30]}...")
                continue
            sequences_data.append((peptide_id, sequence))
    except FileNotFoundError:
        logger.error(f"FASTA 文件未找到: {fasta_file_path}")
        return []
    except Exception as e:
        logger.error(f"解析FASTA文件 {fasta_file_path} 时出错: {e}")
        return []
    logger.info(f"从 {fasta_file_path} 解析到 {len(sequences_data)} 条有效序列用于预测")
    return sequences_data

def extract_single_sequence_features(sequence_string, feature_order_list):
    """从单个肽序列字符串中提取特征，顺序由 feature_order_list 决定。"""
    try:
        pep = Peptide(sequence_string)
        length = len(sequence_string)
        if length == 0:
            logger.warning(f"序列 '{sequence_string}' 为空，无法提取特征。")
            return None

        charge = pep.charge(pH=7.4)
        hydrophobicity = pep.hydrophobicity(scale="Eisenberg")
        
        hm_window = min(11, length) 
        if length < 3: # peptides库对非常短的序列计算hydrophobic_moment可能有问题
            hydrophobic_moment_val = 0.0
        else:
            hydrophobic_moment_val = pep.hydrophobic_moment(window=hm_window)
        if hydrophobic_moment_val is None: hydrophobic_moment_val = 0.0

        instability_index = pep.instability_index()
        isoelectric_point = pep.isoelectric_point()
        aliphatic_index = pep.aliphatic_index()
        hydrophilicity = pep.hydrophobicity(scale="HoppWoods")
        
        aa_comp = {f'AA_{aa}': sequence_string.count(aa) / length for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        
        feature_values_dict = {
            'Length': float(length), 'Charge': float(charge),
            'Hydrophobicity': float(hydrophobicity), 'Hydrophobic_Moment': float(hydrophobic_moment_val),
            'Instability_Index': float(instability_index), 'Isoelectric_Point': float(isoelectric_point),
            'Aliphatic_Index': float(aliphatic_index), 'Hydrophilicity': float(hydrophilicity)
        }
        feature_values_dict.update(aa_comp)
        
        feature_vector = [feature_values_dict.get(name, 0.0) for name in feature_order_list]
        return np.array(feature_vector, dtype=np.float32)

    except Exception as e:
        logger.warning(f"为序列 '{sequence_string[:30]}...' 提取特征时出错: {e}")
        return None

def tokenize_and_pad_sequences(sequences, vocab_dict, max_len, pad_token_id, unk_token_id):
    """将原始氨基酸序列字符串列表转换为分词和填充后的整数ID序列。"""
    tokenized_sequences = []
    for seq_str in sequences:
        seq_str = str(seq_str).upper() # 确保是大写并处理非字符串输入
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

def focal_loss(alpha=0.25, gamma=2.0):
    """定义Focal Loss函数（用于模型加载）"""
    def focal_loss_fixed(y_true, y_pred):
        # 将logits转换为概率
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        # 计算focal loss
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred_sigmoid, 1 - y_pred_sigmoid)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        focal_loss = focal_weight * ce_loss
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed

def load_keras_model(model_path):
    """加载训练好的Keras混合模型"""
    if not os.path.exists(model_path):
        logger.error(f"模型文件未找到: {model_path}")
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    try:
        # 定义自定义对象
        custom_objects = {
            'focal_loss_fixed': focal_loss(alpha=0.4, gamma=2.0)
        }

        # 使用自定义对象加载模型
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"从 {model_path} 成功加载Keras模型")
        return model
    except Exception as e:
        logger.error(f"加载Keras模型时出错: {e}")
        raise

def load_scaler(scaler_path):
    """加载用于全局特征的StandardScaler"""
    if not os.path.exists(scaler_path):
        logger.warning(f"Scaler文件未找到: {scaler_path}。将不使用Scaler。")
        return None
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"从 {scaler_path} 成功加载Scaler。")
        if scaler:
            logger.info(f"Scaler mean (first 5): {scaler.mean_[:5]}")
            logger.info(f"Scaler scale (first 5): {scaler.scale_[:5]}")
        return scaler
    except Exception as e:
        logger.error(f"加载Scaler时出错: {e}")
        return None

def predict_with_hybrid_model(model, seq_data, global_features, threshold=0.5):
    """使用混合模型进行预测"""
    if seq_data.size == 0 or global_features.size == 0:
        logger.warning("没有可预测的数据。")
        return [], []
    
    try:
        # 混合模型输出的是logits
        logits = model.predict([seq_data, global_features])
        probabilities = tf.sigmoid(logits).numpy().flatten()
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions.tolist(), probabilities.tolist()
    except Exception as e:
        logger.error(f"使用混合模型进行预测时出错: {e}")
        return [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的混合模型（LSTM+MLP）预测肽序列的抗革兰氏阴性菌活性。")
    parser.add_argument("--model_path", type=str, default="./model/Best Hybrid Classifier.keras",
                        help="训练好的Keras混合模型文件路径 (.keras)。")
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="用于预测的输入FASTA文件路径。")
    parser.add_argument("--scaler_path", type=str, default="./model/hybrid_model_scaler.pkl",
                        help="保存的scikit-learn scaler对象路径 (.pkl)，用于全局特征标准化。")
    parser.add_argument("--output_file", type=str, default="./predictions/hybrid_model_predictions.txt",
                        help="保存预测结果的路径。")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二分类的预测阈值。")
    parser.add_argument("--feature_names_file", type=str, default="./data/feature_names.txt",
                        help="包含特征名称顺序的文件路径 (每行一个特征名)。")

    args = parser.parse_args()

    # --- 加载模型 ---
    try:
        hybrid_model = load_keras_model(args.model_path)
    except Exception as e:
        logger.critical(f"无法加载模型，脚本终止: {e}")
        exit(1)

    # --- 加载特征名称顺序 ---
    feature_order = []
    if os.path.exists(args.feature_names_file):
        with open(args.feature_names_file, 'r') as f_names:
            feature_order = [line.strip() for line in f_names if line.strip()]
        logger.info(f"从 {args.feature_names_file} 成功加载了 {len(feature_order)} 个特征名称。")
    else:
        logger.error(f"特征名称文件 {args.feature_names_file} 未找到。无法确定正确的特征顺序，脚本终止。")
        exit(1)

    # --- 加载Scaler ---
    scaler = load_scaler(args.scaler_path)

    # --- 加载并预处理FASTA数据 ---
    sequences_to_predict = parse_fasta_for_prediction(args.fasta_file)
    
    if not sequences_to_predict:
        logger.error("未能从FASTA文件中解析到有效序列。请检查输入文件。")
        exit(1)
    
    # 提取序列ID和序列字符串
    sequence_ids = [seq_id for seq_id, _ in sequences_to_predict]
    sequence_strings = [seq_str for _, seq_str in sequences_to_predict]
    
    # --- 处理序列数据（用于LSTM分支）---
    logger.info("处理序列数据用于LSTM分支...")
    tokenized_padded_sequences = tokenize_and_pad_sequences(
        sequence_strings, VOCAB_DICT, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID, UNK_TOKEN_ID
    )
    logger.info(f"序列数据处理完成，形状: {tokenized_padded_sequences.shape}")
    
    # --- 提取全局特征（用于MLP分支）---
    logger.info("从序列中提取全局特征用于MLP分支...")
    all_features_list = []
    processed_sequence_ids = []
    
    for i, (seq_id, sequence_str) in enumerate(sequences_to_predict):
        raw_features = extract_single_sequence_features(sequence_str, feature_order)
        
        if raw_features is not None:
            all_features_list.append(raw_features)
            processed_sequence_ids.append(seq_id)
        else:
            logger.warning(f"无法为序列 {seq_id} 提取特征，将从预测中排除。")
            # 同时从tokenized_padded_sequences中移除对应的序列
            tokenized_padded_sequences = np.delete(tokenized_padded_sequences, i, axis=0)
    
    if not all_features_list:
        logger.error("未能从任何序列中成功提取有效特征。请检查输入FASTA文件或特征提取逻辑。")
        exit(1)
    
    # 转换为numpy数组
    global_features_np = np.array(all_features_list, dtype=np.float32)
    logger.info(f"全局特征提取完成，形状: {global_features_np.shape}")
    
    # 应用Scaler（如果可用）
    scaled_global_features_np = global_features_np
    if scaler:
        try:
            scaled_global_features_np = scaler.transform(global_features_np)
            logger.info("全局特征成功标准化。")
        except Exception as e:
            logger.error(f"应用scaler时发生错误: {e}。将使用未标准化的特征。")
    else:
        logger.info("未加载Scaler或Scaler不适用，将使用原始特征值进行预测。")
    
    # --- 进行预测 ---
    logger.info(f"对 {len(processed_sequence_ids)} 条序列进行预测...")
    predictions, probabilities = predict_with_hybrid_model(
        hybrid_model, tokenized_padded_sequences, scaled_global_features_np, args.threshold
    )
    
    if not predictions:
        logger.error("预测失败，未能获得任何预测结果。")
        exit(1)
    
    # --- 保存或打印结果 ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    with open(args.output_file, 'w') as f_out:
        f_out.write("Sequence_ID\tPrediction\tProbability\n")
        logger.info("预测结果:")
        logger.info("--------------------")
        for i, seq_id in enumerate(processed_sequence_ids):
            if i < len(predictions):
                pred_label = "抗革兰氏阴性菌活性" if predictions[i] == 1 else "非抗革兰氏阴性菌活性"
                f_out.write(f"{seq_id}\t{pred_label}\t{probabilities[i]:.4f}\n")
                logger.info(f"{seq_id}: {pred_label} (概率: {probabilities[i]:.4f})")
            else:
                f_out.write(f"{seq_id}\tPREDICTION_FAILED\tN/A\n")
                logger.warning(f"{seq_id}: PREDICTION_FAILED")
        logger.info("--------------------")
    
    logger.info(f"预测结果已保存到 {args.output_file}")
    logger.info("预测脚本执行完毕。")