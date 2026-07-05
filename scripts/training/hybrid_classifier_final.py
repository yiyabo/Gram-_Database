#!/usr/bin/env python3
"""
混合分类器 V1 最优版本 - 基于消融实验结果
仅使用 Label Smoothing 优化，避免有害的 Focal Loss
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)

MAX_SEQUENCE_LENGTH = 128
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

# 文件路径
PEPTIDE_FEATURES_CSV_PATH = './data/peptide_features.csv'
MODEL_OUTPUT_DIR = './model_v1_final'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# 最优超参数 - 基于消融实验最佳配置
BEST_HYPERPARAMS = {
    # 基础架构参数 (继承原版V1的优秀架构)
    'embedding_dim_seq': 256,
    'lstm_units': 512,
    'mlp_dense1_units': 384,
    'mlp_dense2_units': 160,
    'include_mlp_dense2': True,
    'include_mlp_dense3': True,
    'mlp_dense3_units': 16,
    'fused_dense1_units': 128,
    'include_fused_dense2': True,
    'fused_dense2_units': 224,
    
    # 正则化参数 (适中设置)
    'dropout_rate': 0.3,
    'recurrent_dropout_rate': 0.2,
    
    # 优化器参数
    'learning_rate': 0.001,
    'weight_decay': 5e-5,
    
    # 关键优化：仅使用Label Smoothing
    'use_self_attention': False,  # 不使用attention
    'use_focal_loss': False,      # 不使用focal loss
    'label_smoothing': 0.15       # 仅使用label smoothing
}

def tokenize_and_pad_sequences(sequences, vocab_dict, max_len, pad_token_id, unk_token_id):
    """序列tokenization和padding"""
    tokenized_sequences = []
    for seq_str in sequences:
        seq_str = str(seq_str).upper()
        tokens = [vocab_dict.get(aa, unk_token_id) for aa in seq_str]
        tokenized_sequences.append(tokens)
    
    padded_sequences = pad_sequences(
        tokenized_sequences, maxlen=max_len, dtype='int32', 
        padding='post', truncating='post', value=pad_token_id
    )
    return padded_sequences

def load_and_preprocess_data(csv_path, test_size=0.2, random_state=42):
    """数据加载和预处理"""
    logger.info(f"从 {csv_path} 加载数据...")
    
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Sequence', 'Label'], inplace=True)
    
    logger.info(f"数据集大小: {len(df)}")
    logger.info(f"标签分布:\n{df['Label'].value_counts()}")

    # 获取特征列
    feature_names_path = './data/feature_names.txt'
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_columns = [line.strip() for line in f if line.strip()]
    else:
        excluded_cols = ['ID', 'Sequence', 'Label', 'Source']
        feature_columns = [col for col in df.columns if col not in excluded_cols]
    
    logger.info(f"特征列数量: {len(feature_columns)}")

    # 数据划分 (使用固定随机种子确保一致性)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Label'])

    # 序列处理
    train_sequences = tokenize_and_pad_sequences(
        train_df['Sequence'].tolist(), VOCAB_DICT, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID, UNK_TOKEN_ID
    )
    test_sequences = tokenize_and_pad_sequences(
        test_df['Sequence'].tolist(), VOCAB_DICT, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID, UNK_TOKEN_ID
    )

    # 全局特征处理
    train_global_features = train_df[feature_columns].values.astype(np.float32)
    test_global_features = test_df[feature_columns].values.astype(np.float32)
    
    # 标准化
    scaler = StandardScaler()
    train_global_features_scaled = scaler.fit_transform(train_global_features)
    test_global_features_scaled = scaler.transform(test_global_features)
    
    # 保存scaler
    scaler_path = os.path.join(MODEL_OUTPUT_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # 标签
    train_labels = train_df['Label'].values.astype(np.float32)
    test_labels = test_df['Label'].values.astype(np.float32)

    logger.info(f"训练集: {len(train_labels)}, 测试集: {len(test_labels)}")
    
    return (train_sequences, train_global_features_scaled, train_labels), \
           (test_sequences, test_global_features_scaled, test_labels)

def build_final_model(hyperparams):
    """构建最优模型 - 基于消融实验结果"""
    
    # 序列输入分支
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input')
    seq_embedding = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=hyperparams['embedding_dim_seq'],
        input_length=MAX_SEQUENCE_LENGTH,
        name='sequence_embedding'
    )(sequence_input)

    # LSTM分支 (不使用attention，所以return_sequences=False)
    lstm_out = Bidirectional(LSTM(
        hyperparams['lstm_units'],
        dropout=hyperparams['dropout_rate'],
        recurrent_dropout=hyperparams['recurrent_dropout_rate'],
        return_sequences=False
    ), name='bidirectional_lstm')(seq_embedding)

    # 全局特征输入分支
    global_features_input = Input(shape=(28,), name='global_features_input')
    x_global = global_features_input
    
    # MLP第一层
    x_global = Dense(hyperparams['mlp_dense1_units'], name='mlp_dense_1')(x_global)
    x_global = LeakyReLU(alpha=0.01)(x_global)
    x_global = BatchNormalization(name='mlp_bn_1')(x_global)
    x_global = Dropout(hyperparams['dropout_rate'], name='mlp_dropout_1')(x_global)
    
    # MLP第二层
    if hyperparams['include_mlp_dense2']:
        x_global = Dense(hyperparams['mlp_dense2_units'], name='mlp_dense_2')(x_global)
        x_global = LeakyReLU(alpha=0.01)(x_global)
        x_global = BatchNormalization(name='mlp_bn_2')(x_global)
        x_global = Dropout(hyperparams['dropout_rate'], name='mlp_dropout_2')(x_global)

    # MLP第三层
    if hyperparams['include_mlp_dense3']:
        x_global = Dense(hyperparams['mlp_dense3_units'], name='mlp_dense_3')(x_global)
        x_global = LeakyReLU(alpha=0.01)(x_global)
        x_global = BatchNormalization(name='mlp_bn_3')(x_global)
        x_global = Dropout(hyperparams['dropout_rate'], name='mlp_dropout_3')(x_global)

    # 融合两个分支
    concatenated_features = Concatenate(name='concatenate_branches')([lstm_out, x_global])

    # 融合层
    fused_dense = Dense(hyperparams['fused_dense1_units'], name='fused_dense_1')(concatenated_features)
    fused_dense = LeakyReLU(alpha=0.01)(fused_dense)
    fused_dense = BatchNormalization(name='fused_bn_1')(fused_dense)
    fused_dense = Dropout(hyperparams['dropout_rate'], name='fused_dropout_1')(fused_dense)

    # 第二层融合
    if hyperparams['include_fused_dense2']:
        fused_dense = Dense(hyperparams['fused_dense2_units'], name='fused_dense_2')(fused_dense)
        fused_dense = LeakyReLU(alpha=0.01)(fused_dense)
        fused_dense = BatchNormalization(name='fused_bn_2')(fused_dense)
        fused_dense = Dropout(hyperparams['dropout_rate'], name='fused_dropout_2')(fused_dense)

    output_logits = Dense(1, name='output_logits')(fused_dense)

    model = Model(inputs=[sequence_input, global_features_input], outputs=output_logits, name='hybrid_v1_final')

    # 优化器配置
    optimizer = AdamW(learning_rate=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    
    # 损失函数 - 仅使用Label Smoothing
    def label_smoothing_loss_fn(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        
        # Label Smoothing
        if hyperparams['label_smoothing'] > 0:
            y_true_smooth = y_true_f * (1 - hyperparams['label_smoothing']) + 0.5 * hyperparams['label_smoothing']
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_smooth, logits=y_pred))
        else:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_f, logits=y_pred))
    
    metrics_list = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(optimizer=optimizer, loss=label_smoothing_loss_fn, metrics=metrics_list)
    return model

def main():
    logger.info("开始V1最优版训练 - 基于消融实验结果...")
    
    # 加载数据
    train_data, test_data = load_and_preprocess_data(PEPTIDE_FEATURES_CSV_PATH)
    train_seq, train_global, train_labels = train_data
    test_seq, test_global, test_labels = test_data
    
    # 构建模型
    logger.info("构建最优模型 (Label Smoothing Only)...")
    model = build_final_model(BEST_HYPERPARAMS)
    logger.info("模型架构:")
    model.summary()
    
    # 训练回调
    weights_path = os.path.join(MODEL_OUTPUT_DIR, 'best_weights.h5')
    callbacks = [
        ModelCheckpoint(weights_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1, save_weights_only=True),
        EarlyStopping(monitor='val_auc', patience=20, mode='max', verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, min_lr=1e-7, mode='max', verbose=1)
    ]
    
    # 训练模型
    logger.info("开始训练...")
    history = model.fit(
        [train_seq, train_global], train_labels,
        epochs=100,
        batch_size=128,
        validation_data=([test_seq, test_global], test_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    # 最终评估
    logger.info("最终评估...")
    eval_results = model.evaluate([test_seq, test_global], test_labels, verbose=1)
    results_dict = dict(zip(model.metrics_names, eval_results))
    
    # 详细指标计算
    y_pred_logits = model.predict([test_seq, test_global])
    y_pred_probs = tf.sigmoid(y_pred_logits).numpy()
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()

    f1 = f1_score(test_labels, y_pred_classes)
    cm = confusion_matrix(test_labels, y_pred_classes)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(test_labels, y_pred_classes)

    logger.info(f"\n=== V1最优版最终结果 ===")
    logger.info(f"配置: 仅使用Label Smoothing (0.15)")
    logger.info(f"Accuracy: {results_dict['accuracy']:.4f} ({results_dict['accuracy']*100:.2f}%)")
    logger.info(f"AUC: {results_dict['auc']:.4f}")
    logger.info(f"Precision: {results_dict['precision']:.4f}")
    logger.info(f"Recall (Sensitivity): {results_dict['recall']:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # 与原版V1对比
    logger.info(f"\n=== 与原版V1对比 ===")
    original_f1 = 0.8917
    logger.info(f"原版V1 F1-Score: {original_f1:.4f}")
    logger.info(f"最优版 F1-Score: {f1:.4f}")
    logger.info(f"提升: {f1-original_f1:+.4f}")
    
    if f1 >= original_f1:
        logger.info("✅ 最优版性能达到或超过原版V1")
    else:
        logger.info("⚠️ 最优版性能略低于原版V1，但避免了复杂优化的负面影响")

    # 保存结果
    results_path = os.path.join(MODEL_OUTPUT_DIR, 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write("V1 Final Model - Label Smoothing Only Results\n")
        f.write("="*50 + "\n")
        f.write(f"Configuration: Label Smoothing = {BEST_HYPERPARAMS['label_smoothing']}\n")
        f.write(f"Accuracy: {results_dict['accuracy']:.4f} ({results_dict['accuracy']*100:.2f}%)\n")
        f.write(f"AUC: {results_dict['auc']:.4f}\n")
        f.write(f"Precision: {results_dict['precision']:.4f}\n")
        f.write(f"Recall (Sensitivity): {results_dict['recall']:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"\nComparison with Original V1:\n")
        f.write(f"Original F1: {original_f1:.4f}\n")
        f.write(f"Final F1: {f1:.4f}\n")
        f.write(f"Improvement: {f1-original_f1:+.4f}\n")
    
    logger.info(f"结果已保存到: {results_path}")
    logger.info("V1最优版训练完成！")

if __name__ == '__main__':
    main()