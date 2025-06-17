#!/usr/bin/env python3
"""
混合分类器 V2 - 优化版本
基于最佳超参数进行针对性优化，重点提升Recall
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import logging
import pickle

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
MODEL_OUTPUT_DIR = './model_v2'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def build_optimized_model(best_hps, class_weights=None):
    """基于最佳超参数构建优化模型"""
    
    # 从最佳超参数中提取参数
    embedding_dim_seq = best_hps.get('embedding_dim_seq', 64)
    lstm_units = best_hps.get('lstm_units', 128)
    mlp_dense1_units = best_hps.get('mlp_dense1_units', 128)
    mlp_dense2_units = best_hps.get('mlp_dense2_units', 64)
    include_mlp_dense2 = best_hps.get('include_mlp_dense2', False)
    fused_dense1_units = best_hps.get('fused_dense1_units', 64)
    fused_dense2_units = best_hps.get('fused_dense2_units', 32)
    include_fused_dense2 = best_hps.get('include_fused_dense2', True)
    dropout_rate = best_hps.get('dropout_rate', 0.3)
    recurrent_dropout_rate = best_hps.get('recurrent_dropout_rate', 0.1)
    learning_rate = best_hps.get('learning_rate', 0.001)
    weight_decay = best_hps.get('weight_decay', 1e-5)
    
    logger.info("构建优化模型...")
    logger.info(f"参数: embedding_dim={embedding_dim_seq}, lstm_units={lstm_units}")
    
    # 序列输入分支
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input')
    seq_embedding = Embedding(input_dim=VOCAB_SIZE, 
                              output_dim=embedding_dim_seq, 
                              input_length=MAX_SEQUENCE_LENGTH, 
                              name='sequence_embedding')(sequence_input)
    
    # 双向LSTM
    lstm_out = Bidirectional(LSTM(lstm_units, 
                                  dropout=dropout_rate, 
                                  recurrent_dropout=recurrent_dropout_rate), 
                             name='bidirectional_lstm')(seq_embedding)
    
    # 全局特征输入分支
    global_features_input = Input(shape=(28,), name='global_features_input')
    x_global = global_features_input
    
    # MLP第一层
    x_global = Dense(mlp_dense1_units, name='mlp_dense_1')(x_global)
    x_global = LeakyReLU(alpha=0.01)(x_global)
    x_global = BatchNormalization(name='mlp_bn_1')(x_global)
    x_global = Dropout(dropout_rate, name='mlp_dropout_1')(x_global)
    
    # MLP第二层（可选）
    if include_mlp_dense2:
        x_global = Dense(mlp_dense2_units, name='mlp_dense_2')(x_global)
        x_global = LeakyReLU(alpha=0.01)(x_global)
        x_global = BatchNormalization(name='mlp_bn_2')(x_global)
        x_global = Dropout(dropout_rate, name='mlp_dropout_2')(x_global)
    
    mlp_out = x_global
    
    # 融合两个分支
    concatenated_features = Concatenate(name='concatenate_branches')([lstm_out, mlp_out])
    
    # 融合层
    fused_dense = Dense(fused_dense1_units, name='fused_dense_1')(concatenated_features)
    fused_dense = LeakyReLU(alpha=0.01)(fused_dense)
    fused_dense = BatchNormalization(name='fused_bn_1')(fused_dense)
    fused_dense = Dropout(dropout_rate, name='fused_dropout_1')(fused_dense)
    
    # 第二层融合（可选）
    if include_fused_dense2:
        fused_dense = Dense(fused_dense2_units, name='fused_dense_2')(fused_dense)
        fused_dense = LeakyReLU(alpha=0.01)(fused_dense)
        fused_dense = BatchNormalization(name='fused_bn_2')(fused_dense)
        fused_dense = Dropout(dropout_rate, name='fused_dropout_2')(fused_dense)
    
    # 输出层
    output_logits = Dense(1, name='output_logits')(fused_dense)
    
    model = Model(inputs=[sequence_input, global_features_input], outputs=output_logits, name='hybrid_optimized_model')
    
    # 编译模型 - 使用类别权重
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
    # 使用Focal Loss来处理类别不平衡
    def focal_loss(alpha=0.25, gamma=2.0):
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
    
    # 选择损失函数
    if class_weights is not None:
        # 使用Focal Loss处理不平衡
        loss_fn = focal_loss(alpha=0.4, gamma=2.0)  # 增加对少数类的关注
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    metrics_list = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    return model

def tokenize_sequence(sequence):
    """序列tokenization"""
    tokens = [VOCAB_DICT.get(aa, UNK_TOKEN_ID) for aa in sequence.upper()]
    if len(tokens) > MAX_SEQUENCE_LENGTH:
        tokens = tokens[:MAX_SEQUENCE_LENGTH]
    else:
        tokens = tokens + [PAD_TOKEN_ID] * (MAX_SEQUENCE_LENGTH - len(tokens))
    return tokens

def main():
    """主训练函数"""
    logger.info("开始优化训练...")
    
    # 最佳超参数（从您的结果中获取）
    best_hps = {
        'embedding_dim_seq': 64,
        'lstm_units': 128,
        'mlp_dense1_units': 128,
        'mlp_dense2_units': 64,
        'include_mlp_dense2': False,
        'fused_dense1_units': 64,
        'fused_dense2_units': 32,
        'include_fused_dense2': True,
        'dropout_rate': 0.3,
        'recurrent_dropout_rate': 0.1,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }
    
    # 加载数据
    logger.info("加载数据...")
    df = pd.read_csv(PEPTIDE_FEATURES_CSV_PATH)
    df.dropna(subset=['Sequence', 'Label'], inplace=True)
    
    logger.info(f"数据集大小: {len(df)}")
    logger.info(f"标签分布:\n{df['Label'].value_counts()}")
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(df['Label']), 
                                       y=df['Label'])
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    logger.info(f"类别权重: {class_weight_dict}")
    
    # 数据预处理
    logger.info("数据预处理...")
    sequences = df['Sequence'].tolist()
    labels = df['Label'].values.astype(np.float32)
    
    # 特征列
    feature_columns = [col for col in df.columns if col not in ['ID', 'Sequence', 'Label', 'Source']]
    global_features = df[feature_columns].values.astype(np.float32)
    
    # 序列tokenization
    tokenized_sequences = np.array([tokenize_sequence(seq) for seq in sequences])
    
    # 标准化全局特征
    scaler = StandardScaler()
    global_features_scaled = scaler.fit_transform(global_features)
    
    # 数据分割
    X_seq_train, X_seq_test, X_global_train, X_global_test, y_train, y_test = train_test_split(
        tokenized_sequences, global_features_scaled, labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"训练集大小: {len(X_seq_train)}")
    logger.info(f"测试集大小: {len(X_seq_test)}")
    
    # 构建模型
    model = build_optimized_model(best_hps, class_weight_dict)
    logger.info("模型架构:")
    model.summary()
    
    # 训练回调
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=20, mode='max', verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, min_lr=1e-7, mode='max', verbose=1),
        ModelCheckpoint(os.path.join(MODEL_OUTPUT_DIR, 'best_model_v2.keras'), 
                       monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]
    
    # 训练模型
    logger.info("开始训练...")
    history = model.fit(
        [X_seq_train, X_global_train], y_train,
        epochs=100,
        batch_size=128,
        validation_data=([X_seq_test, X_global_test], y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,  # 使用类别权重
        verbose=1
    )
    
    # 保存scaler
    with open(os.path.join(MODEL_OUTPUT_DIR, 'scaler_v2.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 最终评估
    logger.info("最终评估...")
    y_pred_logits = model.predict([X_seq_test, X_global_test])
    y_pred_probs = tf.nn.sigmoid(y_pred_logits).numpy()
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    
    # 计算指标
    auc_score = roc_auc_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, y_pred_classes)
    
    logger.info(f"最终 AUC: {auc_score:.4f}")
    logger.info(f"最终 F1: {f1:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_classes))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    logger.info("优化训练完成！")

if __name__ == '__main__':
    main()
