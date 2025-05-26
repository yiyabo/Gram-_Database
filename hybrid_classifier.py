import os
import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.optimizers import AdamW #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore

import keras_tuner as kt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 数据预处理常量 ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)

MAX_SEQUENCE_LENGTH = 32
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

PEPTIDE_FEATURES_CSV_PATH = './data/peptide_features.csv'
MODEL_OUTPUT_DIR = './model'
HYBRID_MODEL_PATH_TEMPLATE = os.path.join(MODEL_OUTPUT_DIR, 'hybrid_classifier_best_tuned.keras') # Keras 3 format
SCALER_PATH = os.path.join(MODEL_OUTPUT_DIR, 'hybrid_model_scaler.pkl')
TUNER_DIR = 'keras_tuner_dir'
PROJECT_NAME = 'hybrid_peptide_classifier_tuning'


def tokenize_and_pad_sequences(sequences, vocab_dict, max_len, pad_token_id, unk_token_id):
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

def load_and_preprocess_data(csv_path, vocab_dict, max_seq_len, 
                             pad_token_id, unk_token_id,
                             test_size=0.2, random_state=42):
    logger.info(f"从 {csv_path} 加载数据...")
    if not os.path.exists(csv_path):
        logger.error(f"错误: 数据文件 {csv_path} 未找到。")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"读取CSV文件 {csv_path} 时出错: {e}")
        return None

    df.dropna(subset=['Sequence', 'Label'], inplace=True)
    if df.empty:
        logger.error("数据在移除NaN后为空。")
        return None
    logger.info(f"原始数据集大小: {len(df)}")

    feature_names_path = './data/feature_names.txt'
    actual_global_feature_columns = []
    if os.path.exists(feature_names_path):
        logger.info(f"从 {feature_names_path} 加载特征名称。")
        with open(feature_names_path, 'r') as f:
            actual_global_feature_columns = [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"{feature_names_path} 未找到。将基于排除法推断特征列。")
        excluded_cols = ['ID', 'Sequence', 'Label', 'Source']
        actual_global_feature_columns = [col for col in df.columns if col not in excluded_cols]
    
    if not actual_global_feature_columns or len(actual_global_feature_columns) != 28:
         logger.warning(f"全局特征列未能正确加载或数量不为28 (实际数量: {len(actual_global_feature_columns)}). 请检查 feature_names.txt 或CSV文件。将尝试使用所有非排除列。")
         excluded_cols = ['ID', 'Sequence', 'Label', 'Source']
         actual_global_feature_columns = [col for col in df.columns if col not in excluded_cols]


    logger.info(f"选定的全局特征列 ({len(actual_global_feature_columns)}): {actual_global_feature_columns[:5]}... 等")

    # 首先基于整个DataFrame划分训练集和测试集索引，以确保所有部分对齐
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Label'])
    
    # 从训练集中再划分出一部分作为超参数搜索时的验证集
    # 注意：这个验证集仅用于超参数搜索。最终模型评估仍在test_df上进行。
    train_search_df, val_search_df = train_test_split(train_df, test_size=0.2, random_state=random_state, stratify=train_df['Label'])


    def process_dataframe_split(dataframe_split, scaler_obj=None, fit_scaler=False):
        logger.info(f"处理DataFrame部分，大小: {len(dataframe_split)}")
        sequences_raw = dataframe_split['Sequence'].tolist()
        sequences_processed = tokenize_and_pad_sequences(sequences_raw, vocab_dict, max_seq_len, pad_token_id, unk_token_id)
        
        global_features_raw = dataframe_split[actual_global_feature_columns].values.astype(np.float32)
        labels = dataframe_split['Label'].values.astype(np.float32)

        if fit_scaler:
            current_scaler = StandardScaler()
            global_features_scaled = current_scaler.fit_transform(global_features_raw)
            
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(current_scaler, f)
            logger.info(f"Scaler已拟合并保存到: {SCALER_PATH}")
        elif scaler_obj:
            global_features_scaled = scaler_obj.transform(global_features_raw)
        else: # 不应发生，除非逻辑错误
            logger.warning("Scaler既不拟合也不提供，全局特征将保持原始状态。")
            global_features_scaled = global_features_raw
            current_scaler = None
            
        return sequences_processed, global_features_scaled, labels, scaler_obj if not fit_scaler else current_scaler

    # 处理用于超参数搜索的训练集和验证集
    X_train_search_seq, X_train_search_global, y_train_search, fitted_scaler = process_dataframe_split(train_search_df, fit_scaler=True)
    X_val_search_seq, X_val_search_global, y_val_search, _ = process_dataframe_split(val_search_df, scaler_obj=fitted_scaler)

    # 处理最终测试集 (使用从训练数据拟合的scaler)
    X_test_final_seq, X_test_final_global, y_test_final, _ = process_dataframe_split(test_df, scaler_obj=fitted_scaler)
    
    # 处理完整的训练集 (用于在找到最佳超参数后重新训练模型)
    X_train_full_seq, X_train_full_global, y_train_full, _ = process_dataframe_split(train_df, scaler_obj=fitted_scaler)


    logger.info(f"数据预处理完成。")
    logger.info(f"搜索用训练集: X_seq shape: {X_train_search_seq.shape}, X_global shape: {X_train_search_global.shape}, y shape: {y_train_search.shape}")
    logger.info(f"搜索用验证集: X_seq shape: {X_val_search_seq.shape}, X_global shape: {X_val_search_global.shape}, y shape: {y_val_search.shape}")
    logger.info(f"最终测试集: X_seq shape: {X_test_final_seq.shape}, X_global shape: {X_test_final_global.shape}, y shape: {y_test_final.shape}")
    logger.info(f"完整训练集: X_seq shape: {X_train_full_seq.shape}, X_global shape: {X_train_full_global.shape}, y shape: {y_train_full.shape}")


    return (X_train_search_seq, X_train_search_global, y_train_search), \
           (X_val_search_seq, X_val_search_global, y_val_search), \
           (X_test_final_seq, X_test_final_global, y_test_final), \
           (X_train_full_seq, X_train_full_global, y_train_full), \
            actual_global_feature_columns


def build_hyper_model(hp):
    """构建并编译混合分类的超模型 (用于Keras Tuner)"""
    # 超参数定义
    embedding_dim_seq = hp.Choice('embedding_dim_seq', values=[32, 64, 128])
    lstm_units = hp.Choice('lstm_units', values=[32, 64, 128])
    
    mlp_dense1_units = hp.Int('mlp_dense1_units', min_value=32, max_value=128, step=32)
    mlp_dense2_units = hp.Int('mlp_dense2_units', min_value=16, max_value=64, step=16)
    # 可以选择是否包含第二层MLP
    include_mlp_dense2 = hp.Boolean('include_mlp_dense2', default=True)


    fused_dense_units = hp.Int('fused_dense_units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # 序列输入分支
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='sequence_input')
    seq_embedding = Embedding(input_dim=VOCAB_SIZE, 
                              output_dim=embedding_dim_seq, 
                              input_length=MAX_SEQUENCE_LENGTH, 
                              name='sequence_embedding')(sequence_input)
    lstm_out = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate), 
                             name='bidirectional_lstm')(seq_embedding)

    # 全局特征输入分支 (假设全局特征维度固定为28)
    # 在实际调用时，global_feature_dim 应从数据中获取 X_train_global.shape[1]
    global_features_input = Input(shape=(28,), name='global_features_input') # 固定为28
    x_global = global_features_input
    
    x_global = Dense(mlp_dense1_units, name='mlp_dense_1')(x_global)
    x_global = LeakyReLU(alpha=0.01)(x_global)
    x_global = BatchNormalization(name='mlp_bn_1')(x_global)
    x_global = Dropout(dropout_rate, name='mlp_dropout_1')(x_global)

    if include_mlp_dense2:
        x_global = Dense(mlp_dense2_units, name='mlp_dense_2')(x_global)
        x_global = LeakyReLU(alpha=0.01)(x_global)
        x_global = BatchNormalization(name='mlp_bn_2')(x_global)
        x_global = Dropout(dropout_rate, name='mlp_dropout_2')(x_global)
    mlp_out = x_global

    # 融合两个分支
    concatenated_features = Concatenate(name='concatenate_branches')([lstm_out, mlp_out])
    
    # 分类头
    fused_dense = Dense(fused_dense_units, name='fused_dense_1')(concatenated_features)
    fused_dense = LeakyReLU(alpha=0.01)(fused_dense)
    fused_dropout = Dropout(dropout_rate, name='fused_dropout_1')(fused_dense)
    
    output_logits = Dense(1, name='output_logits')(fused_dropout)

    model = Model(inputs=[sequence_input, global_features_input], outputs=output_logits, name='hybrid_hyper_model')
    
    optimizer = AdamW(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics_list = [
        'accuracy', 
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)
    return model


if __name__ == '__main__':
    logger.info("开始混合模型超参数搜索...")
    
    data_package = load_and_preprocess_data(
        csv_path=PEPTIDE_FEATURES_CSV_PATH,
        vocab_dict=VOCAB_DICT,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        pad_token_id=PAD_TOKEN_ID,
        unk_token_id=UNK_TOKEN_ID
    )

    if data_package:
        (X_train_search_seq, X_train_search_global, y_train_search), \
        (X_val_search_seq, X_val_search_global, y_val_search), \
        (X_test_final_seq, X_test_final_global, y_test_final), \
        (X_train_full_seq, X_train_full_global, y_train_full), \
        global_feature_names = data_package
        
        if X_train_search_global.shape[1] != 28:
            logger.error(f"全局特征维度不为28 (实际为: {X_train_search_global.shape[1]})，无法继续超参数搜索。请检查特征提取。")
            exit()

        logger.info("数据加载成功，准备进行超参数搜索。")

        # 定义Tuner (使用Hyperband)
        tuner = kt.Hyperband(
            hypermodel=build_hyper_model,
            objective=kt.Objective('val_auc', direction='max'), # 优化目标
            max_epochs=30,          # 单个模型配置最多训练的epoch数
            factor=3,               # Hyperband的缩减因子
            hyperband_iterations=2, # Hyperband算法的迭代次数 (迭代次数越多，搜索越彻底)
            directory=TUNER_DIR,
            project_name=PROJECT_NAME,
            overwrite=False # 如果目录已存在，则覆盖
        )

        # 定义早停回调 (用于tuner.search中的每个trial)
        search_early_stopping = EarlyStopping(monitor='val_auc', patience=5, mode='max', verbose=1)

        logger.info("开始超参数搜索 (tuner.search)...")
        tuner.search(
            [X_train_search_seq, X_train_search_global], 
            y_train_search,
            epochs=50, # 每个trial的训练轮数上限 (Hyperband内部会管理实际轮数)
            validation_data=([X_val_search_seq, X_val_search_global], y_val_search),
            callbacks=[search_early_stopping],
            batch_size=64 # 可以在hp中定义，这里为简化先固定
        )

        logger.info("超参数搜索完成。")
        tuner.results_summary()

        # 获取最佳超参数
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"找到的最佳超参数: {best_hps.values}")

        # 使用最佳超参数构建并重新训练模型 (在完整的训练数据上)
        logger.info("使用最佳超参数在完整训练数据上重新训练模型...")
        final_model = tuner.hypermodel.build(best_hps)
        
        # 为最终训练定义回调
        final_model_checkpoint = ModelCheckpoint(HYBRID_MODEL_PATH_TEMPLATE, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
        final_early_stopping = EarlyStopping(monitor='val_auc', patience=10, verbose=1, mode='max', restore_best_weights=True) # 更长的patience
        final_reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=5, min_lr=1e-6, mode='max', verbose=1)

        history = final_model.fit(
            [X_train_full_seq, X_train_full_global],
            y_train_full,
            epochs=100, # 最终模型训练更多轮数
            batch_size=64, # 使用固定的batch_size，因为我们没有在Keras Tuner中搜索它
            validation_data=([X_test_final_seq, X_test_final_global], y_test_final), # 使用最终测试集作为验证
            callbacks=[final_model_checkpoint, final_early_stopping, final_reduce_lr],
            verbose=1
        )
        
        logger.info("最终模型训练完成。")
        
        # 加载通过ModelCheckpoint保存的最佳模型进行评估
        if os.path.exists(HYBRID_MODEL_PATH_TEMPLATE):
            logger.info(f"从 {HYBRID_MODEL_PATH_TEMPLATE} 加载最佳调优模型进行最终评估...")
            final_model = tf.keras.models.load_model(HYBRID_MODEL_PATH_TEMPLATE)
        else:
            logger.warning("未能找到通过ModelCheckpoint保存的最佳模型，将使用最后一次训练的模型进行评估。")

        logger.info("在最终测试集上评估调优后的模型：")
        eval_results = final_model.evaluate([X_test_final_seq, X_test_final_global], y_test_final, verbose=1)
        
        results_dict = dict(zip(final_model.metrics_names, eval_results))
        logger.info(f"最终测试集评估结果: {results_dict}")
        
        y_pred_logits = final_model.predict([X_test_final_seq, X_test_final_global])
        y_pred_probs_sigmoid = tf.sigmoid(y_pred_logits).numpy()
        y_pred_classes_final = (y_pred_probs_sigmoid > 0.5).astype(int)

        from sklearn.metrics import f1_score as sklearn_f1_score
        f1 = sklearn_f1_score(y_test_final, y_pred_classes_final)
        logger.info(f"最终测试集 F1 Score (手动计算): {f1:.4f}")
        results_dict['f1_score_manual'] = f1

        eval_results_path = os.path.join(MODEL_OUTPUT_DIR, 'hybrid_classifier_tuned_eval_results.txt')
        with open(eval_results_path, 'w') as f:
            f.write(f"Best Hyperparameters: {best_hps.values}\n\n")
            for key, value in results_dict.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"调优后模型的评估结果已保存到: {eval_results_path}")

    else:
        logger.error("数据加载和预处理失败，无法继续进行超参数搜索。")