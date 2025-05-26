#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抗革兰氏阴性菌预测网站
提供Web界面用于预测肽段序列是否为抗革兰氏阴性菌
使用混合Keras模型 (LSTM+MLP)
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
from Bio import SeqIO
from peptides import Peptide 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from collections import Counter
import pickle
import traceback # For detailed error logging

print("!!!!!!!!!! DEBUG: app.py IS LOADING FRESHLY (Hybrid Model Version) !!!!!!!!!!")

# --- 从 hybrid_classifier.py 复制/改编的常量和函数 ---
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)

MAX_SEQUENCE_LENGTH = 32 
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

def tokenize_and_pad_sequences_app(sequences, vocab_dict, max_len, pad_token_id, unk_token_id):
    """将原始氨基酸序列字符串列表转换为分词和填充后的整数ID序列。"""
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
# --- 结束复制/改编的部分 ---


def load_keras_model(model_path):
    """加载训练好的Keras模型"""
    print(f"从 {model_path} 加载Keras模型...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Keras模型加载成功!")
        return model
    except FileNotFoundError:
        print(f"错误: Keras模型文件未找到于 {model_path}")
        raise
    except Exception as e:
        print(f"加载Keras模型时发生未知错误: {e}")
        raise

def load_app_scaler(scaler_path):
    """加载用于全局特征的StandardScaler"""
    print(f"尝试从以下路径加载 StandardScaler: {scaler_path}")
    if not os.path.exists(scaler_path):
        error_msg = f"错误: 必需的 StandardScaler 文件 '{scaler_path}' 未找到。无法进行预测。"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"成功从 {scaler_path} 加载 StandardScaler。")
        return scaler
    except Exception as e:
        print(f"从 {scaler_path} 加载 StandardScaler 时发生错误: {e}")
        raise RuntimeError(f"无法加载 StandardScaler 文件 '{scaler_path}': {e}")

def extract_features_from_fasta(fasta_file):
    """从 FASTA 文件中提取肽段特征和原始序列"""
    print(f"从 {fasta_file} 提取特征...")
    records_data = [] 
    
    feature_names_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'feature_names.txt')
    expected_feature_order = []
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f_names:
            expected_feature_order = [line.strip() for line in f_names if line.strip()]
        if len(expected_feature_order) != 28:
            print(f"警告: feature_names.txt 中的特征数量 ({len(expected_feature_order)}) 不是28。将使用默认顺序。")
            expected_feature_order = [] 
    else:
        print(f"警告: {feature_names_path} 未找到。将使用默认特征顺序。")

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
            print(f"跳过无效序列 {peptide_id}: {sequence}")
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
            print(f"处理序列 {peptide_id} 时出错: {e}")
    
    df = pd.DataFrame(records_data)
    if not df.empty:
        cols_to_keep = ['ID', 'Sequence'] + [col for col in expected_feature_order if col in df.columns]
        df = df[cols_to_keep]

    print(f"成功提取 {len(df)} 条序列的特征和原始序列")
    return df

def predict_with_hybrid_model_app(k_model, scaler, sequence_ids_list, sequence_strings_list, global_features_df, threshold=0.5):
    """使用加载的Keras混合模型和Scaler进行预测。"""
    print("!!!!!!!!!! DEBUG: predict_with_hybrid_model_app FUNCTION HAS BEEN CALLED !!!!!!!!!!")
    if not sequence_ids_list:
        print("没有序列可供预测。")
        return pd.DataFrame(columns=['ID', 'Sequence', 'Probability', 'Prediction', 'Label'])

    print(f"处理 {len(sequence_strings_list)} 条序列用于LSTM分支...")
    tokenized_padded_sequences = tokenize_and_pad_sequences_app(
        sequence_strings_list, VOCAB_DICT, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID, UNK_TOKEN_ID
    )
    print(f"序列数据处理完成，形状: {tokenized_padded_sequences.shape}")

    feature_column_names = [col for col in global_features_df.columns if col not in ['ID', 'Sequence']]
    global_features_np = global_features_df[feature_column_names].values.astype(np.float32)
    print(f"全局特征提取完成，形状: {global_features_np.shape}")
    
    scaled_global_features_np = global_features_np
    if scaler:
        try:
            if global_features_np.shape[1] != scaler.n_features_in_:
                 print(f"错误: 输入到scaler的特征数量 ({global_features_np.shape[1]}) 与scaler期望的数量 ({scaler.n_features_in_}) 不符。")
                 error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "特征维度错误"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
                 return pd.DataFrame(error_results)
            scaled_global_features_np = scaler.transform(global_features_np)
            print("全局特征成功标准化。")
        except Exception as e:
            print(f"应用scaler时发生错误: {e}。将使用未标准化的特征。")
    else:
        print("警告: 未提供Scaler，将使用原始全局特征值进行预测。")

    print(f"对 {len(sequence_ids_list)} 条序列进行预测...")
    if tokenized_padded_sequences.shape[0] == 0 or scaled_global_features_np.shape[0] == 0:
        print("警告：处理后的序列数据或全局特征数据为空，无法预测。")
        return pd.DataFrame(columns=['ID', 'Sequence', 'Probability', 'Prediction', 'Label'])
    
    if tokenized_padded_sequences.shape[0] != scaled_global_features_np.shape[0]:
        print(f"错误: 序列数据 ({tokenized_padded_sequences.shape[0]}) 和全局特征数据 ({scaled_global_features_np.shape[0]}) 的样本数量不匹配。")
        error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "数据不匹配"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
        return pd.DataFrame(error_results)

    try:
        logits = k_model.predict([tokenized_padded_sequences, scaled_global_features_np])
        probabilities = tf.sigmoid(logits).numpy().flatten()
        predictions = (probabilities >= threshold).astype(int)
    except Exception as e:
        print(f"Keras模型预测时出错: {e}")
        error_results = [{'ID': sid, 'Sequence': sstr, 'Probability': np.nan, 'Prediction': -1, 'Label': "预测失败"} for sid, sstr in zip(sequence_ids_list, sequence_strings_list)]
        return pd.DataFrame(error_results)

    results_list = []
    for i, seq_id in enumerate(sequence_ids_list):
        results_list.append({
            'ID': seq_id,
            'Sequence': sequence_strings_list[i],
            'Probability': float(probabilities[i]),
            'Prediction': int(predictions[i]),
            'Label': "抗革兰氏阴性菌活性" if predictions[i] == 1 else "非抗革兰氏阴性菌活性"
        })
    results_df = pd.DataFrame(results_list)
    
    if not results_df.empty:
        prediction_counts = Counter(results_df['Label'])
        total = len(results_df)
        print("预测结果统计:")
        for label, count in prediction_counts.items():
            percentage = count / total * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
    return results_df

# 初始化Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# 全局变量
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(APP_ROOT_DIR)
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, 'model', 'Best Hybrid Classifier.keras')
SCALER_PATH_APP = os.path.join(PROJECT_ROOT_DIR, 'model', 'hybrid_model_scaler.pkl')

keras_model_global = None 
global_feature_scaler_app = None 

def load_app_dependencies():
    """加载应用所需的模型和scaler"""
    global keras_model_global, global_feature_scaler_app
    if keras_model_global is None:
        keras_model_global = load_keras_model(KERAS_MODEL_PATH)
    if global_feature_scaler_app is None:
        global_feature_scaler_app = load_app_scaler(SCALER_PATH_APP)
    print("Keras模型和Scaler已加载!")

@app.before_request
def before_first_request_func():
    if keras_model_global is None or global_feature_scaler_app is None:
        load_app_dependencies()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict_sequence_route(): # Renamed to avoid conflict with any potential local 'predict_sequence'
    global keras_model_global, global_feature_scaler_app 
    try:
        temp_fasta = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.fasta")
        
        if 'fasta_file' in request.files and request.files['fasta_file'].filename:
            file = request.files['fasta_file']
            if not file.filename.endswith(('.fasta', '.fa', '.txt')):
                return jsonify({'error': '请上传FASTA格式文件 (.fasta, .fa, .txt)'}), 400
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
            return jsonify({'error': '请上传FASTA文件或输入序列数据'}), 400
        
        if not os.path.exists(temp_fasta) or os.path.getsize(temp_fasta) == 0:
            return jsonify({'error': '创建临时FASTA文件失败或文件为空'}), 400
            
        sequence_count = sum(1 for _ in SeqIO.parse(temp_fasta, "fasta"))
        if sequence_count == 0:
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': '未检测到有效序列'}), 400
        
        MAX_SEQUENCES_ALLOWED = 20000 
        if sequence_count > MAX_SEQUENCES_ALLOWED: 
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': f'序列数量过多 ({sequence_count}). 请限制在{MAX_SEQUENCES_ALLOWED}个以内'}), 400
        
        all_data_df = extract_features_from_fasta(temp_fasta) 
        
        if all_data_df.empty:
            if os.path.exists(temp_fasta): os.remove(temp_fasta)
            return jsonify({'error': '无法从输入序列中提取有效特征'}), 400
        
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
            threshold=0.5 
        )
        
        if os.path.exists(temp_fasta):
            try:
                os.remove(temp_fasta)
            except Exception as e_rm:
                print(f"删除临时FASTA文件 {temp_fasta} 时出错: {e_rm}")

        if results_df.empty and not sequence_ids_list:
             return jsonify({'error': '没有有效序列进行预测'}), 400
        if results_df.empty and sequence_ids_list: # Check if prediction failed for existing sequences
             # Check if all predictions are -1 (error state)
            if all(results_df['Prediction'] == -1):
                # Try to get a more specific error message if available from the Label column
                specific_error = "预测过程中发生错误"
                if 'Label' in results_df.columns and results_df['Label'].nunique() == 1:
                    specific_error = results_df['Label'].iloc[0] # e.g., "特征维度错误"
                return jsonify({'error': f'{specific_error}，未能生成有效结果'}), 500
            # If not all failed, proceed to format results

        results_for_json = []
        for _, row in results_df.iterrows():
            results_for_json.append({
                'id': row['ID'],
                'sequence': row['Sequence'],
                'probability': float(row['Probability']) if pd.notna(row['Probability']) else -1.0,
                'prediction': int(row['Prediction']) if pd.notna(row['Prediction']) else -1,
                'label': row['Label']
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
        
        return jsonify({
            'success': True,
            'results': results_for_json,
            'stats': stats
        })
    
    except Exception as e:
        print(f"预测路由 (/predict) 发生严重错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'处理您的请求时发生内部服务器错误。请稍后再试或联系管理员。'}), 500

@app.route('/export', methods=['POST'])
def export_results():
    """导出预测结果"""
    try:
        data = request.json
        format_type = data.get('format', 'csv')
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': '没有可导出的结果'}), 400
        
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
            return jsonify({'error': '不支持的导出格式'}), 400
    
    except Exception as e:
        print(f"导出错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'导出结果时出错: {str(e)}'}), 500

@app.route('/example')
def get_example():
    """获取示例数据"""
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
