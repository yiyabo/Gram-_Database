#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
云部署精简版 - Gram-Negative Bacteria Prediction Service
移除了生成功能和复杂可视化，专注于核心预测功能
"""

import os
import uuid
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from io import StringIO, BytesIO
from Bio import SeqIO
import pickle
import traceback

# 简化的氨基酸特征计算（替代peptides库）
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# 氨基酸基本属性表（替代peptides库的复杂计算）
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'polar': False},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'polar': False},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polar': True},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polar': True},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'polar': False},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'polar': False},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1, 'polar': True},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'polar': False},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'polar': True},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'polar': False},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'polar': False},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'polar': True},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'polar': False},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polar': True},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'polar': True},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'polar': True},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'polar': True},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'polar': False},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'polar': False},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polar': True}
}

def calculate_simple_features(sequence):
    """简化的特征计算，替代peptides库"""
    if not sequence or not all(aa in AA_PROPERTIES for aa in sequence):
        return None
    
    length = len(sequence)
    
    # 基本特征
    charge = sum(AA_PROPERTIES[aa]['charge'] for aa in sequence)
    hydrophobicity = sum(AA_PROPERTIES[aa]['hydrophobicity'] for aa in sequence) / length
    polar_count = sum(1 for aa in sequence if AA_PROPERTIES[aa]['polar'])
    
    # 氨基酸组成
    aa_composition = {f'AA_{aa}': sequence.count(aa) / length for aa in AMINO_ACIDS}
    
    # 简化的理化性质（近似计算）
    aliphatic_index = (sequence.count('A') + 2.9*sequence.count('V') + 3.9*(sequence.count('I') + sequence.count('L'))) / length * 100
    instability_index = min(abs(charge) * 10 + polar_count / length * 50, 100)  # 简化估算
    isoelectric_point = 7.0 + charge * 0.5  # 简化估算
    hydrophobic_moment = abs(hydrophobicity) * (1 + polar_count / length)  # 简化估算
    hydrophilicity = -hydrophobicity  # 简化为负的疏水性
    
    features = {
        'Length': float(length),
        'Charge': float(charge),
        'Hydrophobicity': float(hydrophobicity),
        'Hydrophobic_Moment': float(hydrophobic_moment),
        'Instability_Index': float(instability_index),
        'Isoelectric_Point': float(isoelectric_point),
        'Aliphatic_Index': float(aliphatic_index),
        'Hydrophilicity': float(hydrophilicity)
    }
    features.update(aa_composition)
    
    return features

def extract_features_from_fasta_simple(fasta_file):
    """简化的FASTA特征提取"""
    records_data = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        peptide_id = record.id
        sequence = str(record.seq).upper()
        
        if not sequence or any(aa not in AMINO_ACIDS for aa in sequence):
            print(f"Skipping invalid sequence {peptide_id}: {sequence}")
            continue
        
        features = calculate_simple_features(sequence)
        if features:
            features['ID'] = peptide_id
            features['Sequence'] = sequence
            records_data.append(features)
    
    return pd.DataFrame(records_data)

# 保留原有的模型相关函数（简化版）
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)
MAX_SEQUENCE_LENGTH = 128

def simple_predict(sequences_df, threshold=0.8):
    """简化的预测函数 - 使用基于规则的方法作为后备"""
    results = []
    
    for _, row in sequences_df.iterrows():
        # 基于序列特征的简单规则预测（作为模型的后备）
        charge = row.get('Charge', 0)
        hydrophobicity = row.get('Hydrophobicity', 0)
        length = row.get('Length', 0)
        
        # 简化的预测逻辑（基于抗菌肽的一般特征）
        score = 0.0
        if charge > 2:  # 正电荷有利于抗菌活性
            score += 0.3
        if 10 <= length <= 50:  # 适中长度
            score += 0.2
        if -1 <= hydrophobicity <= 1:  # 适中疏水性
            score += 0.2
        if row.get('AA_K', 0) + row.get('AA_R', 0) > 0.1:  # 富含碱性氨基酸
            score += 0.3
        
        prediction = 1 if score >= threshold * 0.8 else 0
        
        results.append({
            'ID': row['ID'],
            'Sequence': row['Sequence'],
            'Probability': float(score),
            'Prediction': int(prediction),
            'Label': "Anti-Gram-Negative" if prediction == 1 else "Non-Anti-Gram-Negative"
        })
    
    return pd.DataFrame(results)

# Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return """
    <html>
    <head><title>Gram-Negative Bacteria Prediction (Cloud Version)</title></head>
    <body>
        <h1>Gram-Negative Bacteria Prediction Service</h1>
        <p>云部署精简版 - 专注核心预测功能</p>
        
        <h2>预测功能</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <h3>上传FASTA文件：</h3>
            <input type="file" name="fasta_file" accept=".fasta,.fa,.txt">
            
            <h3>或直接输入序列：</h3>
            <textarea name="fasta_text" rows="6" cols="80" placeholder="请输入FASTA格式序列...">
>Example1
GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV
>Example2
YVPLPNVPQPGRRPFPTFPGQGPFNPKIKWPQGY
            </textarea>
            
            <br><br>
            <input type="submit" value="开始预测">
        </form>
        
        <h2>API接口</h2>
        <p>POST /api/predict - JSON格式预测接口</p>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        temp_fasta = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.fasta")
        
        # 处理文件上传或文本输入
        if 'fasta_file' in request.files and request.files['fasta_file'].filename:
            file = request.files['fasta_file']
            file.save(temp_fasta)
        elif 'fasta_text' in request.form and request.form['fasta_text'].strip():
            fasta_text = request.form['fasta_text']
            with open(temp_fasta, 'w') as f:
                if '>' not in fasta_text:
                    # 如果不是FASTA格式，自动添加序列头
                    lines = fasta_text.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip():
                            f.write(f">Seq_{i+1}\n{line.strip()}\n")
                else:
                    f.write(fasta_text)
        else:
            return jsonify({'error': '请上传FASTA文件或输入序列数据'}), 400
        
        # 提取特征
        sequences_df = extract_features_from_fasta_simple(temp_fasta)
        
        if sequences_df.empty:
            return jsonify({'error': '未检测到有效序列'}), 400
        
        # 进行预测
        results_df = simple_predict(sequences_df)
        
        # 格式化结果
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'id': row['ID'],
                'sequence': row['Sequence'],
                'probability': row['Probability'],
                'prediction': row['Prediction'],
                'label': row['Label']
            })
        
        # 统计信息
        stats = {
            'total': len(results),
            'positive': sum(1 for r in results if r['prediction'] == 1),
            'negative': sum(1 for r in results if r['prediction'] == 0),
            'avg_probability': np.mean([r['probability'] for r in results])
        }
        
        # 清理临时文件
        if os.path.exists(temp_fasta):
            os.remove(temp_fasta)
        
        if request.path == '/api/predict':
            return jsonify({
                'success': True,
                'results': results,
                'stats': stats,
                'note': '这是精简版预测，使用基于规则的方法'
            })
        else:
            # HTML响应
            html_results = "<h2>预测结果</h2><table border='1'>"
            html_results += "<tr><th>ID</th><th>序列</th><th>概率</th><th>预测</th><th>标签</th></tr>"
            for r in results:
                html_results += f"<tr><td>{r['id']}</td><td>{r['sequence'][:50]}...</td><td>{r['probability']:.3f}</td><td>{r['prediction']}</td><td>{r['label']}</td></tr>"
            html_results += "</table>"
            html_results += f"<p>统计：总计{stats['total']}个序列，{stats['positive']}个阳性，{stats['negative']}个阴性</p>"
            html_results += "<a href='/'>返回首页</a>"
            return f"<html><body>{html_results}</body></html>"
        
    except Exception as e:
        print(f"预测错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'cloud-minimal'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
