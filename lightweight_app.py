import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
from Bio import SeqIO
from io import StringIO
import tempfile
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='gram_predictor/templates',
            static_folder='gram_predictor/static')

app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# 模拟预测函数（用于演示）
def mock_predict_sequence(sequence):
    """模拟预测函数 - 基于序列长度和氨基酸组成的简单规则"""
    # 简单的启发式规则
    length_score = min(len(sequence) / 50, 1.0) * 0.3
    
    # 计算正电荷氨基酸比例
    positive_aa = sequence.count('K') + sequence.count('R') + sequence.count('H')
    positive_ratio = positive_aa / len(sequence) if len(sequence) > 0 else 0
    charge_score = min(positive_ratio * 2, 1.0) * 0.4
    
    # 疏水性氨基酸
    hydrophobic_aa = sequence.count('A') + sequence.count('I') + sequence.count('L') + sequence.count('V')
    hydrophobic_ratio = hydrophobic_aa / len(sequence) if len(sequence) > 0 else 0
    hydrophobic_score = min(hydrophobic_ratio * 1.5, 1.0) * 0.3
    
    # 组合得分
    final_score = length_score + charge_score + hydrophobic_score
    return min(final_score, 1.0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # 获取输入
        sequence_input = request.form.get('sequence', '').strip()
        file_input = request.files.get('file')
        
        results = []
        
        if file_input and file_input.filename:
            # 处理文件输入
            content = file_input.read().decode('utf-8')
            sequences = []
            
            if file_input.filename.endswith('.fasta'):
                # FASTA 格式
                fasta_io = StringIO(content)
                for record in SeqIO.parse(fasta_io, 'fasta'):
                    sequences.append((record.id, str(record.seq)))
            else:
                # 纯文本格式
                lines = content.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        sequences.append((f"Sequence_{i+1}", line.strip()))
            
            # 预测每个序列
            for seq_id, seq in sequences:
                if seq:
                    score = mock_predict_sequence(seq)
                    results.append({
                        'id': seq_id,
                        'sequence': seq,
                        'score': score,
                        'prediction': 'Antimicrobial' if score > 0.5 else 'Non-antimicrobial'
                    })
        
        elif sequence_input:
            # 处理单个序列
            score = mock_predict_sequence(sequence_input)
            results.append({
                'id': 'Input_Sequence',
                'sequence': sequence_input,
                'score': score,
                'prediction': 'Antimicrobial' if score > 0.5 else 'Non-antimicrobial'
            })
        
        if not results:
            flash('请输入序列或上传文件', 'error')
            return redirect(url_for('predict'))
        
        return render_template('predict.html', results=results)
        
    except Exception as e:
        logger.error(f"预测错误: {str(e)}")
        flash(f'预测过程中出现错误: {str(e)}', 'error')
        return redirect(url_for('predict'))

@app.route('/generate')
def generate():
    return render_template('generate.html', 
                         message="演示版本暂不支持序列生成功能。完整版本包含基于扩散模型的序列生成。")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence', '')
        
        if not sequence:
            return jsonify({'error': '序列不能为空'}), 400
        
        score = mock_predict_sequence(sequence)
        
        return jsonify({
            'sequence': sequence,
            'score': score,
            'prediction': 'Antimicrobial' if score > 0.5 else 'Non-antimicrobial',
            'note': '这是演示版本的模拟预测结果'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'lightweight'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
