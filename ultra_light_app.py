import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import re

# 设置日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='gram_predictor/templates',
            static_folder='gram_predictor/static')

app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

def simple_amp_prediction(sequence):
    """超简单的抗菌肽预测算法 - 基于氨基酸特征"""
    if not sequence or len(sequence) < 5:
        return 0.1
    
    # 清理序列
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    if len(sequence) == 0:
        return 0.1
    
    # 计算基础特征
    length = len(sequence)
    
    # 正电荷氨基酸 (K, R, H)
    positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
    positive_ratio = positive / length
    
    # 疏水性氨基酸 (A, I, L, V, F, M, W, P)
    hydrophobic = (sequence.count('A') + sequence.count('I') + sequence.count('L') + 
                   sequence.count('V') + sequence.count('F') + sequence.count('M') + 
                   sequence.count('W') + sequence.count('P'))
    hydrophobic_ratio = hydrophobic / length
    
    # 芳香族氨基酸 (F, W, Y)
    aromatic = sequence.count('F') + sequence.count('W') + sequence.count('Y')
    aromatic_ratio = aromatic / length
    
    # 简单评分规则
    score = 0.0
    
    # 长度评分 (10-50最理想)
    if 10 <= length <= 50:
        score += 0.3
    elif 5 <= length <= 70:
        score += 0.15
    
    # 正电荷评分
    if positive_ratio >= 0.2:
        score += 0.4
    elif positive_ratio >= 0.1:
        score += 0.2
    
    # 疏水性评分
    if 0.3 <= hydrophobic_ratio <= 0.7:
        score += 0.2
    elif 0.2 <= hydrophobic_ratio <= 0.8:
        score += 0.1
    
    # 芳香族评分
    if aromatic_ratio >= 0.1:
        score += 0.1
    
    return min(score, 1.0)

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
            lines = content.strip().split('\n')
            
            current_seq = ""
            current_id = "Sequence_1"
            seq_count = 1
            
            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    # 处理前一个序列
                    if current_seq:
                        score = simple_amp_prediction(current_seq)
                        results.append({
                            'id': current_id,
                            'sequence': current_seq,
                            'score': score,
                            'prediction': 'Antimicrobial' if score > 0.5 else 'Non-antimicrobial'
                        })
                    # 开始新序列
                    current_id = line[1:] if len(line) > 1 else f"Sequence_{seq_count}"
                    current_seq = ""
                    seq_count += 1
                elif line:
                    current_seq += line
            
            # 处理最后一个序列
            if current_seq:
                score = simple_amp_prediction(current_seq)
                results.append({
                    'id': current_id,
                    'sequence': current_seq,
                    'score': score,
                    'prediction': 'Antimicrobial' if score > 0.5 else 'Non-antimicrobial'
                })
        
        elif sequence_input:
            # 处理单个序列
            score = simple_amp_prediction(sequence_input)
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
                         message="演示版本：序列生成功能需要完整版本支持")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence', '')
        
        if not sequence:
            return jsonify({'error': '序列不能为空'}), 400
        
        score = simple_amp_prediction(sequence)
        
        return jsonify({
            'sequence': sequence,
            'score': round(score, 3),
            'prediction': 'Antimicrobial' if score > 0.5 else 'Non-antimicrobial',
            'confidence': 'High' if abs(score - 0.5) > 0.3 else 'Medium',
            'note': 'Ultra-lightweight prediction using amino acid composition analysis'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'version': 'ultra-lightweight',
        'features': ['basic_prediction', 'file_upload', 'api'],
        'dependencies': 'minimal'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
