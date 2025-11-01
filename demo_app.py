#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
极简版云平台启动文件
移除机器学习依赖，仅保留基本Web界面
适用于演示和快速部署
"""

import os
import sys
from flask import Flask, render_template, jsonify

# 创建Flask应用
app = Flask(__name__)

# 基础配置
app.config.update(
    DEBUG=False,
    SECRET_KEY=os.environ.get('SECRET_KEY', 'demo-secret-key'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB
)

@app.route('/')
def index():
    """主页"""
    return jsonify({
        "message": "Gram-Negative Bacteria Prediction System",
        "status": "Demo Version - Running Successfully!",
        "version": "Cloud Demo 1.0",
        "features": [
            "Web Interface Available",
            "Database Viewer (Coming Soon)",
            "ML Prediction (Requires Full Deployment)"
        ]
    })

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'message': 'Demo version running successfully',
        'timestamp': '2025-08-03T15:15:00Z',
        'services': {
            'web_interface': 'available',
            'prediction_model': 'not_loaded_in_demo',
            'generation_service': 'not_loaded_in_demo'
        }
    })

@app.route('/predict')
def predict_demo():
    """预测功能演示页面"""
    return jsonify({
        "message": "Prediction Service",
        "status": "Demo Mode",
        "note": "Full ML prediction requires server deployment with GPU/CPU resources"
    })

@app.route('/generate')
def generate_demo():
    """生成功能演示页面"""
    return jsonify({
        "message": "Sequence Generation Service", 
        "status": "Demo Mode",
        "note": "Full generation requires ESM-2 and diffusion models"
    })

@app.route('/database')
def database_demo():
    """数据库查看演示"""
    return jsonify({
        "message": "Database Viewer",
        "status": "Demo Mode",
        "total_sequences": "7285+",
        "note": "Full database access requires complete deployment"
    })

@app.route('/about')
def about():
    """关于页面"""
    return jsonify({
        "project": "Gram-Negative Bacteria Prediction System",
        "description": "AI-powered antimicrobial peptide prediction and generation",
        "technologies": [
            "Flask Web Framework",
            "Deep Learning (LSTM + MLP)",
            "Diffusion Models",
            "ESM-2 Protein Language Model",
            "Bioinformatics Pipeline"
        ],
        "demo_limitations": [
            "No ML model loading (memory constraints)",
            "No file upload processing",
            "No sequence generation",
            "Interface demonstration only"
        ],
        "full_features": [
            "Upload FASTA files for prediction",
            "Generate novel antimicrobial peptides", 
            "Analyze biochemical properties",
            "Export results in multiple formats"
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 启动Demo版本，端口: {port}")
    print("📝 注意: 这是演示版本，不包含ML功能")
    app.run(host='0.0.0.0', port=port, debug=False)
