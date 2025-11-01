#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用原型特征生成抗菌肽序列的示例脚本
"""

import os
import sys
import torch
import argparse

# 解决模块导入问题
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enhanced_architecture.generate_conditional_sequences import main

def generate_with_positive_prototype():
    """使用正样本原型生成抗菌肽"""
    print("🧬 使用正样本原型生成抗菌肽...")
    
    # 模拟命令行参数
    sys.argv = [
        'generate_with_prototypes.py',
        '--checkpoint', 'checkpoints_hybrid_650M/best_model_epoch_20.pt',
        '--num_sequences', '50',
        '--use_prototype',
        '--prototype_type', 'positive',
        '--guidance_scale', '7.5',
        '--output_fasta', 'generated_positive_prototype.fasta',
        '--prediction_output', 'predictions/positive_prototype_predictions.txt'
    ]
    
    main()

def generate_with_interpolation():
    """使用插值特征生成中等活性的肽"""
    print("🧬 使用插值特征生成中等活性的肽...")
    
    # 模拟命令行参数
    sys.argv = [
        'generate_with_prototypes.py',
        '--checkpoint', 'checkpoints_hybrid_650M/best_model_epoch_20.pt',
        '--num_sequences', '50',
        '--use_prototype',
        '--interpolation_alpha', '0.7',  # 70%正样本 + 30%负样本
        '--guidance_scale', '5.0',
        '--output_fasta', 'generated_interpolated.fasta',
        '--prediction_output', 'predictions/interpolated_predictions.txt'
    ]
    
    main()

def generate_comparison():
    """生成对比实验：原型 vs 参考序列"""
    print("🧬 生成对比实验...")
    
    # 1. 使用原型特征
    print("\n1. 使用原型特征生成...")
    sys.argv = [
        'generate_with_prototypes.py',
        '--checkpoint', 'checkpoints_hybrid_650M/best_model_epoch_20.pt',
        '--num_sequences', '100',
        '--use_prototype',
        '--prototype_type', 'positive',
        '--output_fasta', 'comparison_prototype.fasta',
        '--prediction_output', 'predictions/comparison_prototype.txt'
    ]
    main()
    
    # 2. 使用参考序列
    print("\n2. 使用参考序列生成...")
    sys.argv = [
        'generate_with_prototypes.py',
        '--checkpoint', 'checkpoints_hybrid_650M/best_model_epoch_20.pt',
        '--num_sequences', '100',
        '--num_references', '5',
        '--output_fasta', 'comparison_references.fasta',
        '--prediction_output', 'predictions/comparison_references.txt'
    ]
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原型特征生成示例")
    parser.add_argument("--mode", type=str, default="positive", 
                       choices=["positive", "interpolation", "comparison"],
                       help="生成模式")
    
    args = parser.parse_args()
    
    if args.mode == "positive":
        generate_with_positive_prototype()
    elif args.mode == "interpolation":
        generate_with_interpolation()
    elif args.mode == "comparison":
        generate_comparison()
    
    print("✅ 原型特征生成完成！")