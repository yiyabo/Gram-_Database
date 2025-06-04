#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估和序列生成脚本
加载训练好的模型，生成抗菌肽序列并进行全面评估
"""

import torch
import numpy as np
from pathlib import Path
import logging
import os
import sys
from typing import List, Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import get_config
from main_trainer import EnhancedAMPTrainer
from evaluation.evaluator import ModelEvaluator, EvaluationMetrics
from data_loader import tokens_to_sequence

class ModelTester:
    """模型测试和评估器"""
    
    def __init__(self, config_name: str = "production", checkpoint_name: str = "best.pt"):
        """
        初始化模型测试器
        
        Args:
            config_name: 配置名称 (production/quick_test)
            checkpoint_name: 检查点文件名 (best.pt/latest.pt)
        """
        self.config_name = config_name
        self.config = get_config(config_name)
        self.checkpoint_path = Path(self.config.training.output_dir) / "checkpoints" / checkpoint_name
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 检查GPU可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self.trainer = None
        self.evaluator = None
        self.data_loader = None
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('model_evaluation.log')
            ]
        )
    
    def load_model(self):
        """加载训练好的模型"""
        self.logger.info(f"加载模型检查点: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
        
        # 初始化训练器
        self.trainer = EnhancedAMPTrainer(config_name=self.config_name)
        
        # 初始化模型组件
        self.trainer.initialize_models()
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载模型状态
        self.trainer.esm2_encoder.load_state_dict(checkpoint['esm2_encoder_state_dict'])
        self.trainer.diffusion_model.model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        
        # 设置为评估模式
        self.trainer.esm2_encoder.eval()
        self.trainer.diffusion_model.model.eval()
        
        self.logger.info(f"模型加载成功，训练epoch: {checkpoint.get('epoch', 'unknown')}")
        self.logger.info(f"最佳验证损失: {checkpoint.get('best_val_loss', 'unknown')}")
    
    def initialize_evaluator(self):
        """初始化评估器"""
        self.evaluator = ModelEvaluator(self.config.evaluation)
        
        self.logger.info("评估器初始化完成")
    
    def generate_sequences(self, num_samples: int = 100, max_length: int = 50) -> List[str]:
        """
        生成抗菌肽序列
        
        Args:
            num_samples: 生成样本数量
            max_length: 最大序列长度
            
        Returns:
            生成的序列列表
        """
        if self.trainer is None:
            raise RuntimeError("请先加载模型")
        
        self.logger.info(f"生成 {num_samples} 个序列，最大长度: {max_length}")
        
        try:
            # 使用扩散模型直接生成，并添加调试信息
            self.trainer.diffusion_model.model.eval()
            
            with torch.no_grad():
                # 生成token序列
                generated_tokens = self.trainer.diffusion_model.sample(
                    batch_size=num_samples,
                    seq_len=max_length,
                    num_inference_steps=self.config.diffusion.num_inference_steps
                )
                
                # 调试信息：打印原始token值
                self.logger.info(f"生成的原始token张量形状: {generated_tokens.shape}")
                self.logger.info(f"Token值范围: {generated_tokens.min().item()} - {generated_tokens.max().item()}")
                
                # 检查词汇表映射
                from data_loader import VOCAB_TO_AA
                self.logger.info(f"词汇表映射: {VOCAB_TO_AA}")
                
                # 转换为氨基酸序列
                sequences = []
                for i, seq_tokens in enumerate(generated_tokens):
                    # 调试信息：打印每个序列的token值
                    tokens_numpy = seq_tokens.cpu().numpy()
                    self.logger.info(f"序列 {i+1} 原始tokens前10个: {tokens_numpy[:10]}")
                    
                    # 手动检查转换过程
                    sequence_parts = []
                    token_debug = []
                    for j, token in enumerate(tokens_numpy[:10]):  # 只检查前10个token
                        aa = VOCAB_TO_AA.get(int(token), 'X')
                        token_debug.append(f"{int(token)}→{aa}")
                        if aa != 'PAD':
                            sequence_parts.append(aa)
                    
                    self.logger.info(f"序列 {i+1} token转换: {token_debug}")
                    
                    # 使用标准转换函数
                    from data_loader import tokens_to_sequence
                    auto_seq = tokens_to_sequence(seq_tokens.cpu().numpy())
                    
                    self.logger.info(f"序列 {i+1} 转换结果: '{auto_seq}' (长度: {len(auto_seq)})")
                    
                    sequences.append(auto_seq)
            
            self.logger.info(f"序列生成完成，成功生成 {len(sequences)} 个序列")
            return sequences
            
        except Exception as e:
            self.logger.error(f"序列生成失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def evaluate_sequences(self, sequences: List[str]) -> EvaluationMetrics:
        """
        评估生成的序列
        
        Args:
            sequences: 待评估的序列列表
            
        Returns:
            评估指标
        """
        if self.evaluator is None:
            raise RuntimeError("请先初始化评估器")
        
        self.logger.info(f"开始评估 {len(sequences)} 个序列")
        
        # 执行评估
        metrics = self.evaluator.evaluate_generated_sequences(sequences)
        
        self.logger.info("序列评估完成")
        return metrics
    
    def save_results(self, sequences: List[str], metrics: EvaluationMetrics, 
                    output_dir: str = "evaluation_results"):
        """
        保存评估结果
        
        Args:
            sequences: 生成的序列
            metrics: 评估指标
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存生成的序列
        sequences_file = output_path / "generated_sequences.txt"
        with open(sequences_file, 'w') as f:
            for i, seq in enumerate(sequences, 1):
                f.write(f"Sequence_{i:03d}: {seq}\n")
        
        # 保存评估报告
        report_file = output_path / "evaluation_report.txt"
        report = self.evaluator.generate_evaluation_report(metrics, str(report_file))
        
        # 保存指标数据
        metrics_file = output_path / "metrics.txt"
        metrics_dict = metrics.to_dict()
        with open(metrics_file, 'w') as f:
            f.write("评估指标详细数据:\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.6f}\n")
        
        self.logger.info(f"结果已保存到目录: {output_path}")
        
        return {
            'sequences_file': str(sequences_file),
            'report_file': str(report_file),
            'metrics_file': str(metrics_file)
        }
    
    def analyze_training_results(self):
        """分析训练结果"""
        self.logger.info("分析训练结果...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n" + "="*60)
        print("训练结果分析")
        print("="*60)
        
        print(f"检查点文件: {self.checkpoint_path}")
        print(f"训练Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"最佳验证损失: {checkpoint.get('best_val_loss', 'unknown'):.6f}")
        
        # 分析模型大小
        diffusion_params = sum(p.numel() for p in checkpoint['diffusion_model_state_dict'].values())
        esm2_params = sum(p.numel() for p in checkpoint['esm2_encoder_state_dict'].values())
        
        print(f"\n模型参数统计:")
        print(f"- 扩散模型参数: {diffusion_params:,}")
        print(f"- ESM-2编码器参数: {esm2_params:,}")
        print(f"- 总参数量: {diffusion_params + esm2_params:,}")
        
        print("\n配置信息:")
        print(f"- 词汇表大小: {self.config.diffusion.vocab_size}")
        print(f"- 最大序列长度: {self.config.diffusion.max_seq_len}")
        print(f"- 隐藏维度: {self.config.diffusion.hidden_dim}")
        print(f"- 扩散步数: {self.config.diffusion.num_timesteps}")
        
        print("="*60)
    
    def run_complete_evaluation(self, num_samples: int = 100, max_length: int = 50):
        """
        运行完整的模型评估流程
        
        Args:
            num_samples: 生成样本数量
            max_length: 最大序列长度
        """
        try:
            # 1. 分析训练结果
            self.analyze_training_results()
            
            # 2. 加载模型
            self.load_model()
            
            # 3. 初始化评估器
            self.initialize_evaluator()
            
            # 4. 生成序列
            sequences = self.generate_sequences(num_samples, max_length)
            
            if not sequences:
                self.logger.error("未能生成任何有效序列")
                return
            
            # 5. 评估序列
            metrics = self.evaluate_sequences(sequences)
            
            # 6. 保存结果
            file_paths = self.save_results(sequences, metrics)
            
            # 7. 显示结果
            self.display_results(sequences, metrics)
            
            self.logger.info("完整评估流程完成！")
            
            return {
                'sequences': sequences,
                'metrics': metrics,
                'files': file_paths
            }
            
        except Exception as e:
            self.logger.error(f"评估过程中出错: {e}")
            raise
    
    def display_results(self, sequences: List[str], metrics: EvaluationMetrics):
        """显示评估结果摘要"""
        print("\n" + "="*60)
        print("模型评估结果摘要")
        print("="*60)
        
        print(f"生成序列数量: {len(sequences)}")
        print(f"有效序列比例: {metrics.valid_sequences_ratio:.1%}")
        print(f"平均序列长度: {metrics.avg_sequence_length:.1f}")
        print(f"序列多样性分数: {metrics.sequence_diversity_score:.3f}")
        print(f"预测活性序列比例: {metrics.predicted_activity_ratio:.1%}")
        print(f"平均活性分数: {metrics.avg_activity_score:.3f}")
        
        print(f"\n前10个生成序列示例:")
        print("-" * 40)
        for i, seq in enumerate(sequences[:10], 1):
            print(f"{i:2d}. {seq}")
        
        if len(sequences) > 10:
            print(f"... 还有 {len(sequences) - 10} 个序列")
        
        print("="*60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="抗菌肽生成模型评估")
    parser.add_argument("--config", default="production", help="配置名称")
    parser.add_argument("--checkpoint", default="best.pt", help="检查点文件名")
    parser.add_argument("--num_samples", type=int, default=100, help="生成样本数量")
    parser.add_argument("--max_length", type=int, default=50, help="最大序列长度")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelTester(args.config, args.checkpoint)
    
    # 运行完整评估
    results = tester.run_complete_evaluation(args.num_samples, args.max_length)
    
    print(f"\n✅ 评估完成！结果已保存到 evaluation_results/ 目录")

if __name__ == "__main__":
    main()
