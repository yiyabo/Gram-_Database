#!/usr/bin/env python3
"""
使用训练好的生成器和hybrid_predict.py生成10000条序列并计算真实预测得分
- 生成器：使用您训练的扩散模型
- 分类器：调用hybrid_predict.py脚本
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import tempfile
import subprocess
import logging
from typing import List, Tuple
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'enhanced_architecture'))

# 导入生成器模块
from enhanced_architecture.config.model_config import get_config
from enhanced_architecture.esm2_auxiliary_encoder import ESM2AuxiliaryEncoder
from enhanced_architecture.diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet
from enhanced_architecture.data_loader import tokens_to_sequence

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSequenceGenerator:
    """使用生成器 + hybrid_predict.py 的序列生成器"""
    
    def __init__(self, 
                 generator_checkpoint: str,
                 hybrid_predict_script: str = "hybrid_predict.py",
                 model_weights: str = "model/best_weights.h5",
                 scaler_path: str = "model/scaler.pkl", 
                 feature_names_file: str = "data/feature_names.txt",
                 config_name: str = "dual_4090"):
        """
        初始化混合生成器
        
        Args:
            generator_checkpoint: 生成器检查点路径
            hybrid_predict_script: hybrid_predict.py脚本路径
            model_weights: 分类器权重路径
            scaler_path: 标准化器路径
            feature_names_file: 特征名称文件路径
            config_name: 配置名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator_checkpoint = generator_checkpoint
        self.hybrid_predict_script = hybrid_predict_script
        self.model_weights = model_weights
        self.scaler_path = scaler_path
        self.feature_names_file = feature_names_file
        self.config_name = config_name
        
        # 模型组件
        self.generator_model = None
        self.config = None
        
        logger.info(f"初始化混合生成器，设备: {self.device}")
        logger.info(f"生成器检查点: {self.generator_checkpoint}")
        logger.info(f"分类器脚本: {self.hybrid_predict_script}")
    
    def load_generator(self):
        """加载生成器模型"""
        logger.info(f"加载生成器: {self.generator_checkpoint}")
        
        if not os.path.exists(self.generator_checkpoint):
            raise FileNotFoundError(f"生成器检查点不存在: {self.generator_checkpoint}")
        
        # 加载配置
        self.config = get_config(self.config_name)
        
        # 加载检查点
        checkpoint = torch.load(self.generator_checkpoint, map_location=self.device, weights_only=False)
        
        # 初始化ESM-2编码器
        self.esm2_encoder = ESM2AuxiliaryEncoder(self.config.esm2)
        if 'esm2_encoder_state_dict' in checkpoint:
            self.esm2_encoder.load_state_dict(checkpoint['esm2_encoder_state_dict'])
        self.esm2_encoder.to(self.device)
        self.esm2_encoder.eval()
        
        # 初始化扩散模型
        scheduler = D3PMScheduler(
            num_timesteps=self.config.diffusion.num_timesteps,
            schedule_type=self.config.diffusion.schedule_type,
            vocab_size=self.config.diffusion.vocab_size
        )
        
        unet = D3PMUNet(
            vocab_size=self.config.diffusion.vocab_size,
            max_seq_len=self.config.diffusion.max_seq_len,
            hidden_dim=self.config.diffusion.hidden_dim,
            num_layers=self.config.diffusion.num_layers,
            num_heads=self.config.diffusion.num_heads,
            dropout=self.config.diffusion.dropout
        )
        
        if 'diffusion_model_state_dict' in checkpoint:
            unet.load_state_dict(checkpoint['diffusion_model_state_dict'])
        
        self.generator_model = D3PMDiffusion(
            model=unet,
            scheduler=scheduler,
            device=self.device
        )
        
        logger.info("✅ 生成器加载成功")
    
    def check_classifier_dependencies(self):
        """检查分类器依赖文件"""
        dependencies = [
            self.hybrid_predict_script,
            self.model_weights,
            self.scaler_path,
            self.feature_names_file
        ]
        
        missing_files = []
        for file_path in dependencies:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"缺少分类器依赖文件: {missing_files}")
            raise FileNotFoundError(f"缺少分类器依赖文件: {missing_files}")
        
        logger.info("✅ 分类器依赖文件检查通过")
    
    def generate_batch_sequences(self, batch_size: int = 50, seq_length: int = 50,
                               temperature: float = 0.8) -> List[str]:
        """
        生成一批序列
        
        Args:
            batch_size: 批次大小
            seq_length: 序列长度
            temperature: 采样温度
        
        Returns:
            生成的序列列表
        """
        with torch.no_grad():
            # 使用多种采样策略增加多样性
            strategy = np.random.choice(['standard', 'low_temp', 'high_temp'], p=[0.5, 0.3, 0.2])
            
            if strategy == 'standard':
                generated_tokens = self.generator_model.sample(
                    batch_size=batch_size,
                    seq_len=seq_length,
                    temperature=temperature
                )
            elif strategy == 'low_temp':
                generated_tokens = self.generator_model.sample(
                    batch_size=batch_size,
                    seq_len=seq_length,
                    temperature=temperature * 0.7
                )
            else:  # high_temp
                generated_tokens = self.generator_model.sample(
                    batch_size=batch_size,
                    seq_len=seq_length,
                    temperature=temperature * 1.3
                )
        
        # 转换为氨基酸序列
        sequences = []
        for tokens in generated_tokens:
            seq = tokens_to_sequence(tokens.cpu().numpy())
            # 过滤掉太短的序列
            if len(seq) >= 10:
                sequences.append(seq)
        
        return sequences
    
    def predict_sequences_with_hybrid_script(self, sequences: List[str]) -> Tuple[List[float], List[int]]:
        """
        使用hybrid_predict.py脚本预测序列
        
        Args:
            sequences: 序列列表
        
        Returns:
            (概率列表, 预测标签列表)
        """
        if not sequences:
            return [], []
        
        # 创建临时FASTA文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            for i, seq in enumerate(sequences):
                f.write(f">Generated_Seq_{i+1}\n{seq}\n")
            temp_fasta = f.name
        
        # 创建临时输出文件
        temp_output = tempfile.mktemp(suffix='.txt')
        
        try:
            # 构建hybrid_predict.py命令
            cmd = [
                sys.executable, self.hybrid_predict_script,
                "--model_path", self.model_weights,
                "--fasta_file", temp_fasta,
                "--scaler_path", self.scaler_path,
                "--output_file", temp_output,
                "--threshold", "0.5",
                "--feature_names_file", self.feature_names_file
            ]
            
            # 执行预测脚本
            logger.debug(f"执行预测命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning(f"hybrid_predict.py执行失败: {result.stderr}")
                return [], []
            
            # 读取预测结果
            if not os.path.exists(temp_output):
                logger.warning("预测输出文件不存在")
                return [], []
            
            probabilities = []
            predictions = []
            
            with open(temp_output, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # 跳过标题行
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        pred_label = parts[1]
                        prob_str = parts[2]
                        
                        try:
                            probability = float(prob_str)
                            prediction = 1 if "抗革兰氏阴性菌活性" in pred_label else 0
                            
                            probabilities.append(probability)
                            predictions.append(prediction)
                        except ValueError:
                            logger.warning(f"解析预测结果失败: {line.strip()}")
                            probabilities.append(0.0)
                            predictions.append(0)
            
            return probabilities, predictions
            
        except subprocess.TimeoutExpired:
            logger.error("hybrid_predict.py执行超时")
            return [], []
        except Exception as e:
            logger.error(f"调用hybrid_predict.py失败: {e}")
            return [], []
        finally:
            # 清理临时文件
            for temp_file in [temp_fasta, temp_output]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def generate_10k_sequences_with_scores(self, 
                                         output_file: str = "generated_10k_with_hybrid_scores.fasta",
                                         batch_size: int = 50,
                                         base_seq_length: int = 50) -> None:
        """
        生成10000条序列并使用hybrid_predict.py计算得分
        
        Args:
            output_file: 输出文件路径
            batch_size: 批次大小
            base_seq_length: 基础序列长度
        """
        logger.info("🚀 开始生成10000条序列并使用hybrid_predict.py计算得分...")
        
        all_sequences = []
        all_probabilities = []
        all_predictions = []
        
        total_batches = (10000 + batch_size - 1) // batch_size
        
        with tqdm(total=10000, desc="生成和预测序列") as pbar:
            for batch_idx in range(total_batches):
                # 计算当前批次大小
                current_batch_size = min(batch_size, 10000 - len(all_sequences))
                
                if current_batch_size <= 0:
                    break
                
                # 随机序列长度（在合理范围内）
                current_seq_length = np.random.randint(25, 65)
                
                # 随机温度（增加多样性）
                temperature = np.random.uniform(0.7, 1.2)
                
                try:
                    # 生成序列
                    sequences = self.generate_batch_sequences(
                        batch_size=current_batch_size,
                        seq_length=current_seq_length,
                        temperature=temperature
                    )
                    
                    if not sequences:
                        logger.warning(f"批次 {batch_idx} 没有生成有效序列")
                        continue
                    
                    # 预测序列
                    probabilities, predictions = self.predict_sequences_with_hybrid_script(sequences)
                    
                    if len(probabilities) != len(sequences):
                        logger.warning(f"批次 {batch_idx} 预测结果数量不匹配: {len(probabilities)} vs {len(sequences)}")
                        # 补齐缺失的预测结果
                        while len(probabilities) < len(sequences):
                            probabilities.append(0.0)
                            predictions.append(0)
                    
                    # 添加到总结果
                    all_sequences.extend(sequences)
                    all_probabilities.extend(probabilities)
                    all_predictions.extend(predictions)
                    
                    pbar.update(len(sequences))
                    
                    if len(all_sequences) >= 10000:
                        break
                        
                except Exception as e:
                    logger.warning(f"批次 {batch_idx} 处理失败: {e}")
                    continue
        
        # 确保正好有10000条序列
        all_sequences = all_sequences[:10000]
        all_probabilities = all_probabilities[:10000]
        all_predictions = all_predictions[:10000]
        
        logger.info(f"✅ 成功生成并预测 {len(all_sequences)} 条序列")
        
        # 保存到FASTA文件
        self.save_sequences_with_scores(
            sequences=all_sequences,
            probabilities=all_probabilities,
            predictions=all_predictions,
            output_file=output_file
        )
        
        # 打印统计信息
        self.print_statistics(all_sequences, all_probabilities, all_predictions)
    
    def save_sequences_with_scores(self, sequences: List[str], probabilities: List[float],
                                 predictions: List[int], output_file: str):
        """保存序列和得分到FASTA文件"""
        logger.info(f"💾 保存序列到 {output_file}...")
        
        records = []
        for i, (seq, prob, pred) in enumerate(zip(sequences, probabilities, predictions), 1):
            # 创建描述信息
            label = "Anti-Gram-Negative" if pred == 1 else "Non-Anti-Gram-Negative"
            description = f"Generated_Seq_{i:05d} | Score: {prob:.4f} | Prediction: {label} | Length: {len(seq)}"
            
            # 创建SeqRecord
            record = SeqRecord(
                Seq(seq),
                id=f"Generated_Seq_{i:05d}",
                description=description
            )
            records.append(record)
        
        # 写入FASTA文件
        with open(output_file, 'w') as f:
            SeqIO.write(records, f, "fasta")
        
        logger.info(f"🎉 完成！{len(records)}条序列已保存到 {output_file}")
    
    def print_statistics(self, sequences: List[str], probabilities: List[float], predictions: List[int]):
        """打印统计信息"""
        lengths = [len(seq) for seq in sequences]
        
        logger.info("📊 生成序列统计信息:")
        logger.info(f"   序列数量: {len(sequences)}")
        logger.info(f"   平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        logger.info(f"   长度范围: {min(lengths)} - {max(lengths)}")
        logger.info(f"   平均预测得分: {np.mean(probabilities):.3f} ± {np.std(probabilities):.3f}")
        logger.info(f"   得分范围: {min(probabilities):.3f} - {max(probabilities):.3f}")
        logger.info(f"   预测为抗革兰氏阴性菌肽: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        logger.info(f"   预测为非抗革兰氏阴性菌肽: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.1f}%)")
        logger.info(f"   高分序列 (>0.8): {sum(1 for p in probabilities if p > 0.8)}")
        logger.info(f"   中分序列 (0.5-0.8): {sum(1 for p in probabilities if 0.5 <= p <= 0.8)}")
        logger.info(f"   低分序列 (<0.5): {sum(1 for p in probabilities if p < 0.5)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用生成器和hybrid_predict.py生成10000条序列并计算真实预测得分")
    parser.add_argument("--generator_checkpoint", type=str, required=True,
                       help="生成器检查点路径 (例如: checkpoints_hybrid_650M/checkpoint_epoch_200.pt)")
    parser.add_argument("--hybrid_predict_script", type=str, default="hybrid_predict.py",
                       help="hybrid_predict.py脚本路径")
    parser.add_argument("--model_weights", type=str, default="model/best_weights.h5",
                       help="分类器权重路径")
    parser.add_argument("--scaler_path", type=str, default="model/scaler.pkl",
                       help="标准化器路径")
    parser.add_argument("--feature_names_file", type=str, default="data/feature_names.txt",
                       help="特征名称文件路径")
    parser.add_argument("--config", type=str, default="dual_4090",
                       help="配置名称")
    parser.add_argument("--output", type=str, default="generated_10k_with_hybrid_scores.fasta",
                       help="输出FASTA文件路径")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="批次大小")
    
    args = parser.parse_args()
    
    try:
        # 初始化混合生成器
        generator = HybridSequenceGenerator(
            generator_checkpoint=args.generator_checkpoint,
            hybrid_predict_script=args.hybrid_predict_script,
            model_weights=args.model_weights,
            scaler_path=args.scaler_path,
            feature_names_file=args.feature_names_file,
            config_name=args.config
        )
        
        # 检查依赖文件
        generator.check_classifier_dependencies()
        
        # 加载生成器
        generator.load_generator()
        
        # 生成10000条序列
        generator.generate_10k_sequences_with_scores(
            output_file=args.output,
            batch_size=args.batch_size
        )
        
        logger.info("🎊 任务完成！")
        
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()