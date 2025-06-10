#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成高质量抗菌肽序列脚本

功能:
1. 使用多样采样生成序列
2. 用预测模型筛选概率>0.95的序列
3. 与现有数据库去重
4. 输出高质量的FASTA文件

使用方法:
python generate_high_quality_sequences.py --num_batches 10 --batch_size 50 --output generated_sequences.fasta

作者: AI Assistant
日期: 2025-06-10
"""

import os
import sys
import torch
import numpy as np
import pickle
import logging
import argparse
from datetime import datetime
from typing import List, Set, Tuple
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from peptides import Peptide

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENHANCED_ARCH_PATH = os.path.join(PROJECT_ROOT, 'enhanced_architecture')
sys.path.append(PROJECT_ROOT)
sys.path.append(ENHANCED_ARCH_PATH)

# 导入项目模块
try:
    from enhanced_architecture.config.model_config import get_config
    from enhanced_architecture.esm2_auxiliary_encoder import ESM2AuxiliaryEncoder
    from enhanced_architecture.diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet
    from enhanced_architecture.data_loader import tokens_to_sequence
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'generation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 预测相关常量 (从hybrid_predict.py复制)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB_DICT = {aa: i+2 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_DICT['<PAD>'] = 0
VOCAB_DICT['<UNK>'] = 1
VOCAB_SIZE = len(VOCAB_DICT)
MAX_SEQUENCE_LENGTH = 32
PAD_TOKEN_ID = VOCAB_DICT['<PAD>']
UNK_TOKEN_ID = VOCAB_DICT['<UNK>']

class SequenceGenerator:
    """序列生成器"""
    
    def __init__(self, checkpoint_path: str, config_name: str = "dual_4090"):
        """初始化生成器"""
        self.config = get_config(config_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.load_models(checkpoint_path)
        logger.info(f"✅ 序列生成器初始化完成，设备: {self.device}")
    
    def load_models(self, checkpoint_path: str):
        """加载训练好的模型"""
        logger.info(f"📂 加载检查点: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 初始化ESM-2编码器
        self.esm2_encoder = ESM2AuxiliaryEncoder(self.config.esm2)
        self.esm2_encoder.to(self.device)
        
        # 初始化扩散模型
        self.unet = D3PMUNet(
            vocab_size=self.config.diffusion.vocab_size,
            hidden_size=self.config.diffusion.hidden_size,
            n_layers=self.config.diffusion.n_layers,
            n_heads=self.config.diffusion.n_heads,
            max_seq_len=self.config.diffusion.max_seq_len,
            time_embed_dim=self.config.diffusion.time_embed_dim,
            esm_feature_dim=self.config.esm2.feature_dim
        )
        
        self.scheduler = D3PMScheduler(
            num_classes=self.config.diffusion.vocab_size,
            num_timesteps=self.config.diffusion.num_timesteps
        )
        
        self.diffusion_model = D3PMDiffusion(
            unet=self.unet,
            scheduler=self.scheduler,
            vocab_size=self.config.diffusion.vocab_size
        )
        
        # 加载权重
        self.esm2_encoder.load_state_dict(checkpoint['esm2_encoder_state_dict'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        
        # 设置为评估模式
        self.esm2_encoder.eval()
        self.diffusion_model.eval()
        
        logger.info("✅ 模型加载完成")
    
    def generate_sequences(self, num_sequences: int, seq_length: int = 40, 
                          temperature: float = 1.0, diversity_strength: float = 1.5) -> List[str]:
        """生成序列"""
        logger.info(f"🧬 开始生成 {num_sequences} 个序列，长度: {seq_length}")
        
        with torch.no_grad():
            # 准备ESM特征（这里使用零向量，实际可以用参考序列）
            esm_features = torch.zeros(
                num_sequences, 
                self.config.esm2.feature_dim, 
                device=self.device
            )
            
            # 使用多样采样
            generated_tokens = self.diffusion_model.diverse_sample(
                batch_size=num_sequences,
                seq_len=seq_length,
                esm_features=esm_features,
                diversity_strength=diversity_strength,
                temperature=temperature
            )
        
        # 转换为氨基酸序列
        sequences = []
        for i, tokens in enumerate(generated_tokens):
            seq = tokens_to_sequence(tokens.cpu().numpy())
            if seq and len(seq) > 5:  # 过滤太短的序列
                sequences.append(seq)
        
        logger.info(f"✅ 成功生成 {len(sequences)} 个有效序列")
        return sequences

class SequencePredictor:
    """序列预测器"""
    
    def __init__(self, model_path: str, scaler_path: str):
        """初始化预测器"""
        logger.info("📊 初始化预测器...")
        
        # 加载模型
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ 加载预测模型: {model_path}")
        
        # 加载特征标准化器
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"✅ 加载标准化器: {scaler_path}")
    
    def predict_sequences(self, sequences: List[str], threshold: float = 0.95) -> List[Tuple[str, float]]:
        """预测序列并筛选高质量序列"""
        logger.info(f"🔍 开始预测 {len(sequences)} 个序列，阈值: {threshold}")
        
        if not sequences:
            return []
        
        # 提取特征
        features_data = []
        valid_sequences = []
        
        for seq in sequences:
            try:
                # 序列编码
                encoded_seq = [VOCAB_DICT.get(aa, UNK_TOKEN_ID) for aa in seq]
                padded_seq = pad_sequences([encoded_seq], maxlen=MAX_SEQUENCE_LENGTH, 
                                         padding='post', value=PAD_TOKEN_ID)[0]
                
                # 全局特征计算
                peptide = Peptide(seq)
                global_features = [
                    len(seq),
                    peptide.molecular_weight(),
                    peptide.aromaticity(),
                    peptide.instability_index(),
                    peptide.aliphatic_index(),
                    peptide.boman_index(),
                    peptide.hydrophobic_ratio(),
                    peptide.charge(pH=7.0),
                    peptide.charge_density(pH=7.0),
                    peptide.isoelectric_point()
                ]
                
                # 检查特征有效性
                if any(np.isnan(global_features) or np.isinf(global_features)):
                    logger.debug(f"跳过无效特征序列: {seq[:20]}...")
                    continue
                
                features_data.append({
                    'sequence': seq,
                    'encoded_seq': padded_seq,
                    'global_features': global_features
                })
                valid_sequences.append(seq)
                
            except Exception as e:
                logger.debug(f"特征提取失败 {seq[:20]}...: {e}")
                continue
        
        if not features_data:
            logger.warning("没有有效的序列用于预测")
            return []
        
        # 准备预测数据
        X_seq = np.array([item['encoded_seq'] for item in features_data])
        X_global = np.array([item['global_features'] for item in features_data])
        
        # 标准化全局特征
        X_global_scaled = self.scaler.transform(X_global)
        
        # 预测
        predictions = self.model.predict([X_seq, X_global_scaled], verbose=0)
        
        # 筛选高质量序列
        high_quality_sequences = []
        for i, prob in enumerate(predictions.flatten()):
            if prob >= threshold:
                high_quality_sequences.append((valid_sequences[i], float(prob)))
        
        logger.info(f"✅ 筛选出 {len(high_quality_sequences)} 个高质量序列 (≥{threshold})")
        return high_quality_sequences

class SequenceDeduplicator:
    """序列去重器"""
    
    def __init__(self, database_fasta_path: str):
        """初始化去重器"""
        logger.info("🔄 初始化去重器...")
        self.existing_sequences = self.load_existing_sequences(database_fasta_path)
        logger.info(f"✅ 加载了 {len(self.existing_sequences)} 个现有序列")
    
    def load_existing_sequences(self, fasta_path: str) -> Set[str]:
        """加载现有数据库中的序列"""
        sequences = set()
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                seq = str(record.seq).upper()
                sequences.add(seq)
        except FileNotFoundError:
            logger.warning(f"数据库文件未找到: {fasta_path}")
        except Exception as e:
            logger.error(f"加载数据库序列失败: {e}")
        return sequences
    
    def deduplicate(self, sequences: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """去重序列"""
        logger.info(f"🔄 开始去重 {len(sequences)} 个序列...")
        
        unique_sequences = []
        seen_sequences = set()
        
        for seq, prob in sequences:
            seq_upper = seq.upper()
            
            # 检查是否与生成的序列重复
            if seq_upper in seen_sequences:
                logger.debug(f"重复的生成序列: {seq[:20]}...")
                continue
            
            # 检查是否与数据库序列重复
            if seq_upper in self.existing_sequences:
                logger.debug(f"与数据库重复: {seq[:20]}...")
                continue
            
            unique_sequences.append((seq, prob))
            seen_sequences.add(seq_upper)
        
        logger.info(f"✅ 去重完成，保留 {len(unique_sequences)} 个唯一序列")
        return unique_sequences

def save_sequences_to_fasta(sequences: List[Tuple[str, float]], output_path: str):
    """保存序列到FASTA文件"""
    logger.info(f"💾 保存 {len(sequences)} 个序列到: {output_path}")
    
    records = []
    for i, (seq, prob) in enumerate(sequences, 1):
        record_id = f"Generated_AMP_{i:06d}"
        description = f"Probability: {prob:.4f} | Length: {len(seq)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        record = SeqRecord(Seq(seq), id=record_id, description=description)
        records.append(record)
    
    SeqIO.write(records, output_path, "fasta")
    logger.info(f"✅ 序列已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成高质量抗菌肽序列")
    parser.add_argument("--num_batches", type=int, default=10, help="生成批次数")
    parser.add_argument("--batch_size", type=int, default=50, help="每批次序列数量")
    parser.add_argument("--seq_length", type=int, default=40, help="序列长度")
    parser.add_argument("--threshold", type=float, default=0.95, help="预测概率阈值")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--diversity_strength", type=float, default=1.5, help="多样性强度")
    parser.add_argument("--output", type=str, default="generated_high_quality_sequences.fasta", help="输出FASTA文件")
    parser.add_argument("--checkpoint", type=str, default="enhanced_architecture/output/checkpoints/best.pt", help="生成模型检查点")
    parser.add_argument("--predict_model", type=str, default="model/hybrid_classifier_best_tuned.keras", help="预测模型路径")
    parser.add_argument("--scaler", type=str, default="model/hybrid_model_scaler.pkl", help="标准化器路径")
    parser.add_argument("--database", type=str, default="data/Gram+-.fasta", help="现有数据库FASTA文件")
    
    args = parser.parse_args()
    
    logger.info("🚀 开始生成高质量抗菌肽序列")
    logger.info(f"📊 参数: 批次数={args.num_batches}, 批次大小={args.batch_size}, 阈值={args.threshold}")
    
    try:
        # 初始化组件
        generator = SequenceGenerator(args.checkpoint)
        predictor = SequencePredictor(args.predict_model, args.scaler)
        deduplicator = SequenceDeduplicator(args.database)
        
        all_high_quality_sequences = []
        
        # 分批生成和筛选
        for batch_idx in range(args.num_batches):
            logger.info(f"📦 处理批次 {batch_idx + 1}/{args.num_batches}")
            
            # 生成序列
            generated_sequences = generator.generate_sequences(
                num_sequences=args.batch_size,
                seq_length=args.seq_length,
                temperature=args.temperature,
                diversity_strength=args.diversity_strength
            )
            
            if not generated_sequences:
                logger.warning(f"批次 {batch_idx + 1} 没有生成有效序列")
                continue
            
            # 预测筛选
            high_quality_batch = predictor.predict_sequences(
                generated_sequences, 
                threshold=args.threshold
            )
            
            if high_quality_batch:
                all_high_quality_sequences.extend(high_quality_batch)
                logger.info(f"✅ 批次 {batch_idx + 1} 筛选出 {len(high_quality_batch)} 个高质量序列")
            else:
                logger.info(f"❌ 批次 {batch_idx + 1} 没有符合阈值的序列")
        
        if not all_high_quality_sequences:
            logger.error("❌ 没有生成任何高质量序列，请调整参数")
            return
        
        logger.info(f"🎯 总共筛选出 {len(all_high_quality_sequences)} 个高质量序列")
        
        # 去重
        unique_sequences = deduplicator.deduplicate(all_high_quality_sequences)
        
        if not unique_sequences:
            logger.error("❌ 去重后没有剩余序列")
            return
        
        # 按概率排序
        unique_sequences.sort(key=lambda x: x[1], reverse=True)
        
        # 保存结果
        save_sequences_to_fasta(unique_sequences, args.output)
        
        # 统计信息
        probabilities = [prob for _, prob in unique_sequences]
        logger.info("📈 最终统计:")
        logger.info(f"  - 唯一序列数: {len(unique_sequences)}")
        logger.info(f"  - 平均概率: {np.mean(probabilities):.4f}")
        logger.info(f"  - 最高概率: {np.max(probabilities):.4f}")
        logger.info(f"  - 最低概率: {np.min(probabilities):.4f}")
        logger.info(f"  - 输出文件: {args.output}")
        
        logger.info("🎉 生成完成！")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)