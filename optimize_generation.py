#!/usr/bin/env python3
"""
生成器参数优化脚本
用于论文实验：找到最优的温度和多样性参数，生成高质量的抗菌肽序列
"""

import os
import sys
import torch
import json
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('/Users/apple/AIBD/Gram-_Database')
sys.path.append('/Users/apple/AIBD/Gram-_Database/gram_predictor')

# 导入生成器和预测器
from gram_predictor.generation_service import SequenceGenerationService
from gram_predictor.app import tokenize_and_pad_sequences_app, VOCAB_DICT

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SequenceOptimizer:
    def __init__(self, device='cpu'):
        """初始化优化器"""
        # 由于生成服务内部硬编码了设备检测逻辑，暂时使用CPU确保稳定性
        self.device = 'cpu'
        logger.info(f"使用设备: {self.device}（生成服务内部设备检测优先）")
        
        # 初始化生成服务
        try:
            self.gen_service = SequenceGenerationService()
            logger.info("✅ 生成服务初始化成功")
        except Exception as e:
            logger.error(f"❌ 生成服务初始化失败: {e}")
            raise
        
        # 预测器词汇表（22词汇）
        self.predictor_vocab = VOCAB_DICT
        self.vocab_size_predictor = len(self.predictor_vocab)
        
        # 参数搜索范围
        self.temperature_range = [0.7, 0.9, 1.0, 1.2, 1.5]
        self.diversity_range = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        
    def sequence_to_confidence(self, sequences: List[str]) -> List[float]:
        """
        使用预测器计算序列的置信度
        这里简化实现，主要关注序列的统计特征作为质量指标
        """
        confidences = []
        
        for seq in sequences:
            try:
                # 基础质量评估
                quality_score = self._calculate_sequence_quality(seq)
                confidences.append(quality_score)
            except Exception as e:
                logger.warning(f"序列质量评估失败: {e}")
                confidences.append(0.0)
        
        return confidences
    
    def _calculate_sequence_quality(self, sequence: str) -> float:
        """
        计算序列质量分数（作为置信度的代理指标）
        结合多个生物学相关的特征
        """
        if not sequence or len(sequence) < 5:
            return 0.0
        
        # 1. 长度合理性 (最优长度20-50)
        length_score = 1.0
        if len(sequence) < 10:
            length_score = len(sequence) / 10.0
        elif len(sequence) > 80:
            length_score = max(0.1, 1.0 - (len(sequence) - 80) / 100.0)
        
        # 2. 氨基酸多样性
        unique_aa = len(set(sequence))
        diversity_score = min(1.0, unique_aa / 12.0)  # 期望至少12种不同氨基酸
        
        # 3. 疏水性氨基酸比例 (抗菌肽通常需要一定的疏水性)
        hydrophobic_aa = set('AILMFWYV')
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
        hydrophobic_ratio = hydrophobic_count / len(sequence)
        hydrophobic_score = 1.0 - abs(hydrophobic_ratio - 0.4)  # 期望40%左右疏水性氨基酸
        
        # 4. 带电氨基酸比例 (抗菌肽通常带正电)
        positive_aa = set('KRH')
        negative_aa = set('DE')
        positive_count = sum(1 for aa in sequence if aa in positive_aa)
        negative_count = sum(1 for aa in sequence if aa in negative_aa)
        
        charge_ratio = (positive_count - negative_count) / len(sequence)
        charge_score = max(0.0, min(1.0, charge_ratio * 2))  # 期望净正电荷
        
        # 5. 避免过长的重复序列
        repetition_penalty = 1.0
        for i in range(len(sequence) - 2):
            if i < len(sequence) - 5:
                triplet = sequence[i:i+3]
                if sequence.count(triplet) > 2:
                    repetition_penalty *= 0.8
        
        # 综合评分
        quality_score = (
            length_score * 0.2 +
            diversity_score * 0.25 +
            hydrophobic_score * 0.25 +
            charge_score * 0.25 +
            repetition_penalty * 0.05
        )
        
        return max(0.0, min(1.0, quality_score))
    
    
    def test_parameter_combination(self, temperature: float, diversity_strength: float, 
                                 num_sequences: int = 10) -> Tuple[float, List[str]]:
        """测试特定参数组合"""
        logger.info(f"测试参数: temperature={temperature}, diversity_strength={diversity_strength}")
        
        try:
            # 生成序列
            result = self.gen_service.generate_sequences(
                num_sequences=num_sequences,
                seq_length=40,  # 标准长度
                sampling_method="diverse",
                temperature=temperature,
                diversity_strength=diversity_strength,
                reference_sequences=None
            )
            
            
            if not result.get('success', False):
                logger.error(f"生成失败: {result.get('message', 'Unknown error')}")
                return 0.0, []
            
            sequences = [seq_data['sequence'] for seq_data in result['sequences']]
            
            # 计算置信度
            confidences = self.sequence_to_confidence(sequences)
            avg_confidence = np.mean(confidences)
            
            logger.info(f"平均质量分数: {avg_confidence:.4f}")
            
            return avg_confidence, sequences
            
        except Exception as e:
            logger.error(f"参数测试失败: {e}")
            return 0.0, []
    
    def optimize_parameters(self) -> Tuple[float, float, float]:
        """优化参数，返回最佳的温度和多样性参数"""
        logger.info("开始参数优化...")
        
        best_score = 0.0
        best_params = (1.0, 0.3)  # 默认参数
        results = []
        
        total_combinations = len(self.temperature_range) * len(self.diversity_range)
        current_combination = 0
        
        for temperature in self.temperature_range:
            for diversity_strength in self.diversity_range:
                current_combination += 1
                logger.info(f"进度: {current_combination}/{total_combinations}")
                
                avg_confidence, sequences = self.test_parameter_combination(
                    temperature, diversity_strength
                )
                
                results.append({
                    'temperature': temperature,
                    'diversity_strength': diversity_strength,
                    'avg_confidence': avg_confidence,
                    'num_sequences': len(sequences)
                })
                
                if avg_confidence > best_score:
                    best_score = avg_confidence
                    best_params = (temperature, diversity_strength)
                    logger.info(f"🎯 发现更好的参数: T={temperature}, D={diversity_strength}, Score={avg_confidence:.4f}")
        
        logger.info(f"✅ 参数优化完成!")
        logger.info(f"最佳参数: temperature={best_params[0]}, diversity_strength={best_params[1]}")
        logger.info(f"最佳分数: {best_score:.4f}")
        
        return best_params[0], best_params[1], best_score
    
    def generate_final_sequences(self, temperature: float, diversity_strength: float, 
                               num_sequences: int = 20) -> List[Dict]:
        """使用最优参数生成最终的序列"""
        logger.info(f"使用最优参数生成 {num_sequences} 条序列...")
        
        result = self.gen_service.generate_sequences(
            num_sequences=num_sequences,
            seq_length=40,
            sampling_method="diverse",
            temperature=temperature,
            diversity_strength=diversity_strength,
            reference_sequences=None
        )
        
        if not result.get('success', False):
            raise Exception(f"最终生成失败: {result.get('message', 'Unknown error')}")
        
        sequences = result['sequences']
        
        # 为每个序列添加质量分数
        for seq_data in sequences:
            quality_score = self._calculate_sequence_quality(seq_data['sequence'])
            seq_data['quality_score'] = quality_score
        
        # 按质量分数排序
        sequences.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return sequences
    
    def save_results(self, sequences: List[Dict], best_params: Tuple[float, float, float]):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存序列数据
        output_file = f"amplify_synth_sequences_{timestamp}.json"
        results = {
            'generation_info': {
                'timestamp': timestamp,
                'num_sequences': len(sequences),
                'best_temperature': best_params[0],
                'best_diversity_strength': best_params[1],
                'best_score': best_params[2],
                'sampling_method': 'diverse',
                'sequence_length': 40
            },
            'sequences': sequences
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存FASTA格式
        fasta_file = f"amplify_synth_sequences_{timestamp}.fasta"
        with open(fasta_file, 'w') as f:
            for i, seq_data in enumerate(sequences, 1):
                f.write(f">AmplifysynthSeq_{i:02d}_Score_{seq_data['quality_score']:.3f}\n")
                f.write(f"{seq_data['sequence']}\n")
        
        logger.info(f"✅ 结果已保存:")
        logger.info(f"  - JSON格式: {output_file}")
        logger.info(f"  - FASTA格式: {fasta_file}")
        
        return output_file, fasta_file

def main():
    """主函数"""
    logger.info("🚀 开始Amplify-Synth参数优化和序列生成")
    
    try:
        # 初始化优化器
        optimizer = SequenceOptimizer(device='mps')
        
        # 参数优化
        best_temp, best_diversity, best_score = optimizer.optimize_parameters()
        
        # 生成最终序列
        final_sequences = optimizer.generate_final_sequences(
            temperature=best_temp,
            diversity_strength=best_diversity,
            num_sequences=20
        )
        
        # 保存结果
        json_file, fasta_file = optimizer.save_results(
            final_sequences, 
            (best_temp, best_diversity, best_score)
        )
        
        # 输出统计信息
        logger.info("📊 生成统计:")
        logger.info(f"  - 序列数量: {len(final_sequences)}")
        logger.info(f"  - 平均长度: {np.mean([len(seq['sequence']) for seq in final_sequences]):.1f}")
        logger.info(f"  - 最高质量分数: {final_sequences[0]['quality_score']:.4f}")
        logger.info(f"  - 最低质量分数: {final_sequences[-1]['quality_score']:.4f}")
        logger.info(f"  - 平均质量分数: {np.mean([seq['quality_score'] for seq in final_sequences]):.4f}")
        
        print(f"\n🎉 完成! 结果文件:")
        print(f"  - {json_file}")
        print(f"  - {fasta_file}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()