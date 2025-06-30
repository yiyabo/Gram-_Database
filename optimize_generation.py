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
        
        # 精细化参数搜索范围
        self.temperature_range = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]
        
        # 不同采样方法的参数组合
        self.sampling_configs = [
            # Diverse Sampling - 多样性采样
            {
                'method': 'diverse',
                'param_name': 'diversity_strength',
                'param_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            },
            # Top-K Sampling - Top-K采样
            {
                'method': 'top_k', 
                'param_name': 'k',
                'param_range': [5, 8, 10, 12, 15, 18, 20]
            },
            # Nucleus Sampling - 核采样
            {
                'method': 'nucleus',
                'param_name': 'p', 
                'param_range': [0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98]
            },
            # Basic Sampling - 基础采样（仅温度）
            {
                'method': 'basic',
                'param_name': None,
                'param_range': [None]  # 只测试温度
            }
        ]
        
        
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
    
    
    def test_parameter_combination(self, temperature: float, sampling_method: str,
                                 param_name: str = None, param_value: float = None,
                                 num_sequences: int = 10) -> Tuple[float, List[str], Dict]:
        """测试特定参数组合"""
        params_info = f"temperature={temperature}, method={sampling_method}"
        if param_name and param_value is not None:
            params_info += f", {param_name}={param_value}"
        
        logger.info(f"测试参数: {params_info}")
        
        try:
            # 构建参数字典
            kwargs = {'temperature': temperature}
            if param_name and param_value is not None:
                kwargs[param_name] = param_value
            
            # 生成序列
            result = self.gen_service.generate_sequences(
                num_sequences=num_sequences,
                seq_length=40,  # 标准长度
                sampling_method=sampling_method,
                reference_sequences=None,
                **kwargs
            )
            
            
            if not result.get('success', False):
                logger.error(f"生成失败: {result.get('message', 'Unknown error')}")
                return 0.0, [], {}
            
            sequences = [seq_data['sequence'] for seq_data in result['sequences']]
            
            # 计算置信度
            confidences = self.sequence_to_confidence(sequences)
            avg_confidence = np.mean(confidences)
            
            # 计算额外统计信息
            stats = {
                'avg_length': np.mean([len(seq) for seq in sequences]),
                'std_length': np.std([len(seq) for seq in sequences]),
                'unique_sequences': len(set(sequences)),
                'avg_confidence': avg_confidence,
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            }
            
            logger.info(f"质量分数: {avg_confidence:.4f} (±{stats['std_confidence']:.4f})")
            logger.info(f"序列长度: {stats['avg_length']:.1f} (±{stats['std_length']:.1f})")
            logger.info(f"唯一序列: {stats['unique_sequences']}/{num_sequences}")
            
            return avg_confidence, sequences, stats
            
        except Exception as e:
            logger.error(f"参数测试失败: {e}")
            return 0.0, [], {}
    
    def optimize_parameters(self) -> Tuple[Dict, List[Dict]]:
        """优化参数，返回最佳参数和完整结果"""
        logger.info("🚀 开始全面参数优化...")
        
        all_results = []
        best_score = 0.0
        best_config = None
        
        # 计算总组合数
        total_combinations = 0
        for config in self.sampling_configs:
            total_combinations += len(self.temperature_range) * len(config['param_range'])
        
        current_combination = 0
        
        # 遍历所有采样方法
        for config in self.sampling_configs:
            method = config['method']
            param_name = config['param_name']
            param_range = config['param_range']
            
            logger.info(f"\n🔬 测试采样方法: {method.upper()}")
            logger.info("=" * 50)
            
            # 遍历温度参数
            for temperature in self.temperature_range:
                # 遍历特定方法的参数
                for param_value in param_range:
                    current_combination += 1
                    
                    logger.info(f"进度: {current_combination}/{total_combinations}")
                    
                    # 测试参数组合
                    avg_confidence, sequences, stats = self.test_parameter_combination(
                        temperature=temperature,
                        sampling_method=method,
                        param_name=param_name,
                        param_value=param_value,
                        num_sequences=15  # 增加测试序列数量
                    )
                    
                    # 记录结果
                    result = {
                        'method': method,
                        'temperature': temperature,
                        'param_name': param_name,
                        'param_value': param_value,
                        'avg_confidence': avg_confidence,
                        'stats': stats,
                        'num_sequences': len(sequences)
                    }
                    
                    all_results.append(result)
                    
                    # 检查是否是最佳结果
                    if avg_confidence > best_score:
                        best_score = avg_confidence
                        best_config = result.copy()
                        
                        params_str = f"T={temperature}"
                        if param_name and param_value is not None:
                            params_str += f", {param_name}={param_value}"
                        
                        logger.info(f"🎯 发现更好的参数: {method.upper()} - {params_str}, Score={avg_confidence:.4f}")
        
        # 按分数排序，找出前几名
        all_results.sort(key=lambda x: x['avg_confidence'], reverse=True)
        top_results = all_results[:10]  # 前10名
        
        logger.info(f"\n✅ 参数优化完成!")
        logger.info(f"🏆 最佳配置: {best_config['method'].upper()}")
        logger.info(f"📊 最佳分数: {best_score:.4f}")
        
        # 显示Top 5结果
        logger.info(f"\n🏅 Top 5 配置:")
        for i, result in enumerate(top_results[:5], 1):
            params_str = f"T={result['temperature']}"
            if result['param_name'] and result['param_value'] is not None:
                params_str += f", {result['param_name']}={result['param_value']}"
            
            logger.info(f"  {i}. {result['method'].upper()}: {params_str} - Score: {result['avg_confidence']:.4f}")
        
        return best_config, all_results
    
    def generate_final_sequences(self, best_config: Dict, num_sequences: int = 25) -> List[Dict]:
        """使用最优参数生成最终的序列"""
        method = best_config['method']
        temperature = best_config['temperature']
        param_name = best_config['param_name']
        param_value = best_config['param_value']
        
        params_str = f"T={temperature}"
        if param_name and param_value is not None:
            params_str += f", {param_name}={param_value}"
        
        logger.info(f"🎯 使用最优配置生成 {num_sequences} 条序列:")
        logger.info(f"   方法: {method.upper()}")
        logger.info(f"   参数: {params_str}")
        
        # 构建参数
        kwargs = {'temperature': temperature}
        if param_name and param_value is not None:
            kwargs[param_name] = param_value
        
        result = self.gen_service.generate_sequences(
            num_sequences=num_sequences,
            seq_length=40,
            sampling_method=method,
            reference_sequences=None,
            **kwargs
        )
        
        if not result.get('success', False):
            raise Exception(f"最终生成失败: {result.get('message', 'Unknown error')}")
        
        sequences = result['sequences']
        
        # 为每个序列添加质量分数
        for seq_data in sequences:
            quality_score = self._calculate_sequence_quality(seq_data['sequence'])
            seq_data['quality_score'] = quality_score
            seq_data['generation_method'] = method
            seq_data['generation_params'] = kwargs
        
        # 按质量分数排序
        sequences.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return sequences
    
    def save_results(self, sequences: List[Dict], best_config: Dict, all_results: List[Dict]):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整实验数据
        experiment_file = f"amplify_synth_experiment_{timestamp}.json"
        experiment_data = {
            'experiment_info': {
                'timestamp': timestamp,
                'total_combinations_tested': len(all_results),
                'best_config': best_config,
                'search_space': {
                    'temperature_range': self.temperature_range,
                    'sampling_configs': self.sampling_configs
                }
            },
            'all_results': all_results,
            'top_10_results': sorted(all_results, key=lambda x: x['avg_confidence'], reverse=True)[:10]
        }
        
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        # 保存最终序列数据
        sequences_file = f"amplify_synth_sequences_{timestamp}.json"
        sequences_data = {
            'generation_info': {
                'timestamp': timestamp,
                'num_sequences': len(sequences),
                'best_config': best_config,
                'sequence_length': 40
            },
            'sequences': sequences
        }
        
        with open(sequences_file, 'w', encoding='utf-8') as f:
            json.dump(sequences_data, f, indent=2, ensure_ascii=False)
        
        # 保存FASTA格式
        fasta_file = f"amplify_synth_sequences_{timestamp}.fasta"
        method = best_config['method']
        with open(fasta_file, 'w') as f:
            for i, seq_data in enumerate(sequences, 1):
                header = f">AmplifySeq_{i:02d}_{method.upper()}_Score_{seq_data['quality_score']:.3f}"
                f.write(f"{header}\n")
                f.write(f"{seq_data['sequence']}\n")
        
        # 保存Top参数配置的简要报告
        report_file = f"amplify_synth_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🧬 Amplify-Synth 参数优化报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📅 实验时间: {timestamp}\n")
            f.write(f"🔬 测试组合数: {len(all_results)}\n\n")
            
            f.write("🏆 最佳配置:\n")
            params_str = f"T={best_config['temperature']}"
            if best_config['param_name'] and best_config['param_value'] is not None:
                params_str += f", {best_config['param_name']}={best_config['param_value']}"
            f.write(f"   方法: {best_config['method'].upper()}\n")
            f.write(f"   参数: {params_str}\n")
            f.write(f"   分数: {best_config['avg_confidence']:.4f}\n\n")
            
            f.write("🏅 Top 10 配置:\n")
            top_results = sorted(all_results, key=lambda x: x['avg_confidence'], reverse=True)[:10]
            for i, result in enumerate(top_results, 1):
                params_str = f"T={result['temperature']}"
                if result['param_name'] and result['param_value'] is not None:
                    params_str += f", {result['param_name']}={result['param_value']}"
                f.write(f"   {i:2d}. {result['method'].upper():8s}: {params_str:20s} - Score: {result['avg_confidence']:.4f}\n")
        
        logger.info(f"✅ 结果已保存:")
        logger.info(f"  - 实验数据: {experiment_file}")
        logger.info(f"  - 序列数据: {sequences_file}")
        logger.info(f"  - FASTA格式: {fasta_file}")
        logger.info(f"  - 分析报告: {report_file}")
        
        return experiment_file, sequences_file, fasta_file, report_file

def main():
    """主函数"""
    logger.info("🚀 开始Amplify-Synth全面参数优化和序列生成")
    
    try:
        # 初始化优化器
        optimizer = SequenceOptimizer(device='cpu')
        
        # 显示搜索空间信息
        total_combinations = 0
        for config in optimizer.sampling_configs:
            combinations = len(optimizer.temperature_range) * len(config['param_range'])
            total_combinations += combinations
            logger.info(f"📋 {config['method'].upper()}: {len(optimizer.temperature_range)} 温度 × {len(config['param_range'])} 参数 = {combinations} 组合")
        
        logger.info(f"🎯 总计测试组合: {total_combinations}")
        logger.info(f"⏱️  预估时间: {total_combinations * 0.5:.1f} 分钟 (假设每组合30秒)")
        
        # 全面参数优化
        best_config, all_results = optimizer.optimize_parameters()
        
        # 使用最佳配置生成最终序列
        final_sequences = optimizer.generate_final_sequences(
            best_config=best_config,
            num_sequences=25  # 增加到25条
        )
        
        # 保存所有结果
        exp_file, seq_file, fasta_file, report_file = optimizer.save_results(
            final_sequences, best_config, all_results
        )
        
        # 输出统计信息
        logger.info("\n📊 生成统计:")
        logger.info(f"  - 序列数量: {len(final_sequences)}")
        logger.info(f"  - 平均长度: {np.mean([len(seq['sequence']) for seq in final_sequences]):.1f}")
        logger.info(f"  - 最高质量分数: {final_sequences[0]['quality_score']:.4f}")
        logger.info(f"  - 最低质量分数: {final_sequences[-1]['quality_score']:.4f}")
        logger.info(f"  - 平均质量分数: {np.mean([seq['quality_score'] for seq in final_sequences]):.4f}")
        logger.info(f"  - 唯一序列: {len(set(seq['sequence'] for seq in final_sequences))}/{len(final_sequences)}")
        
        # 按采样方法统计Top结果
        logger.info("\n🏅 各采样方法最佳结果:")
        method_best = {}
        for result in all_results:
            method = result['method']
            if method not in method_best or result['avg_confidence'] > method_best[method]['avg_confidence']:
                method_best[method] = result
        
        for method, result in method_best.items():
            params_str = f"T={result['temperature']}"
            if result['param_name'] and result['param_value'] is not None:
                params_str += f", {result['param_name']}={result['param_value']}"
            logger.info(f"  {method.upper():8s}: {params_str:25s} Score: {result['avg_confidence']:.4f}")
        
        print(f"\n🎉 全面优化完成! 结果文件:")
        print(f"  📊 实验数据: {exp_file}")
        print(f"  🧬 序列数据: {seq_file}")
        print(f"  📄 FASTA文件: {fasta_file}")
        print(f"  📋 分析报告: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()