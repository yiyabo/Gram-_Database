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
        
        # APD数据库天然抗菌肽参考序列（革兰氏阴性菌活性）
        self.reference_sequences = {
            # 短肽类（15-20aa）
            "AP00007": "GNNRPVYIPQPRPPHPRL",                    # 18aa
            "AP00168": "GRPNPVNNKPTPHPRL",                      # 16aa  
            "AP00169": "GRPNPVNTKPTPYPRL",                      # 16aa
            "AP00142": "GLKKLLGKLLKKLGKLLLK",                   # 19aa
            
            # 中等长度（25-35aa）
            "AP00051": "GIGSAILSAGKSALKGLAKGLAEHFAN",           # 26aa
            "AP00126": "GGLKKLGKKLEGVGKRVFKASEKALPVAVGIKALG",   # 33aa
            "AP00129": "GWLKKIGKKIERVGQNTRDATVKGLEVAQQAANVAATVR", # 36aa
            
            # 富含脯氨酸的长肽（40+aa）
            "AP00009": "RFRPPIRRPPIRPPFYPPFRPPIRPPIFPPIRPPFRPPLGPFP",        # 43aa
            "AP00010": "RRIRPRPPRLPRPRPRPLPFPRPGPRPIPRPLPFPRPGPRPIPRPLPFPRPGPRPIPRPL", # 59aa
            
            # 富含半胱氨酸的结构肽（35-40aa）  
            "AP00036": "DFASCHTNGGICLPNRCPGHMIQIGICFRPRVKCCRSW",  # 38aa
            "AP00040": "QVVRNPQSCRWNMGVCIPISCPGNMRQIGTCFGPRVPCCRRW", # 39aa
        }
        
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
        
        # 专注于diverse采样方法
        self.sampling_configs = [
            # Diverse Sampling - 多样性采样（唯一测试方法）
            {
                'method': 'diverse',
                'param_name': 'diversity_strength',
                'param_range': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
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
    
    def test_conditional_generation(self, temperature: float, sampling_method: str,
                                  param_name: str = None, param_value: float = None,
                                  reference_sequences: List[str] = None,
                                  num_sequences: int = 12) -> Tuple[float, List[str], Dict]:
        """测试条件生成参数组合"""
        params_info = f"temperature={temperature}, method={sampling_method}"
        if param_name and param_value is not None:
            params_info += f", {param_name}={param_value}"
        
        logger.info(f"测试条件生成参数: {params_info}")
        logger.info(f"参考序列数量: {len(reference_sequences) if reference_sequences else 0}")
        
        try:
            # 构建参数字典
            kwargs = {'temperature': temperature}
            if param_name and param_value is not None:
                kwargs[param_name] = param_value
            
            # 条件生成序列
            result = self.gen_service.generate_sequences(
                num_sequences=num_sequences,
                seq_length=40,
                sampling_method=sampling_method,
                reference_sequences=reference_sequences,  # 传入参考序列
                **kwargs
            )
            
            if not result.get('success', False):
                logger.error(f"条件生成失败: {result.get('message', 'Unknown error')}")
                return 0.0, [], {}
            
            sequences = [seq_data['sequence'] for seq_data in result['sequences']]
            
            # 计算质量分数
            confidences = self.sequence_to_confidence(sequences)
            avg_confidence = np.mean(confidences)
            
            # 计算与参考序列的相似性
            ref_similarities = []
            if reference_sequences:
                for seq in sequences:
                    max_sim = max(self._calculate_similarity(seq, ref_seq) for ref_seq in reference_sequences)
                    ref_similarities.append(max_sim)
            
            # 计算额外统计信息
            stats = {
                'avg_length': np.mean([len(seq) for seq in sequences]),
                'std_length': np.std([len(seq) for seq in sequences]),
                'unique_sequences': len(set(sequences)),
                'avg_confidence': avg_confidence,
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'avg_ref_similarity': np.mean(ref_similarities) if ref_similarities else 0.0,
                'std_ref_similarity': np.std(ref_similarities) if ref_similarities else 0.0
            }
            
            logger.info(f"质量分数: {avg_confidence:.4f} (±{stats['std_confidence']:.4f})")
            logger.info(f"序列长度: {stats['avg_length']:.1f} (±{stats['std_length']:.1f})")
            logger.info(f"唯一序列: {stats['unique_sequences']}/{num_sequences}")
            if ref_similarities:
                logger.info(f"参考相似性: {stats['avg_ref_similarity']:.3f} (±{stats['std_ref_similarity']:.3f})")
            
            return avg_confidence, sequences, stats
            
        except Exception as e:
            logger.error(f"条件生成测试失败: {e}")
            return 0.0, [], {}
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """计算两个序列的相似性（简单的氨基酸匹配率）"""
        if not seq1 or not seq2:
            return 0.0
        
        # 简单的局部对齐相似性计算
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        # 计算最佳局部匹配
        best_match = 0
        for i in range(len(seq1) - min_len + 1):
            for j in range(len(seq2) - min_len + 1):
                matches = sum(1 for k in range(min_len) 
                            if i+k < len(seq1) and j+k < len(seq2) and seq1[i+k] == seq2[j+k])
                best_match = max(best_match, matches)
        
        return best_match / max_len if max_len > 0 else 0.0
    
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
    
    def optimize_with_references(self, selected_refs: List[str] = None) -> Tuple[Dict, List[Dict]]:
        """使用参考序列进行条件生成优化"""
        if selected_refs is None:
            # 你可以选择任意组合，这里提供几个选项：
            
            # 选择8条有代表性的参考序列（平衡覆盖面和性能）
            selected_refs = [
                # 短肽类（2条）
                "AP00007",   # 富含脯氨酸
                "AP00142",   # 富含赖氨酸和亮氨酸
                
                # 中等长度类（3条）
                "AP00051",   # α-螺旋结构
                "AP00126",   # 富含赖氨酸
                "AP00129",   # 长α-螺旋
                
                # 结构复杂类（2条）
                "AP00036",   # 富含半胱氨酸
                "AP00040",   # 另一个半胱氨酸富集
                
                # 富含脯氨酸长肽（1条）
                "AP00009"    # 43aa，脯氨酸重复序列
            ]
            
            # 其他选项（已注释）：
            # 选项1: 代表性4条
            # selected_refs = ["AP00007", "AP00051", "AP00126", "AP00036"]
            
            # 选项2: 使用全部11条（最全面但计算量大）
            # selected_refs = list(self.reference_sequences.keys())
        
        logger.info("🧬 开始参考序列条件生成优化...")
        logger.info(f"📋 使用参考序列: {', '.join(selected_refs)}")
        
        # 获取参考序列
        ref_sequences = [self.reference_sequences[name] for name in selected_refs if name in self.reference_sequences]
        if not ref_sequences:
            raise ValueError("未找到有效的参考序列")
        
        logger.info(f"🎯 参考序列详情:")
        for name, seq in zip(selected_refs, ref_sequences):
            if name in self.reference_sequences:
                logger.info(f"  {name}: {seq} (长度: {len(seq)})")
        
        all_results = []
        best_score = 0.0
        best_config = None
        
        # 专注于diverse采样的精细化搜索
        focused_configs = [
            {
                'method': 'diverse',
                'param_name': 'diversity_strength',
                'param_range': [0.5, 0.7, 0.9, 1.1]  # 重点测试高多样性区间
            }
        ]
        
        focused_temperature_range = [0.8, 1.0, 1.1, 1.2, 1.3]  # 重点测试中等温度
        
        # 计算总组合数
        total_combinations = 0
        for config in focused_configs:
            total_combinations += len(focused_temperature_range) * len(config['param_range'])
        
        current_combination = 0
        
        # 遍历采样方法
        for config in focused_configs:
            method = config['method']
            param_name = config['param_name']
            param_range = config['param_range']
            
            logger.info(f"\n🔬 测试条件生成方法: {method.upper()}")
            logger.info("=" * 50)
            
            # 遍历参数组合
            for temperature in focused_temperature_range:
                for param_value in param_range:
                    current_combination += 1
                    
                    logger.info(f"进度: {current_combination}/{total_combinations}")
                    
                    # 测试条件生成
                    avg_confidence, sequences, stats = self.test_conditional_generation(
                        temperature=temperature,
                        sampling_method=method,
                        param_name=param_name,
                        param_value=param_value,
                        reference_sequences=ref_sequences,
                        num_sequences=12  # 条件生成测试序列数
                    )
                    
                    # 记录结果
                    result = {
                        'method': method,
                        'temperature': temperature,
                        'param_name': param_name,
                        'param_value': param_value,
                        'avg_confidence': avg_confidence,
                        'stats': stats,
                        'num_sequences': len(sequences),
                        'reference_sequences': selected_refs,
                        'generation_type': 'conditional'
                    }
                    
                    all_results.append(result)
                    
                    # 检查是否是最佳结果
                    if avg_confidence > best_score:
                        best_score = avg_confidence
                        best_config = result.copy()
                        
                        params_str = f"T={temperature}"
                        if param_name and param_value is not None:
                            params_str += f", {param_name}={param_value}"
                        
                        logger.info(f"🎯 发现更好的条件生成参数: {method.upper()} - {params_str}, Score={avg_confidence:.4f}")
        
        # 按分数排序
        all_results.sort(key=lambda x: x['avg_confidence'], reverse=True)
        
        logger.info(f"\n✅ 条件生成优化完成!")
        logger.info(f"🏆 最佳条件生成配置: {best_config['method'].upper()}")
        logger.info(f"📊 最佳分数: {best_score:.4f}")
        
        # 显示Top 5条件生成结果
        logger.info(f"\n🏅 Top 5 条件生成配置:")
        for i, result in enumerate(all_results[:5], 1):
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
    import sys
    
    # 检查命令行参数选择模式
    mode = "unconditional"  # 默认无条件生成
    if len(sys.argv) > 1:
        if sys.argv[1] == "conditional":
            mode = "conditional"
        elif sys.argv[1] == "both":
            mode = "both"
    
    logger.info("🚀 开始Amplify-Synth参数优化和序列生成")
    logger.info(f"🎯 运行模式: {mode.upper()}")
    
    if mode == "unconditional":
        logger.info("📋 运行无条件生成优化...")
    elif mode == "conditional":
        logger.info("📋 运行条件生成优化...")
    else:
        logger.info("📋 运行完整对比实验（无条件 + 条件生成）...")
    
    try:
        # 初始化优化器
        optimizer = SequenceOptimizer(device='cpu')
        
        # 显示可用参考序列
        if mode in ["conditional", "both"]:
            logger.info(f"\n📖 可用参考序列:")
            for name, seq in optimizer.reference_sequences.items():
                logger.info(f"  {name:15s}: {seq[:30]}{'...' if len(seq) > 30 else ''} (长度: {len(seq)})")
        
        all_results = []
        
        # 无条件生成优化
        if mode in ["unconditional", "both"]:
            logger.info(f"\n" + "="*60)
            logger.info("🔬 第一阶段: 无条件生成优化")
            logger.info("="*60)
            
            # 显示搜索空间信息
            total_combinations = 0
            for config in optimizer.sampling_configs:
                combinations = len(optimizer.temperature_range) * len(config['param_range'])
                total_combinations += combinations
                logger.info(f"📋 {config['method'].upper()}: {len(optimizer.temperature_range)} 温度 × {len(config['param_range'])} 参数 = {combinations} 组合")
            
            logger.info(f"🎯 无条件生成总计测试组合: {total_combinations}")
            logger.info(f"⏱️  预估时间: {total_combinations * 0.5:.1f} 分钟")
            
            # 无条件参数优化
            best_unconditional, unconditional_results = optimizer.optimize_parameters()
            all_results.extend(unconditional_results)
            
            logger.info(f"\n✅ 无条件生成最佳配置:")
            params_str = f"T={best_unconditional['temperature']}"
            if best_unconditional['param_name'] and best_unconditional['param_value'] is not None:
                params_str += f", {best_unconditional['param_name']}={best_unconditional['param_value']}"
            logger.info(f"   方法: {best_unconditional['method'].upper()}")
            logger.info(f"   参数: {params_str}")
            logger.info(f"   分数: {best_unconditional['avg_confidence']:.4f}")
        
        # 条件生成优化
        best_conditional = None
        if mode in ["conditional", "both"]:
            logger.info(f"\n" + "="*60)
            logger.info("🧬 第二阶段: 条件生成优化")
            logger.info("="*60)
            
            # 条件生成优化
            best_conditional, conditional_results = optimizer.optimize_with_references()
            all_results.extend(conditional_results)
            
            logger.info(f"\n✅ 条件生成最佳配置:")
            params_str = f"T={best_conditional['temperature']}"
            if best_conditional['param_name'] and best_conditional['param_value'] is not None:
                params_str += f", {best_conditional['param_name']}={best_conditional['param_value']}"
            logger.info(f"   方法: {best_conditional['method'].upper()}")
            logger.info(f"   参数: {params_str}")
            logger.info(f"   分数: {best_conditional['avg_confidence']:.4f}")
            logger.info(f"   参考序列: {', '.join(best_conditional['reference_sequences'])}")
        
        # 选择最终的最佳配置
        if mode == "unconditional":
            final_best_config = best_unconditional
        elif mode == "conditional":
            final_best_config = best_conditional
        else:  # both
            # 比较两种模式的最佳结果
            if best_conditional['avg_confidence'] > best_unconditional['avg_confidence']:
                final_best_config = best_conditional
                logger.info(f"\n🏆 条件生成获胜! (分数: {best_conditional['avg_confidence']:.4f} vs {best_unconditional['avg_confidence']:.4f})")
            else:
                final_best_config = best_unconditional
                logger.info(f"\n🏆 无条件生成获胜! (分数: {best_unconditional['avg_confidence']:.4f} vs {best_conditional['avg_confidence']:.4f})")
        
        # 使用最佳配置生成最终序列
        logger.info(f"\n" + "="*60)
        logger.info("🎯 最终序列生成")
        logger.info("="*60)
        
        final_sequences = optimizer.generate_final_sequences(
            best_config=final_best_config,
            num_sequences=25
        )
        
        # 保存所有结果
        exp_file, seq_file, fasta_file, report_file = optimizer.save_results(
            final_sequences, final_best_config, all_results
        )
        
        # 输出统计信息
        logger.info("\n📊 最终序列统计:")
        logger.info(f"  - 生成模式: {final_best_config.get('generation_type', 'unconditional').upper()}")
        logger.info(f"  - 序列数量: {len(final_sequences)}")
        logger.info(f"  - 平均长度: {np.mean([len(seq['sequence']) for seq in final_sequences]):.1f}")
        logger.info(f"  - 最高质量分数: {final_sequences[0]['quality_score']:.4f}")
        logger.info(f"  - 最低质量分数: {final_sequences[-1]['quality_score']:.4f}")
        logger.info(f"  - 平均质量分数: {np.mean([seq['quality_score'] for seq in final_sequences]):.4f}")
        logger.info(f"  - 唯一序列: {len(set(seq['sequence'] for seq in final_sequences))}/{len(final_sequences)}")
        
        # 如果是条件生成，显示参考序列信息
        if final_best_config.get('generation_type') == 'conditional':
            logger.info(f"  - 参考序列: {', '.join(final_best_config['reference_sequences'])}")
        
        print(f"\n🎉 优化完成! 结果文件:")
        print(f"  📊 实验数据: {exp_file}")
        print(f"  🧬 序列数据: {seq_file}")
        print(f"  📄 FASTA文件: {fasta_file}")
        print(f"  📋 分析报告: {report_file}")
        print(f"\n💡 使用方法:")
        print(f"  无条件生成: python optimize_generation.py")
        print(f"  条件生成:   python optimize_generation.py conditional")
        print(f"  完整对比:   python optimize_generation.py both")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()