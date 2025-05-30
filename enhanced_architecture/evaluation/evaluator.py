"""
模型评估器：评估生成的抗菌肽质量和活性
包括序列质量、多样性、抗菌活性预测等指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging
from dataclasses import dataclass
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # 序列质量指标
    avg_sequence_length: float
    length_diversity: float
    amino_acid_diversity: float
    valid_sequences_ratio: float
    
    # 生物学特性指标
    avg_molecular_weight: float
    avg_isoelectric_point: float
    avg_hydrophobicity: float
    avg_charge_density: float
    
    # 多样性指标
    pairwise_similarity: float
    sequence_diversity_score: float
    novel_sequences_ratio: float
    
    # 抗菌活性相关指标
    predicted_activity_ratio: float
    avg_activity_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'avg_sequence_length': self.avg_sequence_length,
            'length_diversity': self.length_diversity,
            'amino_acid_diversity': self.amino_acid_diversity,
            'valid_sequences_ratio': self.valid_sequences_ratio,
            'avg_molecular_weight': self.avg_molecular_weight,
            'avg_isoelectric_point': self.avg_isoelectric_point,
            'avg_hydrophobicity': self.avg_hydrophobicity,
            'avg_charge_density': self.avg_charge_density,
            'pairwise_similarity': self.pairwise_similarity,
            'sequence_diversity_score': self.sequence_diversity_score,
            'novel_sequences_ratio': self.novel_sequences_ratio,
            'predicted_activity_ratio': self.predicted_activity_ratio,
            'avg_activity_score': self.avg_activity_score
        }

class SequenceAnalyzer:
    """序列分析工具"""
    
    def __init__(self):
        """初始化分析器"""
        # 氨基酸理化性质
        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        self.charge_scale = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
            'Q': 0, 'E': -1, 'G': 0, 'H': 0.5, 'I': 0,
            'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
            'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def is_valid_sequence(self, sequence: str) -> bool:
        """检查序列是否有效"""
        if not sequence or len(sequence) < 3:
            return False
        
        # 检查是否只包含标准氨基酸
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        return all(aa in valid_aa for aa in sequence.upper())
    
    def calculate_physicochemical_properties(self, sequence: str) -> Dict[str, float]:
        """计算序列的理化性质"""
        if not self.is_valid_sequence(sequence):
            return {
                'molecular_weight': 0.0,
                'isoelectric_point': 0.0,
                'hydrophobicity': 0.0,
                'charge_density': 0.0
            }
        
        try:
            # 使用BioPython计算分子量和等电点
            protein_analysis = ProteinAnalysis(sequence.upper())
            molecular_weight = protein_analysis.molecular_weight()
            isoelectric_point = protein_analysis.isoelectric_point()
            
            # 计算疏水性
            hydrophobicity = np.mean([
                self.hydrophobicity_scale.get(aa, 0) for aa in sequence.upper()
            ])
            
            # 计算电荷密度
            charge_density = sum([
                self.charge_scale.get(aa, 0) for aa in sequence.upper()
            ]) / len(sequence)
            
            return {
                'molecular_weight': molecular_weight,
                'isoelectric_point': isoelectric_point,
                'hydrophobicity': hydrophobicity,
                'charge_density': charge_density
            }
            
        except Exception as e:
            self.logger.warning(f"计算理化性质时出错: {e}")
            return {
                'molecular_weight': 0.0,
                'isoelectric_point': 0.0,
                'hydrophobicity': 0.0,
                'charge_density': 0.0
            }
    
    def calculate_amino_acid_diversity(self, sequences: List[str]) -> float:
        """计算氨基酸使用多样性"""
        if not sequences:
            return 0.0
        
        # 统计所有氨基酸出现频率
        all_aa = ''.join([seq.upper() for seq in sequences if self.is_valid_sequence(seq)])
        if not all_aa:
            return 0.0
        
        aa_counts = Counter(all_aa)
        total_aa = len(all_aa)
        
        # 计算香农熵
        entropy = 0.0
        for count in aa_counts.values():
            p = count / total_aa
            if p > 0:
                entropy -= p * np.log2(p)
        
        # 归一化 (最大熵为log2(20) ≈ 4.32)
        max_entropy = np.log2(20)
        return entropy / max_entropy
    
    def calculate_sequence_similarity(self, sequences: List[str]) -> float:
        """计算序列间相似度"""
        valid_sequences = [seq for seq in sequences if self.is_valid_sequence(seq)]
        
        if len(valid_sequences) < 2:
            return 0.0
        
        # 使用编辑距离计算相似度
        similarities = []
        
        for i in range(len(valid_sequences)):
            for j in range(i + 1, len(valid_sequences)):
                seq1, seq2 = valid_sequences[i], valid_sequences[j]
                
                # 计算编辑距离
                edit_distance = self._edit_distance(seq1, seq2)
                max_len = max(len(seq1), len(seq2))
                
                if max_len > 0:
                    similarity = 1.0 - edit_distance / max_len
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]

class ActivityPredictor:
    """抗菌活性预测器"""
    
    def __init__(self):
        """初始化预测器"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 简单的基于规则的活性预测
        # 在实际应用中，可以替换为训练好的机器学习模型
        self.active_motifs = [
            'RR', 'KK', 'RK', 'KR',  # 带正电荷的基序
            'LL', 'LI', 'IL', 'VL',  # 疏水性基序
            'GG', 'AG', 'GA',        # 柔性区域
        ]
        
        self.activity_thresholds = {
            'min_length': 5,
            'max_length': 100,
            'min_positive_charge': 2,
            'min_hydrophobic_ratio': 0.3,
            'max_hydrophobic_ratio': 0.8
        }
    
    def predict_activity(self, sequences: List[str]) -> Tuple[List[float], List[bool]]:
        """
        预测序列的抗菌活性
        
        Args:
            sequences: 氨基酸序列列表
            
        Returns:
            (activity_scores, is_active): 活性分数和活性预测结果
        """
        activity_scores = []
        is_active = []
        
        for sequence in sequences:
            score, active = self._predict_single_sequence(sequence)
            activity_scores.append(score)
            is_active.append(active)
        
        return activity_scores, is_active
    
    def _predict_single_sequence(self, sequence: str) -> Tuple[float, bool]:
        """预测单个序列的活性"""
        if not sequence or len(sequence) < self.activity_thresholds['min_length']:
            return 0.0, False
        
        sequence = sequence.upper()
        length = len(sequence)
        
        # 特征提取
        features = self._extract_features(sequence)
        
        # 基于规则的评分
        score = 0.0
        
        # 1. 长度分数
        if self.activity_thresholds['min_length'] <= length <= self.activity_thresholds['max_length']:
            score += 0.2
        
        # 2. 电荷分数
        if features['positive_charge'] >= self.activity_thresholds['min_positive_charge']:
            score += 0.3
        
        # 3. 疏水性分数
        hydrophobic_ratio = features['hydrophobic_ratio']
        if (self.activity_thresholds['min_hydrophobic_ratio'] <= 
            hydrophobic_ratio <= self.activity_thresholds['max_hydrophobic_ratio']):
            score += 0.2
        
        # 4. 活性基序分数
        motif_score = 0.0
        for motif in self.active_motifs:
            if motif in sequence:
                motif_score += 0.1
        score += min(motif_score, 0.3)  # 最大0.3分
        
        # 归一化到[0, 1]
        score = min(score, 1.0)
        
        # 活性阈值
        is_active = score > 0.5
        
        return score, is_active
    
    def _extract_features(self, sequence: str) -> Dict[str, float]:
        """提取序列特征"""
        length = len(sequence)
        
        # 氨基酸计数
        positive_aa = 'RK'
        negative_aa = 'DE'
        hydrophobic_aa = 'ILVMFWYC'
        
        positive_count = sum(1 for aa in sequence if aa in positive_aa)
        negative_count = sum(1 for aa in sequence if aa in negative_aa)
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
        
        return {
            'positive_charge': positive_count - negative_count,
            'hydrophobic_ratio': hydrophobic_count / length if length > 0 else 0,
            'charge_density': (positive_count - negative_count) / length if length > 0 else 0
        }

class ModelEvaluator:
    """模型评估器主类"""
    
    def __init__(self, config):
        """初始化评估器"""
        self.config = config
        self.sequence_analyzer = SequenceAnalyzer()
        self.activity_predictor = ActivityPredictor()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 参考数据集 (用于比较新颖性)
        self.reference_sequences = self._load_reference_sequences()
    
    def _load_reference_sequences(self) -> List[str]:
        """加载参考序列数据集"""
        try:
            # 加载训练数据作为参考
            reference_file = getattr(self.config, 'reference_sequences_path', None)
            if reference_file:
                with open(reference_file, 'r') as f:
                    sequences = [line.strip() for line in f if line.strip()]
                return sequences
            else:
                return []
        except Exception as e:
            self.logger.warning(f"无法加载参考序列: {e}")
            return []
    
    def evaluate_generated_sequences(self, sequences: List[str]) -> EvaluationMetrics:
        """
        评估生成的序列
        
        Args:
            sequences: 生成的氨基酸序列列表
            
        Returns:
            评估指标
        """
        self.logger.info(f"评估 {len(sequences)} 个生成序列...")
        
        # 过滤有效序列
        valid_sequences = [seq for seq in sequences if self.sequence_analyzer.is_valid_sequence(seq)]
        valid_ratio = len(valid_sequences) / len(sequences) if sequences else 0.0
        
        if not valid_sequences:
            self.logger.warning("没有有效序列可供评估")
            return self._empty_metrics()
        
        # 计算序列质量指标
        lengths = [len(seq) for seq in valid_sequences]
        avg_length = np.mean(lengths)
        length_diversity = np.std(lengths) / avg_length if avg_length > 0 else 0.0
        
        # 计算氨基酸多样性
        aa_diversity = self.sequence_analyzer.calculate_amino_acid_diversity(valid_sequences)
        
        # 计算理化性质
        properties = [
            self.sequence_analyzer.calculate_physicochemical_properties(seq)
            for seq in valid_sequences
        ]
        
        avg_mw = np.mean([prop['molecular_weight'] for prop in properties])
        avg_pi = np.mean([prop['isoelectric_point'] for prop in properties])
        avg_hydrophobicity = np.mean([prop['hydrophobicity'] for prop in properties])
        avg_charge_density = np.mean([prop['charge_density'] for prop in properties])
        
        # 计算序列多样性
        pairwise_similarity = self.sequence_analyzer.calculate_sequence_similarity(valid_sequences)
        diversity_score = 1.0 - pairwise_similarity
        
        # 计算新颖性
        novel_ratio = self._calculate_novelty(valid_sequences)
        
        # 预测抗菌活性
        activity_scores, is_active = self.activity_predictor.predict_activity(valid_sequences)
        activity_ratio = np.mean(is_active) if is_active else 0.0
        avg_activity_score = np.mean(activity_scores) if activity_scores else 0.0
        
        # 构建评估指标
        metrics = EvaluationMetrics(
            avg_sequence_length=avg_length,
            length_diversity=length_diversity,
            amino_acid_diversity=aa_diversity,
            valid_sequences_ratio=valid_ratio,
            avg_molecular_weight=avg_mw,
            avg_isoelectric_point=avg_pi,
            avg_hydrophobicity=avg_hydrophobicity,
            avg_charge_density=avg_charge_density,
            pairwise_similarity=pairwise_similarity,
            sequence_diversity_score=diversity_score,
            novel_sequences_ratio=novel_ratio,
            predicted_activity_ratio=activity_ratio,
            avg_activity_score=avg_activity_score
        )
        
        self.logger.info(f"评估完成，有效序列比例: {valid_ratio:.3f}")
        return metrics
    
    def _calculate_novelty(self, sequences: List[str]) -> float:
        """计算序列新颖性"""
        if not self.reference_sequences:
            return 1.0  # 没有参考数据时假设都是新颖的
        
        novel_count = 0
        for seq in sequences:
            is_novel = True
            for ref_seq in self.reference_sequences:
                similarity = 1.0 - (
                    self.sequence_analyzer._edit_distance(seq, ref_seq) / 
                    max(len(seq), len(ref_seq))
                )
                if similarity > 0.8:  # 相似度阈值
                    is_novel = False
                    break
            
            if is_novel:
                novel_count += 1
        
        return novel_count / len(sequences) if sequences else 0.0
    
    def _empty_metrics(self) -> EvaluationMetrics:
        """返回空的评估指标"""
        return EvaluationMetrics(
            avg_sequence_length=0.0,
            length_diversity=0.0,
            amino_acid_diversity=0.0,
            valid_sequences_ratio=0.0,
            avg_molecular_weight=0.0,
            avg_isoelectric_point=0.0,
            avg_hydrophobicity=0.0,
            avg_charge_density=0.0,
            pairwise_similarity=0.0,
            sequence_diversity_score=0.0,
            novel_sequences_ratio=0.0,
            predicted_activity_ratio=0.0,
            avg_activity_score=0.0
        )
    
    def generate_evaluation_report(self, metrics: EvaluationMetrics, 
                                 output_path: str = None) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            output_path: 输出路径
            
        Returns:
            报告文本
        """
        report = f"""
抗菌肽生成模型评估报告
========================

序列质量指标:
- 平均序列长度: {metrics.avg_sequence_length:.2f}
- 长度多样性: {metrics.length_diversity:.3f}
- 氨基酸多样性: {metrics.amino_acid_diversity:.3f}
- 有效序列比例: {metrics.valid_sequences_ratio:.3f}

生物学特性:
- 平均分子量: {metrics.avg_molecular_weight:.2f} Da
- 平均等电点: {metrics.avg_isoelectric_point:.2f}
- 平均疏水性: {metrics.avg_hydrophobicity:.3f}
- 平均电荷密度: {metrics.avg_charge_density:.3f}

多样性指标:
- 序列间相似度: {metrics.pairwise_similarity:.3f}
- 序列多样性分数: {metrics.sequence_diversity_score:.3f}
- 新颖序列比例: {metrics.novel_sequences_ratio:.3f}

抗菌活性预测:
- 预测活性序列比例: {metrics.predicted_activity_ratio:.3f}
- 平均活性分数: {metrics.avg_activity_score:.3f}

评估总结:
- 模型生成了{'高质量' if metrics.valid_sequences_ratio > 0.8 else '中等质量' if metrics.valid_sequences_ratio > 0.5 else '低质量'}的序列
- 序列多样性{'良好' if metrics.sequence_diversity_score > 0.5 else '一般' if metrics.sequence_diversity_score > 0.3 else '较低'}
- 预测抗菌活性{'较高' if metrics.predicted_activity_ratio > 0.6 else '中等' if metrics.predicted_activity_ratio > 0.3 else '较低'}
"""
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"评估报告已保存到: {output_path}")
        
        return report

def main():
    """测试评估器"""
    # 示例序列
    test_sequences = [
        "RRWWLLKKLIPP",
        "KLAKLAKKLKLK",
        "RRIILVFYLL",
        "INVALID_SEQ_123",
        "GGRRKKLLII",
        "KWKLWKKLK"
    ]
    
    # 创建简单配置
    class SimpleConfig:
        pass
    
    config = SimpleConfig()
    
    # 创建评估器
    evaluator = ModelEvaluator(config)
    
    # 评估序列
    metrics = evaluator.evaluate_generated_sequences(test_sequences)
    
    # 生成报告
    report = evaluator.generate_evaluation_report(metrics)
    print(report)

if __name__ == "__main__":
    main()
