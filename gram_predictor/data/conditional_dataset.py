#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
条件扩散模型的数据集和数据加载器
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Dict, Optional, Callable

import logging
import Levenshtein # 用于计算编辑距离

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConditionalDataset(Dataset):
    """
    为条件扩散模型准备数据的数据集类。
    根据指定的策略，为每个目标序列配对一个或多个参考序列。
    """
    
    def __init__(self, 
                 sequences: List[str], 
                 pairing_strategy: str = 'random',
                 num_references: int = 1,
                 strategy_config: Optional[Dict] = None):
        """
        初始化条件数据集。
        
        Args:
            sequences: 所有的序列列表。
            pairing_strategy: 配对策略 ('random', 'similarity', 'cluster').
            num_references: 每个目标序列配对的参考序列数量。
            strategy_config: 特定策略的配置字典。
        """
        super().__init__()
        
        if not sequences:
            raise ValueError("序列列表不能为空。")
            
        self.sequences = sequences
        self.pairing_strategy = pairing_strategy
        self.num_references = num_references
        self.strategy_config = strategy_config or {}
        
        self.num_sequences = len(self.sequences)
        
        logger.info(f"条件数据集初始化完成。")
        logger.info(f"总序列数: {self.num_sequences}")
        logger.info(f"配对策略: {self.pairing_strategy}")
        logger.info(f"每个目标的参考序列数: {self.num_references}")
        
        # 为特定策略进行预处理
        self._prepare_strategy()

    def _prepare_strategy(self):
        """根据选择的策略进行必要的预计算。"""
        if self.pairing_strategy == 'random':
            # 随机策略不需要预计算
            pass
        elif self.pairing_strategy == 'similarity':
            # 我们采用动态计算的方式，以节省内存，所以这里也不需要预计算。
            logger.info("使用基于'similarity'的动态配对策略。")
            pass
        elif self.pairing_strategy == 'cluster':
            logger.info("正在为'cluster'策略进行预计算...")
            # TODO: 实现基于聚类的预计算，例如运行CD-HIT或MMseqs2
            # self.clusters = self._perform_clustering()
            raise NotImplementedError("'cluster'策略尚未实现。")
        else:
            raise ValueError(f"未知的配对策略: {self.pairing_strategy}")

    def __len__(self) -> int:
        """返回数据集中的序列总数。"""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        """
        获取一个数据样本。
        
        Args:
            idx: 目标序列的索引。
            
        Returns:
            一个字典，包含:
            - 'target_sequence': 目标序列。
            - 'reference_sequences': 参考序列列表。
        """
        target_sequence = self.sequences[idx]
        
        if self.pairing_strategy == 'random':
            reference_sequences = self._get_random_references(idx)
        elif self.pairing_strategy == 'similarity':
            reference_sequences = self._get_similarity_references(idx)
        elif self.pairing_strategy == 'cluster':
            reference_sequences = self._get_cluster_references(idx)
        else:
            # 这个情况理论上不会发生，因为在__init__中已经检查过了
            raise ValueError(f"未知的配对策略: {self.pairing_strategy}")
            
        return {
            'target_sequence': [target_sequence], # 保持格式一致，都是列表
            'reference_sequences': reference_sequences
        }

    def _get_random_references(self, target_idx: int) -> List[str]:
        """获取随机的参考序列。"""
        # 创建一个不包含目标索引的索引池
        possible_indices = list(range(self.num_sequences))
        possible_indices.pop(target_idx)
        
        # 从中随机选择
        num_to_sample = min(self.num_references, len(possible_indices))
        reference_indices = random.sample(possible_indices, num_to_sample)
        
        return [self.sequences[i] for i in reference_indices]

    def _get_similarity_references(self, target_idx: int) -> List[str]:
        """获取基于Levenshtein距离的最相似的参考序列。"""
        target_seq = self.sequences[target_idx]
        
        distances = []
        for i, seq in enumerate(self.sequences):
            if i == target_idx:
                continue
            # 计算归一化的Levenshtein距离，使其不受序列长度影响
            dist = Levenshtein.distance(target_seq, seq) / max(len(target_seq), len(seq))
            distances.append((dist, i))
            
        # 按距离从小到大排序
        distances.sort(key=lambda x: x[0])
        
        # 获取最近的 K 个序列的索引
        num_to_sample = min(self.num_references, len(distances))
        reference_indices = [idx for dist, idx in distances[:num_to_sample]]
        
        return [self.sequences[i] for i in reference_indices]

    def _get_cluster_references(self, target_idx: int) -> List[str]:
        """获取基于聚类的参考序列。"""
        # 这是需要实现的占位符
        raise NotImplementedError("'cluster'策略的采样尚未实现。")


def load_sequences_from_file(file_path: str) -> List[str]:
    """从文本文件中加载序列，每行一个序列。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"序列文件未找到: {file_path}")
    
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    logger.info(f"从 {file_path} 加载了 {len(sequences)} 条序列。")
    return sequences


if __name__ == '__main__':
    # 测试代码
    logger.info("开始测试 ConditionalDataset...")
    
    # 创建一个模拟的序列文件
    mock_sequences = [
        "A" * 10, "B" * 10, "C" * 10, "D" * 10, "E" * 10,
        "F" * 10, "G" * 10, "H" * 10, "I" * 10, "J" * 10,
    ]
    mock_file_path = "mock_sequences.txt"
    with open(mock_file_path, 'w') as f:
        for seq in mock_sequences:
            f.write(seq + "\n")
            
    # 1. 加载序列
    sequences = load_sequences_from_file(mock_file_path)
    
    # 2. 测试随机配对策略
    logger.info("\n--- 测试 'random' 策略 ---")
    try:
        random_dataset = ConditionalDataset(
            sequences, 
            pairing_strategy='random', 
            num_references=3
        )
        
        # 检查长度
        assert len(random_dataset) == len(sequences)
        logger.info(f"数据集长度正确: {len(random_dataset)}")
        
        # 获取一个样本并检查
        sample = random_dataset[0]
        logger.info(f"样本 0: {sample}")
        
        assert 'target_sequence' in sample
        assert 'reference_sequences' in sample
        assert isinstance(sample['target_sequence'], list)
        assert isinstance(sample['reference_sequences'], list)
        assert len(sample['reference_sequences']) == 3
        assert sample['target_sequence'][0] not in sample['reference_sequences']
        logger.info("✅ 'random' 策略测试通过！")
        
    except Exception as e:
        logger.error(f"❌ 'random' 策略测试失败: {e}")

    # 3. 测试相似度配对策略
    logger.info("\n--- 测试 'similarity' 策略 ---")
    try:
        # 创建一组距离可预测的序列
        sim_sequences = [
            "ABCDE",  # 目标
            "ABCDG",  # dist=1
            "ABCEG",  # dist=2
            "AXCYG",  # dist=3
            "XXXXX",  # dist=5
        ]
        similarity_dataset = ConditionalDataset(
            sim_sequences,
            pairing_strategy='similarity',
            num_references=2
        )
        
        # 获取第一个序列 ("ABCDE") 的样本
        sample_sim = similarity_dataset[0]
        logger.info(f"相似度配对样本: {sample_sim}")
        
        # 预期的参考序列应该是 "ABCDG" 和 "ABCEG"
        expected_refs = ["ABCDG", "ABCEG"]
        assert len(sample_sim['reference_sequences']) == 2
        assert set(sample_sim['reference_sequences']) == set(expected_refs)
        logger.info("✅ 'similarity' 策略测试通过！")
        
    except ImportError:
        logger.warning("⚠️  'similarity' 策略测试跳过，因为未安装 'python-Levenshtein' 库。")
        logger.warning("请运行: pip install python-Levenshtein")
    except Exception as e:
        logger.error(f"❌ 'similarity' 策略测试失败: {e}", exc_info=True)

    # 4. 测试未实现的策略
    logger.info("\n--- 测试 'cluster' 策略 ---")
    try:
        ConditionalDataset(sequences, pairing_strategy='cluster')
    except NotImplementedError:
        logger.info("✅ 'cluster' 策略按预期抛出 NotImplementedError。")
    except Exception as e:
        logger.error(f"❌ 'cluster' 策略测试异常: {e}")

    # 清理模拟文件
    os.remove(mock_file_path)
    logger.info(f"已清理模拟文件: {mock_file_path}")
    
    logger.info("\nConditionalDataset 测试完成！")