"""
数据加载器：处理抗菌肽序列数据，支持对比学习和扩散模型训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 氨基酸词汇表
AMINO_ACID_VOCAB = {
    'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9,
    'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
    'Y': 19, 'V': 20
}

VOCAB_TO_AA = {v: k for k, v in AMINO_ACID_VOCAB.items()}

def sequence_to_tokens(sequence: str, max_length: int = 100) -> torch.Tensor:
    """将氨基酸序列转换为token张量"""
    sequence = sequence.upper().strip()
    
    # 转换为token ID
    tokens = []
    for aa in sequence:
        if aa in AMINO_ACID_VOCAB:
            tokens.append(AMINO_ACID_VOCAB[aa])
        else:
            # 未知氨基酸跳过
            continue
    
    # 截断或填充到指定长度
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        # 用PAD填充
        tokens.extend([AMINO_ACID_VOCAB['PAD']] * (max_length - len(tokens)))
    
    return torch.tensor(tokens, dtype=torch.long)

def tokens_to_sequence(tokens: torch.Tensor) -> str:
    """将token张量转换为氨基酸序列"""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    
    sequence = ""
    for token in tokens:
        aa = VOCAB_TO_AA.get(int(token), 'X')
        if aa != 'PAD':
            sequence += aa
    
    return sequence.strip()

def create_attention_mask(tokens: torch.Tensor) -> torch.Tensor:
    """创建注意力掩码，PAD位置为0，其他为1"""
    return (tokens != AMINO_ACID_VOCAB['PAD']).long()

class AntimicrobialPeptideDataset(Dataset):
    """抗菌肽数据集 - 用于扩散模型训练"""
    
    def __init__(self, sequences_file: str, max_length: int = 100, tokenizer=None):
        """
        初始化数据集
        
        Args:
            sequences_file: 序列文件路径
            max_length: 最大序列长度
            tokenizer: 分词器（可选，用于兼容性）
        """
        self.sequences_file = sequences_file
        self.max_length = max_length
        
        # 加载序列
        self.sequences = self._load_sequences()
        logger.info(f"加载了 {len(self.sequences)} 个序列")
    
    def _load_sequences(self) -> List[str]:
        """加载序列文件"""
        sequences = []
        
        if not os.path.exists(self.sequences_file):
            raise FileNotFoundError(f"序列文件不存在: {self.sequences_file}")
        
        with open(self.sequences_file, 'r', encoding='utf-8') as f:
            for line in f:
                seq = line.strip()
                if seq and len(seq) >= 3:  # 过滤太短的序列
                    sequences.append(seq)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据项"""
        sequence = self.sequences[idx]
        
        # 转换为tokens
        input_ids = sequence_to_tokens(sequence, self.max_length)
        attention_mask = create_attention_mask(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sequences': sequence  # 原始序列用于ESM-2特征提取
        }
    
    def get_tokenizer(self):
        """返回简单的分词器（兼容接口）"""
        class SimpleTokenizer:
            def decode(self, tokens):
                return tokens_to_sequence(torch.tensor(tokens))
        
        return SimpleTokenizer()

class ContrastiveAMPDataset(Dataset):
    """对比学习数据集 - 正负样本对"""
    
    def __init__(self, positive_file: str, negative_file: str, 
                 max_length: int = 100, negative_sample_ratio: float = 1.0):
        """
        初始化对比学习数据集
        
        Args:
            positive_file: 正样本文件（anti-negative sequences）
            negative_file: 负样本文件（anti-positive sequences）
            max_length: 最大序列长度
            negative_sample_ratio: 负样本采样比例
        """
        self.max_length = max_length
        self.negative_sample_ratio = negative_sample_ratio
        
        # 加载正负样本
        self.positive_sequences = self._load_sequences(positive_file)
        self.negative_sequences = self._load_sequences(negative_file)
        
        logger.info(f"对比学习数据集: {len(self.positive_sequences)} 正样本, "
                   f"{len(self.negative_sequences)} 负样本")
    
    def _load_sequences(self, file_path: str) -> List[str]:
        """加载序列文件"""
        sequences = []
        
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return sequences
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                seq = line.strip()
                if seq and len(seq) >= 3:
                    sequences.append(seq)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.positive_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取正负样本对"""
        # 正样本
        positive_seq = self.positive_sequences[idx]
        
        # 随机选择负样本
        if self.negative_sequences:
            neg_idx = np.random.randint(0, len(self.negative_sequences))
            negative_seq = self.negative_sequences[neg_idx]
        else:
            # 如果没有负样本，使用其他正样本作为负样本
            neg_idx = (idx + 1) % len(self.positive_sequences)
            negative_seq = self.positive_sequences[neg_idx]
        
        return {
            'positive_sequences': [positive_seq],  # ESM-2需要序列列表
            'negative_sequences': [negative_seq],
            'positive_tokens': sequence_to_tokens(positive_seq, self.max_length),
            'negative_tokens': sequence_to_tokens(negative_seq, self.max_length)
        }

def collate_amp_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数 - 用于AntimicrobialPeptideDataset
    """
    # 收集所有数据
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    sequences = [item['sequences'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'sequences': sequences
    }

def collate_contrastive_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数 - 用于ContrastiveAMPDataset
    """
    # 收集正负样本序列
    positive_sequences = []
    negative_sequences = []
    
    for item in batch:
        positive_sequences.extend(item['positive_sequences'])
        negative_sequences.extend(item['negative_sequences'])
    
    return {
        'positive_sequences': positive_sequences,
        'negative_sequences': negative_sequences
    }

# 测试代码
if __name__ == "__main__":
    # 测试序列转换
    test_seq = "KRWWKWWRR"
    tokens = sequence_to_tokens(test_seq, 20)
    recovered_seq = tokens_to_sequence(tokens)
    
    print(f"原始序列: {test_seq}")
    print(f"Tokens: {tokens}")
    print(f"恢复序列: {recovered_seq}")
    print(f"注意力掩码: {create_attention_mask(tokens)}")
    
    # 测试数据集（如果文件存在）
    try:
        dataset = AntimicrobialPeptideDataset(
            "./main_training_sequences.txt", 
            max_length=50
        )
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本形状: {sample['input_ids'].shape}")
            print(f"样本序列: {sample['sequences']}")
    except FileNotFoundError:
        print("测试数据文件未找到，跳过数据集测试")
