#!/usr/bin/env python3
"""
修复扩散模型生成多样性问题
解决甘氨酸过度生成的问题
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
from data_loader import AMINO_ACID_VOCAB, VOCAB_TO_AA, sequence_to_tokens, tokens_to_sequence

class DiversityAwareSampler:
    """多样性感知采样器，防止过度生成某些氨基酸"""
    
    def __init__(self, vocab_size: int = 21, target_distribution: Optional[Dict[str, float]] = None):
        self.vocab_size = vocab_size
        
        # 设置目标氨基酸分布（基于训练数据）
        if target_distribution is None:
            self.target_distribution = {
                'K': 0.1136, 'G': 0.1045, 'L': 0.0897, 'R': 0.0888, 'A': 0.0719,
                'I': 0.0612, 'V': 0.0577, 'P': 0.0569, 'S': 0.0500, 'C': 0.0459,
                'F': 0.0436, 'T': 0.0362, 'N': 0.0320, 'Q': 0.0248, 'D': 0.0247,
                'E': 0.0238, 'W': 0.0216, 'Y': 0.0209, 'H': 0.0198, 'M': 0.0122
            }
        else:
            self.target_distribution = target_distribution
        
        # 转换为token分布
        self.target_token_probs = torch.zeros(vocab_size)
        for aa, prob in self.target_distribution.items():
            if aa in AMINO_ACID_VOCAB:
                token_id = AMINO_ACID_VOCAB[aa]
                self.target_token_probs[token_id] = prob
        
        # 归一化
        self.target_token_probs = self.target_token_probs / self.target_token_probs.sum()
        
        print(f"✓ 多样性采样器初始化，目标分布设置完成")
    
    def diverse_sampling(self, logits: torch.Tensor, generated_so_far: torch.Tensor, 
                        diversity_strength: float = 0.3, temperature: float = 1.0) -> torch.Tensor:
        """
        多样性采样，考虑当前生成序列的氨基酸分布
        
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            generated_so_far: 已生成的序列 [batch_size, current_len]
            diversity_strength: 多样性强度 (0-1)
            temperature: 采样温度
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # 将目标分布移到正确设备
        target_probs = self.target_token_probs.to(device)
        
        # 计算当前序列的氨基酸分布
        current_distributions = []
        for i in range(batch_size):
            if generated_so_far.shape[1] > 0:
                # 统计当前序列中各氨基酸的出现次数
                current_counts = torch.bincount(generated_so_far[i], minlength=vocab_size).float()
                current_dist = current_counts / (current_counts.sum() + 1e-8)
            else:
                current_dist = torch.zeros(vocab_size, device=device)
            current_distributions.append(current_dist)
        
        current_distributions = torch.stack(current_distributions)  # [batch_size, vocab_size]
        
        # 计算分布偏差惩罚
        distribution_penalty = torch.zeros_like(logits)
        for i in range(batch_size):
            for pos in range(seq_len):
                # 对于过度出现的氨基酸，降低其概率
                overpresented_mask = current_distributions[i] > target_probs * 2  # 超过目标2倍
                distribution_penalty[i, pos, overpresented_mask] = -diversity_strength * 2
                
                # 对于不足的氨基酸，提高其概率
                underpresented_mask = current_distributions[i] < target_probs * 0.5  # 低于目标一半
                distribution_penalty[i, pos, underpresented_mask] = diversity_strength
        
        # 应用多样性惩罚
        adjusted_logits = logits + distribution_penalty
        
        # 温度缩放
        scaled_logits = adjusted_logits / temperature
        
        # 防止PAD token被选中
        scaled_logits[:, :, 0] = float('-inf')  # PAD token
        
        # 采样
        probs = F.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch_size, seq_len)
        
        return samples
    
    def nucleus_sampling(self, logits: torch.Tensor, generated_so_far: torch.Tensor,
                        top_p: float = 0.9, diversity_strength: float = 0.2,
                        temperature: float = 1.0) -> torch.Tensor:
        """Nucleus (top-p) 采样with多样性控制"""
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # 应用多样性调整
        adjusted_logits = self._apply_diversity_adjustment(logits, generated_so_far, diversity_strength)
        
        # 温度缩放
        scaled_logits = adjusted_logits / temperature
        
        # 防止PAD token
        scaled_logits[:, :, 0] = float('-inf')
        
        # 应用nucleus采样
        probs = F.softmax(scaled_logits, dim=-1)
        
        samples = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            for j in range(seq_len):
                # 排序概率
                sorted_probs, sorted_indices = torch.sort(probs[i, j], descending=True)
                
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                
                # 找到nucleus边界
                nucleus_mask = cumulative_probs <= top_p
                if nucleus_mask.sum() == 0:
                    nucleus_mask[0] = True  # 至少保留一个
                
                # 从nucleus中采样
                nucleus_probs = sorted_probs[nucleus_mask]
                nucleus_indices = sorted_indices[nucleus_mask]
                
                # 重新归一化
                nucleus_probs = nucleus_probs / nucleus_probs.sum()
                
                # 采样
                selected_idx = torch.multinomial(nucleus_probs, num_samples=1)
                samples[i, j] = nucleus_indices[selected_idx]
        
        return samples
    
    def _apply_diversity_adjustment(self, logits: torch.Tensor, generated_so_far: torch.Tensor,
                                   diversity_strength: float) -> torch.Tensor:
        """应用多样性调整"""
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        target_probs = self.target_token_probs.to(device)
        
        # 计算当前分布
        current_distributions = []
        for i in range(batch_size):
            if generated_so_far.shape[1] > 0:
                current_counts = torch.bincount(generated_so_far[i], minlength=vocab_size).float()
                current_dist = current_counts / (current_counts.sum() + 1e-8)
            else:
                current_dist = torch.zeros(vocab_size, device=device)
            current_distributions.append(current_dist)
        
        current_distributions = torch.stack(current_distributions)
        
        # 计算调整
        adjustment = torch.zeros_like(logits)
        for i in range(batch_size):
            for pos in range(seq_len):
                # 对过度出现的氨基酸进行惩罚
                overpresented = current_distributions[i] > target_probs * 1.5
                adjustment[i, pos, overpresented] = -diversity_strength * 3
                
                # 对不足的氨基酸进行奖励
                underpresented = current_distributions[i] < target_probs * 0.3
                adjustment[i, pos, underpresented] = diversity_strength * 2
        
        return logits + adjustment


def test_diversity_sampler():
    """测试多样性采样器"""
    print("🧪 测试多样性采样器...")
    
    # 创建采样器
    sampler = DiversityAwareSampler()
    
    # 模拟一些logits (偏向G)
    batch_size, seq_len, vocab_size = 2, 10, 21
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 人为提高G(token_id=8)的概率
    logits[:, :, 8] += 3.0  # 大幅提高G的logits
    
    # 模拟已生成的序列（大量G）
    generated_so_far = torch.full((batch_size, 5), 8)  # 全是G
    
    print("原始logits中G的相对概率:")
    probs = F.softmax(logits[0, 0], dim=0)
    print(f"G (token_8): {probs[8]:.3f}")
    
    # 使用多样性采样
    diverse_samples = sampler.diverse_sampling(
        logits, generated_so_far, 
        diversity_strength=0.5, 
        temperature=1.0
    )
    
    print("\n多样性采样结果:")
    for i in range(batch_size):
        sample_seq = tokens_to_sequence(diverse_samples[i])
        print(f"序列{i+1}: {sample_seq}")
        
        # 统计氨基酸分布
        unique, counts = torch.unique(diverse_samples[i], return_counts=True)
        aa_counts = {}
        for token_id, count in zip(unique, counts):
            if token_id.item() in VOCAB_TO_AA:
                aa = VOCAB_TO_AA[token_id.item()]
                aa_counts[aa] = count.item()
        
        print(f"  氨基酸分布: {aa_counts}")
    
    print("✓ 多样性采样器测试完成")


if __name__ == "__main__":
    test_diversity_sampler()
