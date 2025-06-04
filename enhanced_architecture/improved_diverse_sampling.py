#!/usr/bin/env python3
"""
改进的多样性采样器
基于当前已有的良好结果，进行更温和的优化
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
from data_loader import AMINO_ACID_VOCAB, VOCAB_TO_AA, sequence_to_tokens, tokens_to_sequence
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet

class ImprovedDiversitySampler:
    """改进的多样性采样器，更温和的调整策略"""
    
    def __init__(self):
        # 基于真实训练数据的目标分布
        self.target_distribution = {
            'K': 0.1136, 'G': 0.1045, 'L': 0.0897, 'R': 0.0888, 'A': 0.0719,
            'I': 0.0612, 'V': 0.0577, 'P': 0.0569, 'S': 0.0500, 'C': 0.0459,
            'F': 0.0436, 'T': 0.0362, 'N': 0.0320, 'Q': 0.0248, 'D': 0.0247,
            'E': 0.0238, 'W': 0.0216, 'Y': 0.0209, 'H': 0.0198, 'M': 0.0122
        }
        
        # 转换为token概率
        self.target_token_probs = torch.zeros(len(AMINO_ACID_VOCAB))
        for aa, prob in self.target_distribution.items():
            if aa in AMINO_ACID_VOCAB:
                token_id = AMINO_ACID_VOCAB[aa]
                self.target_token_probs[token_id] = prob
        
        print("✓ 改进的多样性采样器初始化完成")
    
    def gentle_diverse_sample(self, diffusion: D3PMDiffusion, batch_size: int, seq_len: int,
                             esm_features: Optional[torch.Tensor] = None,
                             num_inference_steps: Optional[int] = None,
                             diversity_strength: float = 0.1,  # 更温和的强度
                             temperature: float = 1.1,  # 略高的温度
                             anti_repeat_strength: float = 0.2) -> torch.Tensor:
        """
        温和的多样性采样
        
        Args:
            diffusion: 扩散模型
            batch_size: 批次大小
            seq_len: 序列长度
            esm_features: ESM特征
            num_inference_steps: 推理步数
            diversity_strength: 多样性强度（降低到0.1）
            temperature: 采样温度（略微提高）
            anti_repeat_strength: 防重复强度
        """
        if num_inference_steps is None:
            num_inference_steps = diffusion.scheduler.num_timesteps
        
        device = diffusion.device
        target_probs = self.target_token_probs.to(device)
        
        # 从随机氨基酸开始
        x = torch.randint(1, diffusion.scheduler.vocab_size, (batch_size, seq_len), device=device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(diffusion.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            predicted_logits = diffusion.model(x, t_batch, esm_features)
            
            # 屏蔽PAD token
            predicted_logits[:, :, 0] = float('-inf')
            
            if i < len(timesteps) - 1:
                # 温和的多样性调整
                adjusted_logits = self._apply_gentle_diversity_adjustment(
                    predicted_logits, x, target_probs, diversity_strength, anti_repeat_strength
                )
                
                # 温度缩放
                scaled_logits = adjusted_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                
                # 采样
                x = torch.multinomial(probs.view(-1, diffusion.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                # 最后一步使用argmax
                x = torch.argmax(predicted_logits, dim=-1)
        
        return x
    
    def _apply_gentle_diversity_adjustment(self, logits: torch.Tensor, current_x: torch.Tensor,
                                         target_probs: torch.Tensor, diversity_strength: float,
                                         anti_repeat_strength: float) -> torch.Tensor:
        """应用温和的多样性调整"""
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        adjustment = torch.zeros_like(logits)
        
        for b in range(batch_size):
            # 计算当前序列的分布
            current_counts = torch.bincount(current_x[b], minlength=vocab_size).float()
            current_dist = current_counts / (current_counts.sum() + 1e-8)
            
            # 1. 温和的全局多样性调整
            for pos in range(seq_len):
                # 只对严重偏离目标的氨基酸进行调整
                deviation = current_dist - target_probs
                
                # 对严重过多的氨基酸进行温和惩罚
                severely_overpresented = deviation > 0.05  # 超过5%才调整
                adjustment[b, pos, severely_overpresented] = -diversity_strength * deviation[severely_overpresented]
                
                # 对严重不足的氨基酸进行温和奖励
                severely_underpresented = deviation < -0.03  # 少于3%才调整
                adjustment[b, pos, severely_underpresented] = -diversity_strength * deviation[severely_underpresented] * 0.5
            
            # 2. 防重复调整（局部模式）
            if anti_repeat_strength > 0:
                seq = current_x[b]
                for pos in range(seq_len):
                    # 检查前面的重复模式
                    if pos > 0:
                        # 防止连续相同氨基酸
                        prev_token = seq[pos-1]
                        adjustment[b, pos, prev_token] -= anti_repeat_strength
                    
                    if pos > 1:
                        # 防止三连重复
                        if seq[pos-1] == seq[pos-2]:
                            repeat_token = seq[pos-1]
                            adjustment[b, pos, repeat_token] -= anti_repeat_strength * 2
        
        return logits + adjustment
    
    def evaluate_sequence_quality(self, sequences: List[str]) -> Dict:
        """评估序列质量"""
        if not sequences:
            return {"error": "No sequences provided"}
        
        # 统计氨基酸分布
        all_aa = ''.join(sequences)
        aa_counts = Counter(all_aa)
        total_aa = len(all_aa)
        
        aa_distribution = {}
        for aa, count in aa_counts.items():
            aa_distribution[aa] = count / total_aa
        
        # 计算与目标分布的偏差
        distribution_score = 0.0
        for aa, target_prob in self.target_distribution.items():
            actual_prob = aa_distribution.get(aa, 0.0)
            deviation = abs(actual_prob - target_prob)
            distribution_score += deviation
        
        # 计算多样性分数
        unique_aa = len(set(all_aa))
        diversity_score = unique_aa / 20  # 20种标准氨基酸
        
        # 检查重复模式
        repeat_patterns = 0
        for seq in sequences:
            for i in range(len(seq) - 2):
                if seq[i] == seq[i+1] == seq[i+2]:  # 三连重复
                    repeat_patterns += 1
        
        repeat_ratio = repeat_patterns / (len(sequences) * max(1, sum(len(s) for s in sequences) - 2))
        
        return {
            "total_sequences": len(sequences),
            "total_amino_acids": total_aa,
            "unique_amino_acids": unique_aa,
            "diversity_score": diversity_score,
            "distribution_deviation": distribution_score,
            "repeat_patterns": repeat_patterns,
            "repeat_ratio": repeat_ratio,
            "amino_acid_distribution": aa_distribution
        }


def test_improved_sampling():
    """测试改进的多样性采样"""
    print("=" * 60)
    print("🧪 测试改进的多样性采样")
    print("=" * 60)
    
    # 创建模型
    device = torch.device("cpu")
    vocab_size = len(AMINO_ACID_VOCAB)
    
    scheduler = D3PMScheduler(num_timesteps=100, vocab_size=vocab_size)
    unet = D3PMUNet(vocab_size=vocab_size, hidden_dim=256, num_layers=4, max_seq_len=50)
    diffusion = D3PMDiffusion(unet, scheduler, device)
    
    print("✓ 模型创建成功")
    
    # 创建改进的采样器
    sampler = ImprovedDiversitySampler()
    
    # 测试参数
    batch_size = 4
    seq_len = 25
    num_batches = 5
    
    print(f"\n📊 测试参数:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 批次数量: {num_batches}")
    print(f"  - 总序列数: {batch_size * num_batches}")
    
    # 标准采样
    print(f"\n🎲 1. 标准采样:")
    standard_sequences = []
    for _ in range(num_batches):
        generated = diffusion.sample(batch_size=batch_size, seq_len=seq_len, 
                                   num_inference_steps=20, temperature=1.0)
        for seq_tokens in generated:
            seq = tokens_to_sequence(seq_tokens)
            if seq:  # 只添加非空序列
                standard_sequences.append(seq)
    
    print(f"标准采样生成了 {len(standard_sequences)} 个有效序列")
    standard_quality = sampler.evaluate_sequence_quality(standard_sequences)
    
    # 改进的多样性采样
    print(f"\n🌈 2. 改进的多样性采样:")
    diverse_sequences = []
    for _ in range(num_batches):
        generated = sampler.gentle_diverse_sample(
            diffusion, batch_size=batch_size, seq_len=seq_len,
            num_inference_steps=20, diversity_strength=0.1,
            temperature=1.05, anti_repeat_strength=0.15
        )
        for seq_tokens in generated:
            seq = tokens_to_sequence(seq_tokens)
            if seq:  # 只添加非空序列
                diverse_sequences.append(seq)
    
    print(f"改进采样生成了 {len(diverse_sequences)} 个有效序列")
    diverse_quality = sampler.evaluate_sequence_quality(diverse_sequences)
    
    # 对比分析
    print(f"\n📈 3. 详细对比分析:")
    print(f"")
    print(f"序列质量指标:")
    print(f"  标准采样:")
    print(f"    - 多样性分数: {standard_quality['diversity_score']:.3f}")
    print(f"    - 分布偏差: {standard_quality['distribution_deviation']:.3f}")
    print(f"    - 重复模式: {standard_quality['repeat_patterns']}")
    print(f"    - 重复比例: {standard_quality['repeat_ratio']:.3f}")
    
    print(f"  改进采样:")
    print(f"    - 多样性分数: {diverse_quality['diversity_score']:.3f}")
    print(f"    - 分布偏差: {diverse_quality['distribution_deviation']:.3f}")
    print(f"    - 重复模式: {diverse_quality['repeat_patterns']}")
    print(f"    - 重复比例: {diverse_quality['repeat_ratio']:.3f}")
    
    # 氨基酸分布对比
    print(f"\n氨基酸分布对比:")
    all_aa = set(standard_quality['amino_acid_distribution'].keys()) | \
             set(diverse_quality['amino_acid_distribution'].keys())
    
    print(f"{'AA':>3} {'标准':>8} {'改进':>8} {'目标':>8} {'改善':>8}")
    print("-" * 40)
    
    improvements = []
    for aa in sorted(all_aa):
        standard_pct = standard_quality['amino_acid_distribution'].get(aa, 0) * 100
        diverse_pct = diverse_quality['amino_acid_distribution'].get(aa, 0) * 100
        target_pct = sampler.target_distribution.get(aa, 0) * 100
        
        # 计算改善程度（距离目标的偏差是否减小）
        standard_dev = abs(standard_pct - target_pct)
        diverse_dev = abs(diverse_pct - target_pct)
        improvement = (standard_dev - diverse_dev) / max(standard_dev, 0.001) * 100
        improvements.append(improvement)
        
        print(f"{aa:>3} {standard_pct:>7.1f}% {diverse_pct:>7.1f}% {target_pct:>7.1f}% {improvement:>+6.1f}%")
    
    # 序列示例
    print(f"\n🔍 4. 序列示例对比:")
    print(f"")
    print(f"标准采样示例:")
    for i, seq in enumerate(standard_sequences[:5]):
        print(f"  {i+1:2d}. {seq}")
    
    print(f"\n改进采样示例:")
    for i, seq in enumerate(diverse_sequences[:5]):
        print(f"  {i+1:2d}. {seq}")
    
    # 总结
    avg_improvement = np.mean(improvements)
    print(f"\n" + "=" * 60)
    if avg_improvement > 5:
        print(f"🎉 改进效果显著！平均改善: {avg_improvement:+.1f}%")
    elif avg_improvement > 0:
        print(f"✅ 改进效果温和。平均改善: {avg_improvement:+.1f}%")
    else:
        print(f"⚠️  改进效果有限。平均改善: {avg_improvement:+.1f}%")
    
    print(f"建议:")
    if diverse_quality['distribution_deviation'] < standard_quality['distribution_deviation']:
        print(f"  - 分布偏差有所改善，可继续使用改进采样")
    if diverse_quality['repeat_ratio'] < standard_quality['repeat_ratio']:
        print(f"  - 重复模式减少，序列质量提升")
    
    print("=" * 60)


if __name__ == "__main__":
    test_improved_sampling()
