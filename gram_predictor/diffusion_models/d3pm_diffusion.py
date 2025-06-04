"""
D3PM (Discrete Denoising Diffusion Probabilistic Models) implementation for protein sequences
Specifically designed for antimicrobial peptide generation with discrete amino acid tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

# 导入统一的词汇表和序列处理函数
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AMINO_ACID_VOCAB, VOCAB_TO_AA, sequence_to_tokens, tokens_to_sequence

class D3PMScheduler:
    """噪声调度器，用于控制扩散过程中的噪声水平"""
    
    def __init__(self, num_timesteps: int = 1000, vocab_size: int = None, 
                 schedule_type: str = 'linear'):
        self.num_timesteps = num_timesteps
        # 使用统一的词汇表大小
        self.vocab_size = vocab_size if vocab_size is not None else len(AMINO_ACID_VOCAB)
        self.schedule_type = schedule_type
        
        # 创建噪声调度
        if schedule_type == 'linear':
            self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 为离散扩散计算转移矩阵
        self._compute_transition_matrices()
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """余弦噪声调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _compute_transition_matrices(self):
        """计算离散扩散的转移矩阵"""
        # 均匀转移矩阵 (添加噪声)
        self.uniform_prob = 1.0 / self.vocab_size
        
        # Qt matrices for forward process
        self.Qt = []
        for t in range(self.num_timesteps):
            # 在时间步t的转移概率
            alpha_t = self.alphas_cumprod[t]
            
            # 创建转移矩阵: (1-beta_t) * I + beta_t * uniform
            Qt = torch.zeros(self.vocab_size, self.vocab_size)
            
            # 对角线元素 (保持原状的概率)
            diagonal_prob = alpha_t + (1 - alpha_t) * self.uniform_prob
            Qt.fill_diagonal_(diagonal_prob)
            
            # 非对角线元素 (转移到其他token的概率)
            off_diagonal_prob = (1 - alpha_t) * self.uniform_prob
            Qt.fill_(off_diagonal_prob)
            Qt.fill_diagonal_(diagonal_prob)
            
            self.Qt.append(Qt)
        
        self.Qt = torch.stack(self.Qt)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程，给干净序列添加噪声"""
        if noise is None:
            # 生成随机噪声 (随机选择非PAD token)
            # 确保噪声不包含PAD token (0)
            noise = torch.randint(1, self.vocab_size, x_start.shape, 
                                device=x_start.device)
        
        # 矢量化实现：更高效的批量处理
        batch_size, seq_len = x_start.shape
        
        # 创建PAD掩码：PAD位置永远不变
        pad_mask = (x_start == 0)  # PAD token = 0
        
        # 获取每个样本对应的alpha_cumprod_t
        alpha_cumprod_t = self.alphas_cumprod[t]  # [batch_size]
        alpha_cumprod_t = alpha_cumprod_t.view(batch_size, 1)  # [batch_size, 1]
        
        # 生成随机概率矩阵
        rand_probs = torch.rand(x_start.shape, device=x_start.device)  # [batch_size, seq_len]
        
        # 创建掩码：True表示保持原token，False表示使用噪声
        keep_mask = rand_probs < alpha_cumprod_t  # 广播比较
        
        # 应用掩码：保持原token或使用噪声
        x_noisy = torch.where(keep_mask, x_start, noise)
        
        # 确保PAD位置永远不变
        x_noisy = torch.where(pad_mask, x_start, x_noisy)
        
        return x_noisy
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, 
                                 t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1}|x_t, x_0) 的均值和方差
        
        基于D3PM论文的公式:
        q(x_{t-1}|x_t, x_0) ∝ q(x_t|x_{t-1}) * q(x_{t-1}|x_0)
        
        Args:
            x_start: 原始清洁序列 [batch_size, seq_len]
            x_t: 当前时间步的噪声序列 [batch_size, seq_len]  
            t: 时间步 [batch_size]
        Returns:
            posterior_mean: 后验均值 [batch_size, seq_len, vocab_size]
            posterior_log_variance: 后验对数方差 [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x_start.shape
        device = x_start.device
        
        # 处理边界情况：t=0时没有前一个时间步
        t = t.clamp(min=1)
        
        # 获取转移概率参数
        alpha_cumprod_t = self.alphas_cumprod[t]  # [batch_size]
        alpha_cumprod_t_prev = self.alphas_cumprod[t-1]  # [batch_size]
        alpha_t = self.alphas[t]  # [batch_size]
        
        # 为了方便计算，将参数扩展到正确的维度
        alpha_cumprod_t = alpha_cumprod_t.view(batch_size, 1, 1)  # [batch_size, 1, 1]
        alpha_cumprod_t_prev = alpha_cumprod_t_prev.view(batch_size, 1, 1)
        alpha_t = alpha_t.view(batch_size, 1, 1)
        
        # 创建one-hot编码
        x_start_onehot = F.one_hot(x_start, num_classes=self.vocab_size).float()  # [batch_size, seq_len, vocab_size]
        x_t_onehot = F.one_hot(x_t, num_classes=self.vocab_size).float()  # [batch_size, seq_len, vocab_size]
        
        # 计算 q(x_{t-1}|x_0) 的概率分布
        # 对于离散扩散，这是从原始状态经过t-1步后的分布
        q_t_minus_1_given_0 = alpha_cumprod_t_prev * x_start_onehot + \
                              (1 - alpha_cumprod_t_prev) / self.vocab_size
        
        # 计算 q(x_t|x_{t-1}) 的概率分布
        # 这需要对所有可能的x_{t-1}进行计算
        uniform_prob = torch.ones(batch_size, seq_len, self.vocab_size, device=device) / self.vocab_size
        
        # 计算后验分布 q(x_{t-1}|x_t, x_0)
        # 使用贝叶斯公式: p(A|B,C) ∝ p(B|A) * p(A|C)
        posterior_unnormalized = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        for k in range(self.vocab_size):
            # 对于每个可能的 x_{t-1} = k
            x_t_minus_1_k = torch.full_like(x_start, k)  # [batch_size, seq_len]
            
            # q(x_{t-1}|x_0) 在 x_{t-1} = k 时的概率
            q_t_minus_1_k_given_0 = q_t_minus_1_given_0[:, :, k]  # [batch_size, seq_len]
            
            # q(x_t|x_{t-1}=k) 的概率
            # 如果 x_t == k，概率为 alpha_t + (1-alpha_t)/vocab_size
            # 如果 x_t != k，概率为 (1-alpha_t)/vocab_size
            q_t_given_t_minus_1_k = torch.where(
                x_t == k,
                alpha_t.squeeze(-1) + (1 - alpha_t.squeeze(-1)) / self.vocab_size,
                (1 - alpha_t.squeeze(-1)) / self.vocab_size
            )  # [batch_size, seq_len]
            
            # 后验概率 ∝ q(x_t|x_{t-1}=k) * q(x_{t-1}=k|x_0)
            posterior_unnormalized[:, :, k] = q_t_given_t_minus_1_k * q_t_minus_1_k_given_0
        
        # 归一化
        posterior_sum = posterior_unnormalized.sum(dim=-1, keepdim=True)
        posterior_sum = torch.clamp(posterior_sum, min=1e-8)  # 避免除零
        posterior_mean = posterior_unnormalized / posterior_sum
        
        # 计算方差 (对于离散分布，使用概率分布的方差公式)
        # Var[X] = E[X^2] - E[X]^2
        # 对于分类分布，方差可以计算为 p(1-p)
        posterior_variance = posterior_mean * (1 - posterior_mean)
        
        # 返回对数方差以保持数值稳定性
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-8))
        
        return posterior_mean, posterior_log_variance


class D3PMUNet(nn.Module):
    """用于蛋白质序列的U-Net架构扩散模型"""
    
    def __init__(self, vocab_size: int = 21, hidden_dim: int = 512, 
                 num_layers: int = 8, num_heads: int = 8, 
                 max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # 时间步嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层 - 预测每个位置每个token的概率
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """生成时间步的正弦位置编码"""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                esm_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 噪声序列 [batch_size, seq_len]
            timesteps: 时间步 [batch_size]
            esm_features: ESM-2特征 [batch_size, esm_dim] (可选)
        Returns:
            预测的去噪logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Token嵌入
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, hidden_dim]
        
        # 时间步嵌入
        time_emb = self._get_time_embedding(timesteps)  # [batch_size, hidden_dim]
        time_emb = self.time_embedding(time_emb)  # [batch_size, hidden_dim]
        
        # 组合嵌入
        h = token_emb + pos_emb + time_emb.unsqueeze(1)  # [batch_size, seq_len, hidden_dim]
        
        # 如果有ESM特征，进行特征融合
        if esm_features is not None:
            # 简单的特征融合：通过线性层投影ESM特征并加到序列特征上
            if not hasattr(self, 'esm_projection'):
                self.esm_projection = nn.Linear(esm_features.shape[-1], 
                                              self.hidden_dim).to(x.device)
            esm_proj = self.esm_projection(esm_features)  # [batch_size, hidden_dim]
            h = h + esm_proj.unsqueeze(1)  # 广播到所有位置
        
        h = self.dropout(h)
        
        # Transformer处理
        h = self.transformer(h)  # [batch_size, seq_len, hidden_dim]
        
        # 输出投影
        logits = self.output_projection(h)  # [batch_size, seq_len, vocab_size]
        
        return logits


class D3PMDiffusion:
    """D3PM扩散模型的主要类"""
    
    def __init__(self, model: D3PMUNet, scheduler: D3PMScheduler, 
                 device: torch.device = torch.device('cuda')):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # 移动到设备
        self.model.to(device)
        if hasattr(scheduler, 'Qt'):
            scheduler.Qt = scheduler.Qt.to(device)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    def training_loss(self, x_start: torch.Tensor, 
                     esm_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算训练损失"""
        batch_size = x_start.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), 
                         device=self.device)
        
        # 前向扩散：添加噪声
        x_noisy = self.scheduler.q_sample(x_start, t)
        
        # 预测去噪后的logits
        predicted_logits = self.model(x_noisy, t, esm_features)
        
        # 创建掩码，只在非PAD位置计算损失
        non_pad_mask = (x_start != 0)  # PAD token = 0
        
        if non_pad_mask.sum() == 0:
            # 如果所有位置都是PAD，返回零损失
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 只在非PAD位置计算损失
        masked_logits = predicted_logits[non_pad_mask]  # [num_non_pad, vocab_size]
        masked_targets = x_start[non_pad_mask]  # [num_non_pad]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            reduction='mean'
        )
        
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, 
               esm_features: Optional[torch.Tensor] = None,
               num_inference_steps: Optional[int] = None,
               temperature: float = 1.0) -> torch.Tensor:
        """
        生成新的蛋白质序列
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            esm_features: ESM-2特征（可选）
            num_inference_steps: 推理步数
            temperature: 采样温度，控制随机性（1.0=标准，>1.0更随机，<1.0更确定）
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # 从随机氨基酸开始（不包含PAD token）
        x = torch.randint(1, self.scheduler.vocab_size, (batch_size, seq_len), 
                         device=self.device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, 
                                 device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # 预测去噪后的logits
            predicted_logits = self.model(x, t_batch, esm_features)
            
            # 屏蔽PAD token的概率，避免生成PAD
            predicted_logits[:, :, 0] = float('-inf')  # PAD token probability = 0
            
            # 采样下一个状态
            if i < len(timesteps) - 1:
                # 不是最后一步，添加温度控制的随机性
                scaled_logits = predicted_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                # 最后一步，使用argmax确保确定性
                x = torch.argmax(predicted_logits, dim=-1)
        
        return x
    
    def ddim_sample(self, batch_size: int, seq_len: int, 
                    esm_features: Optional[torch.Tensor] = None,
                    num_inference_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """DDIM采样（确定性）"""
        # 简化的DDIM实现
        return self.sample(batch_size, seq_len, esm_features, num_inference_steps)
    
    def top_k_sample(self, batch_size: int, seq_len: int, 
                     esm_features: Optional[torch.Tensor] = None,
                     num_inference_steps: Optional[int] = None,
                     k: int = 10, temperature: float = 1.0) -> torch.Tensor:
        """Top-k采样：只从概率最高的k个token中采样"""
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # 从随机氨基酸开始（不包含PAD token）
        x = torch.randint(1, self.scheduler.vocab_size, (batch_size, seq_len), 
                         device=self.device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, 
                                 device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            predicted_logits = self.model(x, t_batch, esm_features)
            
            # 屏蔽PAD token
            predicted_logits[:, :, 0] = float('-inf')
            
            if i < len(timesteps) - 1:
                # Top-k采样
                scaled_logits = predicted_logits / temperature
                
                # 获取top-k logits
                top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
                
                # 创建掩码，只保留top-k
                mask = torch.full_like(scaled_logits, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_logits)
                
                probs = F.softmax(mask, dim=-1)
                x = torch.multinomial(probs.view(-1, self.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                x = torch.argmax(predicted_logits, dim=-1)
        
        return x
    
    def nucleus_sample(self, batch_size: int, seq_len: int, 
                       esm_features: Optional[torch.Tensor] = None,
                       num_inference_steps: Optional[int] = None,
                       p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """Nucleus (top-p)采样：从累积概率达到p的最小token集合中采样"""
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # 从随机氨基酸开始（不包含PAD token）
        x = torch.randint(1, self.scheduler.vocab_size, (batch_size, seq_len), 
                         device=self.device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, 
                                 device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            predicted_logits = self.model(x, t_batch, esm_features)
            
            # 屏蔽PAD token
            predicted_logits[:, :, 0] = float('-inf')
            
            if i < len(timesteps) - 1:
                # Nucleus采样
                scaled_logits = predicted_logits / temperature
                
                # 排序概率
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 创建nucleus掩码
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 应用掩码
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(scaled_logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                x = torch.argmax(predicted_logits, dim=-1)
        
        return x
    
    def diverse_sample(self, batch_size: int, seq_len: int, 
                      esm_features: Optional[torch.Tensor] = None,
                      num_inference_steps: Optional[int] = None,
                      diversity_strength: float = 0.3, 
                      temperature: float = 1.0) -> torch.Tensor:
        """
        多样性感知采样：防止过度生成某些氨基酸
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            esm_features: ESM-2特征（可选）
            num_inference_steps: 推理步数
            diversity_strength: 多样性强度 (0-1)
            temperature: 采样温度
        """
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # 设置目标氨基酸分布（基于训练数据）
        target_distribution = {
            'K': 0.1136, 'G': 0.1045, 'L': 0.0897, 'R': 0.0888, 'A': 0.0719,
            'I': 0.0612, 'V': 0.0577, 'P': 0.0569, 'S': 0.0500, 'C': 0.0459,
            'F': 0.0436, 'T': 0.0362, 'N': 0.0320, 'Q': 0.0248, 'D': 0.0247,
            'E': 0.0238, 'W': 0.0216, 'Y': 0.0209, 'H': 0.0198, 'M': 0.0122
        }
        
        # 转换为token分布
        target_token_probs = torch.zeros(self.scheduler.vocab_size, device=self.device)
        for aa, prob in target_distribution.items():
            if aa in AMINO_ACID_VOCAB:
                token_id = AMINO_ACID_VOCAB[aa]
                target_token_probs[token_id] = prob
        target_token_probs = target_token_probs / target_token_probs.sum()
        
        # 从随机氨基酸开始（不包含PAD token）
        x = torch.randint(1, self.scheduler.vocab_size, (batch_size, seq_len), 
                         device=self.device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, 
                                 device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            predicted_logits = self.model(x, t_batch, esm_features)
            
            # 屏蔽PAD token
            predicted_logits[:, :, 0] = float('-inf')
            
            # 应用多样性调整
            if diversity_strength > 0:
                # 计算当前序列的氨基酸分布
                current_distributions = []
                for b in range(batch_size):
                    current_counts = torch.bincount(x[b], minlength=self.scheduler.vocab_size).float()
                    current_dist = current_counts / (current_counts.sum() + 1e-8)
                    current_distributions.append(current_dist)
                current_distributions = torch.stack(current_distributions)
                
                # 计算分布偏差调整
                diversity_adjustment = torch.zeros_like(predicted_logits)
                for b in range(batch_size):
                    for pos in range(seq_len):
                        # 惩罚过度出现的氨基酸
                        overpresented = current_distributions[b] > target_token_probs * 1.5
                        diversity_adjustment[b, pos, overpresented] = -diversity_strength * 3
                        
                        # 奖励不足的氨基酸
                        underpresented = current_distributions[b] < target_token_probs * 0.5
                        diversity_adjustment[b, pos, underpresented] = diversity_strength * 2
                
                predicted_logits = predicted_logits + diversity_adjustment
            
            # 采样下一个状态
            if i < len(timesteps) - 1:
                # 温度缩放和采样
                scaled_logits = predicted_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                # 最后一步，使用argmax
                x = torch.argmax(predicted_logits, dim=-1)
        
        return x
        

# 导入统一的词汇表和序列处理函数
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AMINO_ACID_VOCAB, VOCAB_TO_AA, sequence_to_tokens, tokens_to_sequence


if __name__ == "__main__":
    # 测试代码
    print("Testing D3PM Diffusion Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型和调度器
    scheduler = D3PMScheduler(num_timesteps=1000, vocab_size=21)
    model = D3PMUNet(vocab_size=21, hidden_dim=256, num_layers=4, max_seq_len=50)
    diffusion = D3PMDiffusion(model, scheduler, device)
    
    # 测试训练
    batch_size = 4
    seq_len = 30
    
    # 创建假数据
    x = torch.randint(0, 21, (batch_size, seq_len), device=device)
    
    # 计算损失
    loss = diffusion.training_loss(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # 测试生成
    generated = diffusion.sample(batch_size=2, seq_len=20)
    print(f"Generated sequences shape: {generated.shape}")
    
    # 转换为氨基酸序列
    for i, seq_tokens in enumerate(generated):
        seq = tokens_to_sequence(seq_tokens)
        print(f"Generated sequence {i+1}: {seq}")
    
    print("D3PM test completed!")
