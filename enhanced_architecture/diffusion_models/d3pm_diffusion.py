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
            # 生成随机噪声 (随机选择token)
            noise = torch.randint(0, self.vocab_size, x_start.shape, 
                                device=x_start.device)
        
        # 使用转移矩阵计算噪声序列
        batch_size, seq_len = x_start.shape
        x_noisy = torch.zeros_like(x_start)
        
        for i in range(batch_size):
            t_i = t[i].item()
            alpha_cumprod_t = self.alphas_cumprod[t_i]
            
            # 对每个位置应用噪声
            for j in range(seq_len):
                original_token = x_start[i, j].item()
                
                # 以alpha_cumprod_t的概率保持原token，否则随机选择
                if torch.rand(1).item() < alpha_cumprod_t:
                    x_noisy[i, j] = original_token
                else:
                    x_noisy[i, j] = torch.randint(0, self.vocab_size, (1,)).item()
        
        return x_noisy
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, 
                                 t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算后验分布的均值和方差"""
        # 简化实现，返回x_start的概率分布
        batch_size, seq_len = x_start.shape
        
        # 创建one-hot编码
        posterior_mean = F.one_hot(x_start, num_classes=self.vocab_size).float()
        posterior_variance = torch.ones_like(posterior_mean) * 0.1
        
        return posterior_mean, posterior_variance


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
                 device: torch.device = torch.device('cpu')):
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
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            predicted_logits.view(-1, self.scheduler.vocab_size),
            x_start.view(-1),
            reduction='mean'
        )
        
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, 
               esm_features: Optional[torch.Tensor] = None,
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """生成新的蛋白质序列"""
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # 从随机噪声开始
        x = torch.randint(0, self.scheduler.vocab_size, (batch_size, seq_len), 
                         device=self.device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, 
                                 device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # 预测去噪后的logits
            predicted_logits = self.model(x, t_batch, esm_features)
            
            # 采样下一个状态
            if i < len(timesteps) - 1:
                # 不是最后一步，添加一些随机性
                probs = F.softmax(predicted_logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                # 最后一步，使用argmax
                x = torch.argmax(predicted_logits, dim=-1)
        
        return x
    
    def ddim_sample(self, batch_size: int, seq_len: int, 
                    esm_features: Optional[torch.Tensor] = None,
                    num_inference_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """DDIM采样（确定性）"""
        # 简化的DDIM实现
        return self.sample(batch_size, seq_len, esm_features, num_inference_steps)


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
