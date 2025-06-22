#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
条件离散去噪扩散概率模型 (Conditional D3PM)
深度融合ESM-2特征，实现基于参考序列的条件生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

# 导入现有的调度器和词汇表
from .d3pm_diffusion import D3PMScheduler
from gram_predictor.data_loader import AMINO_ACID_VOCAB

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConditionalTransformerLayer(nn.Module):
    """条件化的Transformer层，在每层都融合条件信息"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 条件交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 序列特征 [batch_size, seq_len, hidden_dim]
            condition: 条件特征 [batch_size, hidden_dim]
        """
        # 自注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # 条件交叉注意力
        x_norm = self.norm2(x)
        condition_kv = condition.unsqueeze(1)  # [B, 1, H]
        cross_attn_out, _ = self.cross_attention(
            query=x_norm,
            key=condition_kv,
            value=condition_kv
        )
        x = x + self.dropout(cross_attn_out)
        
        # 前馈网络
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        
        return x


class ConditionalD3PMUNet(nn.Module):
    """条件扩散U-Net，深度融合ESM-2特征"""
    
    def __init__(self, 
                 vocab_size: int = len(AMINO_ACID_VOCAB), 
                 hidden_dim: int = 512, 
                 num_layers: int = 8, 
                 num_heads: int = 8, 
                 max_seq_len: int = 100, 
                 dropout: float = 0.1,
                 condition_dim: int = 512, # 必须与ESM-2特征提取器的输出维度一致
                 condition_dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.condition_dim = condition_dim
        
        # === 基础嵌入层 ===
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # === 时间步嵌入 ===
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # === 条件特征处理 ===
        # 假设条件特征已经由外部的ESM-2提取器处理好
        # 只需要一个简单的线性投影来匹配维度（如果需要）
        if condition_dim != hidden_dim:
            self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        else:
            self.condition_projection = nn.Identity()
        
        # === 条件化Transformer层 ===
        self.transformer_layers = nn.ModuleList([
            ConditionalTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # === 输出层 ===
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # === 条件缺失处理 (用于Classifier-Free Guidance) ===
        self.null_condition = nn.Parameter(torch.randn(hidden_dim))
        self.condition_dropout = condition_dropout
        
    def _get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """生成时间步的正弦位置编码"""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, 
                x: torch.Tensor, 
                timesteps: torch.Tensor, 
                condition_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        条件扩散前向传播
        
        Args:
            x: 噪声序列 [batch_size, seq_len]
            timesteps: 时间步 [batch_size]
            condition_features: 条件特征 [batch_size, condition_dim] (可选)
        """
        batch_size, seq_len = x.shape
        
        # === 基础特征嵌入 ===
        token_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        time_emb = self._get_time_embedding(timesteps)
        time_emb = self.time_embedding(time_emb)
        
        h = token_emb + pos_emb + time_emb.unsqueeze(1)
        
        # === 条件特征处理 ===
        if condition_features is not None:
            # 投影条件特征
            condition_emb = self.condition_projection(condition_features) # [B, H]
            
            # Classifier-Free Guidance: 随机丢弃条件
            if self.training and self.condition_dropout > 0:
                dropout_mask = torch.rand(batch_size, 1, device=x.device) < self.condition_dropout
                condition_emb = torch.where(dropout_mask, self.null_condition, condition_emb)
        else:
            # 无条件生成，使用null condition
            condition_emb = self.null_condition.unsqueeze(0).expand(batch_size, -1)
        
        h = self.dropout(h)
        
        # === 条件化Transformer处理 ===
        for layer in self.transformer_layers:
            h = layer(h, condition_emb)
        
        # === 输出预测 ===
        logits = self.output_projection(h)
        
        return logits

class ConditionalD3PMDiffusion:
    """条件D3PM扩散模型的主类，包含训练和采样逻辑"""

    def __init__(self, model: ConditionalD3PMUNet, scheduler: D3PMScheduler,
                 device: torch.device = torch.device('cuda')):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        self.model.to(device)
        # 确保调度器的张量也在正确的设备上
        if hasattr(self.scheduler, 'alphas_cumprod'):
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        if hasattr(self.scheduler, 'Qt'):
            self.scheduler.Qt = self.scheduler.Qt.to(self.device)

    def training_loss(self, x_start: torch.Tensor,
                      condition_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算训练损失 (L_simple)
        
        Args:
            x_start: 原始清洁序列 [batch_size, seq_len]
            condition_features: 条件特征 [batch_size, condition_dim]
        """
        batch_size = x_start.shape[0]
        
        # 1. 随机采样时间步
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,),
                          device=self.device, dtype=torch.long)
        
        # 2. 前向扩散：添加噪声
        x_noisy = self.scheduler.q_sample(x_start, t)
        
        # 3. 模型预测去噪后的logits
        #    模型内部会处理Classifier-Free Guidance的条件丢弃
        predicted_logits = self.model(x_noisy, t, condition_features)
        
        # 4. 计算损失
        # 创建掩码，只在非PAD位置计算损失
        non_pad_mask = (x_start != 0)
        if non_pad_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        masked_logits = predicted_logits[non_pad_mask]
        masked_targets = x_start[non_pad_mask]
        
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        
        return loss

    @torch.no_grad()
    def sample(self,
               batch_size: int,
               seq_len: int,
               condition_features: Optional[torch.Tensor] = None,
               guidance_scale: float = 5.0,
               temperature: float = 1.0) -> torch.Tensor:
        """
        使用Classifier-Free Guidance生成新的蛋白质序列
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            condition_features: 条件特征 [batch_size, condition_dim]
            guidance_scale: 指导强度。1.0表示无指导。
            temperature: 采样温度。
        """
        self.model.eval()
        
        # 从随机氨基酸开始（不包含PAD token）
        x = torch.randint(1, self.scheduler.vocab_size, (batch_size, seq_len),
                          device=self.device)
        
        # 逆向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0,
                                   self.scheduler.num_timesteps, dtype=torch.long,
                                   device=self.device)
        
        for t in timesteps:
            t_batch = t.repeat(batch_size)
            
            # Classifier-Free Guidance (CFG)
            # 1. 预测有条件的logits
            logits_cond = self.model(x, t_batch, condition_features)
            
            # 2. 预测无条件的logits
            logits_uncond = self.model(x, t_batch, None)
            
            # 3. 结合有条件和无条件的预测
            #    guidance_scale=1.0时，完全依赖条件预测
            #    guidance_scale=0.0时，完全是无条件生成
            guided_logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            
            # 屏蔽PAD token的概率，避免生成PAD
            guided_logits[:, :, 0] = float('-inf')
            
            # 应用温度
            scaled_logits = guided_logits / temperature
            
            # 从概率分布中采样
            probs = F.softmax(scaled_logits, dim=-1)
            
            # 在最后一步使用argmax以获得确定性结果，否则进行多项式采样
            if t == 0:
                x = torch.argmax(probs, dim=-1)
            else:
                x = torch.multinomial(probs.view(-1, self.scheduler.vocab_size),
                                      num_samples=1).view(batch_size, seq_len)
        
        self.model.train()
        return x


if __name__ == '__main__':
    # 测试代码
    logger.info("开始测试 Conditional D3PM 完整流程...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型参数
    B, L, V, H, C = 4, 50, 21, 128, 256
    NUM_TIMESTEPS = 100 # 使用较少的时间步进行快速测试
    
    # 1. 初始化所有组件
    logger.info("\n--- 1. 初始化组件 ---")
    scheduler = D3PMScheduler(num_timesteps=NUM_TIMESTEPS, vocab_size=V)
    unet = ConditionalD3PMUNet(
        vocab_size=V,
        hidden_dim=H,
        num_layers=2,
        num_heads=4,
        max_seq_len=L,
        condition_dim=C
    )
    diffusion_model = ConditionalD3PMDiffusion(unet, scheduler, device)
    logger.info("✅ 组件初始化成功！")

    # 2. 测试训练损失计算
    logger.info("\n--- 2. 测试 training_loss ---")
    try:
        x_start = torch.randint(1, V, (B, L)).to(device)
        condition_feats = torch.randn(B, C).to(device)
        
        loss = diffusion_model.training_loss(x_start, condition_feats)
        assert loss.item() > 0
        assert loss.requires_grad
        logger.info(f"计算得到的损失: {loss.item():.4f}")
        logger.info("✅ training_loss 测试通过！")
    except Exception as e:
        logger.error(f"❌ training_loss 测试失败: {e}", exc_info=True)

    # 3. 测试采样生成
    logger.info("\n--- 3. 测试 sample ---")
    try:
        # 使用更少的推理步骤以加快测试
        scheduler.num_timesteps = 10
        
        # 有条件生成
        logger.info("测试有条件生成...")
        generated_cond = diffusion_model.sample(
            batch_size=B,
            seq_len=L,
            condition_features=condition_feats
        )
        assert generated_cond.shape == (B, L)
        assert not torch.any(generated_cond == 0) # 确保没有生成PAD token
        logger.info("✅ 有条件生成测试通过！")
        
        # 无条件生成
        logger.info("测试无条件生成...")
        generated_uncond = diffusion_model.sample(
            batch_size=B,
            seq_len=L,
            condition_features=None
        )
        assert generated_uncond.shape == (B, L)
        logger.info("✅ 无条件生成测试通过！")
        
    except Exception as e:
        logger.error(f"❌ sample 测试失败: {e}", exc_info=True)

    logger.info("\nConditional D3PM 完整流程测试完成！")