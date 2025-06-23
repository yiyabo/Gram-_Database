#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专为条件扩散设计的ESM-2特征提取器
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import EsmModel, EsmTokenizer
import logging
from typing import List, Tuple, Optional, Dict
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttentionPooling(nn.Module):
    """注意力池化层 - 将变长序列转换为固定长度表示"""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, sequence_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            sequence_features: [batch_size, seq_len, feature_dim]
            attention_mask: [batch_size, seq_len] - 1表示有效token，0表示padding
        Returns:
            pooled_features: [batch_size, feature_dim]
        """
        # 计算注意力权重
        attention_scores = self.attention(sequence_features)  # [B, T, 1]
        
        if attention_mask is not None:
            # 将padding位置的attention score设为极小值
            attention_mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T, 1]
        
        # 加权求和
        pooled = torch.sum(sequence_features * attention_weights, dim=1)  # [B, D]
        
        return pooled

class ContrastiveLoss(nn.Module):
    """对比学习损失函数 (InfoNCE)"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        # features_a: anchor/positive, features_b: negative
        features_a = F.normalize(features_a, p=2, dim=1)
        features_b = F.normalize(features_b, p=2, dim=1)
        
        # 拉近 a 和 a' (同一批次内的其他正样本)
        # 推远 a 和 b (负样本)
        logits = torch.cat([features_a @ features_a.T, features_a @ features_b.T], dim=1)
        logits /= self.temperature
        
        # 正样本的标签是其在批次内的索引
        labels = torch.arange(len(features_a), device=features_a.device)
        
        # 屏蔽对角线上的自相似度
        mask = torch.eye(len(features_a), device=features_a.device).bool()
        logits.masked_fill_(mask, -1e9)

        return F.cross_entropy(logits, labels)

class ConditionalESM2FeatureExtractor(nn.Module):
    """专为条件扩散设计的ESM-2特征提取器，支持对比学习辅助任务"""
    
    def __init__(self,
                 model_name: str = "facebook/esm2_t33_650M_UR50D",
                 condition_dim: int = 512,
                 use_layers: List[int] = [6, 12, 18, 24],
                 pooling_strategy: str = "attention",
                 cache_dir: str = "./esm2_cache",
                 max_cache_size: int = 10000,
                 freeze_esm: bool = False, # 新增：是否冻结ESM-2
                 use_contrastive: bool = True): # 新增：是否使用对比学习
        super().__init__()
        
        self.model_name = model_name
        self.condition_dim = condition_dim
        self.use_layers = use_layers
        self.pooling_strategy = pooling_strategy
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载ESM-2模型
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm_model = EsmModel.from_pretrained(model_name, output_hidden_states=True)
        
        # 根据配置决定是否冻结ESM-2
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            logger.info("ESM-2预训练权重已冻结。")
        else:
            logger.info("ESM-2预训练权重将进行微调。")
        
        # 获取ESM-2维度
        self.esm_dim = self.esm_model.config.hidden_size
        
        # === 多层特征处理 ===
        self.layer_processors = nn.ModuleDict()
        for layer_idx in use_layers:
            self.layer_processors[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Linear(self.esm_dim, condition_dim // len(use_layers)),
                nn.LayerNorm(condition_dim // len(use_layers)),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # === 池化策略 ===
        if pooling_strategy == "attention":
            self.pooling_layers = nn.ModuleDict()
            for layer_idx in use_layers:
                self.pooling_layers[f"layer_{layer_idx}"] = AttentionPooling(
                    self.esm_dim, hidden_dim=128
                )
        
        # === 层融合机制 ===
        fused_input_dim = len(use_layers) * (condition_dim // len(use_layers))
        self.layer_fusion = nn.Sequential(
            nn.Linear(fused_input_dim, condition_dim * 2),
            nn.LayerNorm(condition_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(condition_dim * 2, condition_dim)
        )
        
        # === 缓存管理 ===
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # === 对比学习组件 ===
        if use_contrastive:
            self.contrastive_loss_fn = ContrastiveLoss()
            # 对比学习需要一个单独的投影头
            self.contrastive_projection = nn.Sequential(
                nn.Linear(self.esm_dim, self.esm_dim),
                nn.ReLU(),
                nn.Linear(self.esm_dim, 128) # 对比学习通常使用较小的特征维度
            )
        
        logger.info(f"ConditionalESM2FeatureExtractor初始化完成")
        logger.info(f"使用层: {use_layers}, 条件维度: {condition_dim}")
    
    def _compute_sequence_hash(self, sequences: List[str]) -> str:
        """计算序列列表的哈希值用于缓存"""
        combined = "|".join(sorted(sequences))
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _extract_multi_layer_features(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """提取多层ESM-2特征"""
        # Tokenization
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 移到设备
        device = next(self.esm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # ESM-2前向传播
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
        
        # 提取指定层的特征
        hidden_states = outputs.hidden_states  # Tuple of [B, L, D]
        attention_mask = inputs['attention_mask'][:, 1:-1]  # 去除CLS和SEP
        
        layer_features = {}
        for layer_idx in self.use_layers:
            # 获取该层特征（去除CLS和SEP tokens）
            layer_hidden = hidden_states[layer_idx][:, 1:-1, :]  # [B, L-2, D]
            
            # 池化
            if self.pooling_strategy == "attention":
                pooled = self.pooling_layers[f"layer_{layer_idx}"](
                    layer_hidden, attention_mask
                )
            elif self.pooling_strategy == "mean":
                masked_hidden = layer_hidden * attention_mask.unsqueeze(-1)
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            elif self.pooling_strategy == "max":
                masked_hidden = layer_hidden.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
                pooled = masked_hidden.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
            # 投影到目标维度
            processed = self.layer_processors[f"layer_{layer_idx}"](pooled)
            layer_features[f"layer_{layer_idx}"] = processed
        
        return layer_features
    
    def _fuse_layer_features(self, layer_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """融合多层特征"""
        # 拼接所有层的特征
        concatenated = torch.cat(list(layer_features.values()), dim=-1)  # [B, condition_dim]
        
        # 通过融合网络
        fused = self.layer_fusion(concatenated)  # [B, condition_dim]
        
        return fused
    
    def extract_condition_features(self, 
                                 reference_sequences: List[str],
                                 use_cache: bool = True) -> torch.Tensor:
        """
        提取条件特征的主要接口
        
        Args:
            reference_sequences: 参考序列列表
            use_cache: 是否使用缓存
        Returns:
            condition_features: [batch_size, condition_dim]
        """
        # 检查缓存
        if use_cache:
            seq_hash = self._compute_sequence_hash(reference_sequences)
            if seq_hash in self.cache:
                self.cache_hits += 1
                logger.debug(f"缓存命中: {seq_hash[:8]}...")
                return self.cache[seq_hash].clone()
        
        # 提取多层特征
        layer_features = self._extract_multi_layer_features(reference_sequences)
        
        # 融合特征
        condition_features = self._fuse_layer_features(layer_features)
        
        # 缓存结果
        if use_cache:
            self.cache_misses += 1
            if len(self.cache) < self.max_cache_size:
                self.cache[seq_hash] = condition_features.clone().detach()
                logger.debug(f"特征已缓存: {seq_hash[:8]}...")
        
        return condition_features
    
    def batch_extract_condition_features(self,
                                       reference_batches: List[List[str]],
                                       batch_size: int = 8) -> torch.Tensor:
        """
        批量提取条件特征
        
        Args:
            reference_batches: 参考序列批次列表
            batch_size: 处理批次大小
        Returns:
            all_features: [total_batches, condition_dim]
        """
        all_features = []
        
        for i in range(0, len(reference_batches), batch_size):
            batch_refs = reference_batches[i:i+batch_size]
            
            batch_features = []
            for ref_seqs in batch_refs:
                features = self.extract_condition_features(ref_seqs)
                # 如果有多个参考序列，取平均
                if features.shape[0] > 1:
                    features = features.mean(dim=0, keepdim=True)
                batch_features.append(features)
            
            batch_tensor = torch.cat(batch_features, dim=0)
            all_features.append(batch_tensor)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"已处理 {i+len(batch_refs)}/{len(reference_batches)} 批次")
        
        return torch.cat(all_features, dim=0)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "max_cache_size": self.max_cache_size
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("特征缓存已清空")
    
    def save_cache(self, cache_file: str):
        """保存缓存到文件"""
        cache_path = os.path.join(self.cache_dir, cache_file)
        torch.save({
            'cache': self.cache,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }, cache_path)
        logger.info(f"缓存已保存到: {cache_path}")
    
    def load_cache(self, cache_file: str):
        """从文件加载缓存"""
        cache_path = os.path.join(self.cache_dir, cache_file)
        if os.path.exists(cache_path):
            checkpoint = torch.load(cache_path, map_location='cpu')
            self.cache = checkpoint['cache']
            self.cache_hits = checkpoint.get('cache_hits', 0)
            self.cache_misses = checkpoint.get('cache_misses', 0)
            logger.info(f"缓存已从 {cache_path} 加载，大小: {len(self.cache)}")
        else:
            logger.warning(f"缓存文件不存在: {cache_path}")

    def _encode_for_contrastive(self, sequences: List[str]) -> torch.Tensor:
        """专用于对比学习的编码和投影"""
        inputs = self.tokenizer(
            sequences, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(next(self.esm_model.parameters()).device)
        
        # ESM-2前向传播，需要梯度
        outputs = self.esm_model(**inputs)
        
        # 使用[CLS] token的表示作为序列的全局表示
        cls_token_repr = outputs.last_hidden_state[:, 0, :]
        
        # 通过对比学习投影头
        contrastive_features = self.contrastive_projection(cls_token_repr)
        return contrastive_features

    def compute_contrastive_loss(self, positive_seqs: List[str], negative_seqs: List[str]) -> torch.Tensor:
        """计算对比学习损失"""
        if not hasattr(self, 'contrastive_loss_fn'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        pos_features = self._encode_for_contrastive(positive_seqs)
        neg_features = self._encode_for_contrastive(negative_seqs)
        
        return self.contrastive_loss_fn(pos_features, neg_features)


class ConditionalFeatureManager:
    """条件特征管理器 - 统一管理条件特征的提取、缓存和预处理"""
    
    def __init__(self, 
                 feature_extractor: ConditionalESM2FeatureExtractor,
                 precompute_cache_file: str = "precomputed_features.pt"):
        self.feature_extractor = feature_extractor
        self.precompute_cache_file = precompute_cache_file
        self.precomputed_features = {}
        
        # 尝试加载预计算特征
        self.load_precomputed_features()
    
    def precompute_dataset_features(self, 
                                  dataset_sequences: List[str],
                                  batch_size: int = 16,
                                  save_interval: int = 100):
        """预计算数据集中所有序列的特征"""
        logger.info(f"开始预计算 {len(dataset_sequences)} 条序列的特征...")
        
        for i in range(0, len(dataset_sequences), batch_size):
            batch_seqs = dataset_sequences[i:i+batch_size]
            
            for seq in batch_seqs:
                if seq not in self.precomputed_features:
                    features = self.feature_extractor.extract_condition_features([seq])
                    self.precomputed_features[seq] = features.squeeze(0)  # [condition_dim]
            
            # 定期保存
            if (i // batch_size + 1) % save_interval == 0:
                self.save_precomputed_features()
                logger.info(f"已预计算 {i+len(batch_seqs)}/{len(dataset_sequences)} 序列")
        
        self.save_precomputed_features()
        logger.info(f"预计算完成，共 {len(self.precomputed_features)} 条特征")
    
    def get_condition_features(self, reference_sequences: List[str]) -> torch.Tensor:
        """获取条件特征（优先使用预计算特征）"""
        features = []
        
        for seq in reference_sequences:
            if seq in self.precomputed_features:
                features.append(self.precomputed_features[seq])
            else:
                # 实时计算
                feat = self.feature_extractor.extract_condition_features([seq])
                features.append(feat.squeeze(0))
        
        return torch.stack(features)  # [num_refs, condition_dim]
    
    def save_precomputed_features(self):
        """保存预计算特征"""
        cache_path = os.path.join(
            self.feature_extractor.cache_dir, 
            self.precompute_cache_file
        )
        torch.save(self.precomputed_features, cache_path)
        logger.debug(f"预计算特征已保存: {len(self.precomputed_features)} 条")
    
    def load_precomputed_features(self):
        """加载预计算特征"""
        cache_path = os.path.join(
            self.feature_extractor.cache_dir, 
            self.precompute_cache_file
        )
        if os.path.exists(cache_path):
            self.precomputed_features = torch.load(cache_path, map_location='cpu')
            logger.info(f"预计算特征已加载: {len(self.precomputed_features)} 条")
        else:
            logger.info("未找到预计算特征文件，将从空开始")

if __name__ == '__main__':
    # 测试代码
    logger.info("开始测试 ConditionalESM2FeatureExtractor...")
    
    # 模拟序列
    test_sequences = [
        "GLFDIVKKVVGALGSLGLVVR",
        "KWVKAMDGVIDMLFYKMVYK",
        "FLGALFKALAALFVSSSK",
        "GLFDIVKKVVGALGSLGLVVR"  # 重复序列用于测试缓存
    ]
    
    # 初始化提取器
    extractor = ConditionalESM2FeatureExtractor(
        model_name="facebook/esm2_t8_215M_UR50D", # 使用小模型进行测试
        condition_dim=128,
        use_layers=[2, 4, 6],
        cache_dir="./test_cache"
    )
    
    # 移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.to(device)
    
    # 第一次提取
    logger.info("第一次提取特征...")
    features1 = extractor.extract_condition_features(test_sequences)
    logger.info(f"特征形状: {features1.shape}")
    logger.info(f"缓存状态: {extractor.get_cache_stats()}")
    
    # 第二次提取（应该命中缓存）
    logger.info("\n第二次提取特征（测试缓存）...")
    features2 = extractor.extract_condition_features(test_sequences)
    logger.info(f"缓存状态: {extractor.get_cache_stats()}")
    
    # 验证两次提取结果是否一致
    assert torch.allclose(features1, features2), "缓存结果与首次计算不一致"
    logger.info("✅ 缓存功能验证成功！")
    
    # 测试特征管理器
    logger.info("\n测试 ConditionalFeatureManager...")
    manager = ConditionalFeatureManager(extractor)
    
    # 预计算
    manager.precompute_dataset_features(test_sequences[:3])
    
    # 获取特征
    logger.info("从管理器获取特征...")
    manager_features = manager.get_condition_features(test_sequences)
    logger.info(f"管理器提取的特征形状: {manager_features.shape}")
    
    # 清理测试缓存
    import shutil
    if os.path.exists("./test_cache"):
        shutil.rmtree("./test_cache")
        logger.info("测试缓存目录已清理")
        
    logger.info("✅ ConditionalESM2FeatureExtractor 测试完成！")