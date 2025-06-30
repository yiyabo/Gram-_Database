#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESM-2增强辅助编码器
替换原有的从头训练Transformer，使用Meta的ESM-2预训练蛋白质语言模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import EsmModel, EsmTokenizer
import logging
from typing import List, Tuple, Optional

# 导入统一配置
from gram_predictor.config.model_config import ESM2Config

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
    """对比学习损失函数"""
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, positive_features: torch.Tensor, negative_features: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        Args:
            positive_features: [batch_size, feature_dim] 正样本特征
            negative_features: [batch_size, feature_dim] 负样本特征
        Returns:
            loss: 对比损失值
        """
        # L2归一化
        pos_norm = F.normalize(positive_features, p=2, dim=1)
        neg_norm = F.normalize(negative_features, p=2, dim=1)
        
        # 计算相似度矩阵
        pos_sim = torch.matmul(pos_norm, pos_norm.T) / self.temperature  # 正样本内部相似度
        neg_sim = torch.matmul(pos_norm, neg_norm.T) / self.temperature  # 正负样本相似度
        
        # InfoNCE损失 - 拉近正样本，推远负样本
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)
        
        # 避免自相似度
        pos_mask = torch.eye(pos_exp.size(0), device=pos_exp.device).bool()
        pos_exp = pos_exp.masked_fill(pos_mask, 0)
        
        # 计算损失
        numerator = pos_exp.sum(dim=1)
        denominator = pos_exp.sum(dim=1) + neg_exp.sum(dim=1)
        
        loss = -torch.log(numerator / (denominator + 1e-8)).mean()
        
        return loss

class ESM2AuxiliaryEncoder(nn.Module):
    """ESM-2增强辅助编码器"""
    def __init__(self, config: ESM2Config = None):
        super().__init__()
        self.config = config or ESM2Config()
        
        logger.info(f"初始化ESM-2模型: {self.config.model_name}")
        
        # 加载ESM-2预训练模型
        try:
            # 首先尝试从本地缓存加载
            try:
                logger.info("尝试从本地缓存加载ESM-2模型...")
                self.tokenizer = EsmTokenizer.from_pretrained(
                    self.config.model_name, 
                    local_files_only=True
                )
                self.esm_model = EsmModel.from_pretrained(
                    self.config.model_name, 
                    local_files_only=True
                )
                logger.info("✅ 从本地缓存加载ESM-2模型成功")
                
            except Exception as cache_error:
                logger.warning(f"本地缓存加载失败: {str(cache_error)[:100]}...")
                logger.info("尝试在线下载模型...")
                
                # 设置环境变量优化下载
                import os
                import requests
                
                # 检查代理连接
                proxy_working = False
                if 'http_proxy' in os.environ or 'https_proxy' in os.environ:
                    try:
                        # 测试代理连接
                        proxy_url = os.environ.get('https_proxy', os.environ.get('http_proxy'))
                        logger.info(f"检测到代理设置: {proxy_url}")
                        
                        # 简单测试代理是否可用
                        test_response = requests.get('https://httpbin.org/ip', 
                                                   proxies={'https': proxy_url, 'http': proxy_url}, 
                                                   timeout=5)
                        if test_response.status_code == 200:
                            proxy_working = True
                            logger.info("✅ 代理连接正常")
                        else:
                            logger.warning("⚠️  代理连接异常")
                    except Exception as proxy_error:
                        logger.warning(f"⚠️  代理测试失败: {str(proxy_error)[:50]}...")
                        logger.info("尝试禁用代理进行下载...")
                        # 临时禁用代理
                        for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
                            if proxy_var in os.environ:
                                os.environ[f'_BACKUP_{proxy_var}'] = os.environ[proxy_var]
                                del os.environ[proxy_var]
                
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
                # 尝试在线下载
                self.tokenizer = EsmTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    resume_download=True
                )
                self.esm_model = EsmModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    resume_download=True
                )
                logger.info("✅ 在线下载ESM-2模型成功")
            
            if self.config.freeze_esm:
                # 冻结预训练权重
                for param in self.esm_model.parameters():
                    param.requires_grad = False
                logger.info("ESM-2预训练权重已冻结")
            
        except Exception as e:
            logger.error(f"加载ESM-2模型失败: {e}")
            logger.error("请确保已安装transformers库: pip install transformers")
            logger.error("如果网络连接有问题，请配置代理或使用离线模式")
            raise
        
        # 获取ESM-2输出维度
        esm_dim = self.esm_model.config.hidden_size
        logger.info(f"ESM-2特征维度: {esm_dim}")
        
        # 注意力池化层
        if self.config.pooling_method == 'attention':
            self.pooling = AttentionPooling(esm_dim, hidden_dim=128)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)  # 简单平均池化
        
        # 投影层 (将ESM特征投影到目标维度)
        self.projection = nn.Sequential(
            nn.Linear(esm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.config.projection_dim)
        )
        
        # 对比学习组件
        if self.config.use_contrastive_learning:
            self.contrastive_loss = ContrastiveLoss(
                temperature=self.config.contrastive_temperature,
                margin=self.config.contrastive_margin
            )
            logger.info("对比学习组件已启用")
        
        # 设备配置 - 强制使用CPU以避免MPS兼容性问题
        # device = "cpu"
        # 如果在服务器上，可以改为：
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        logger.info(f"ESM2AuxiliaryEncoder初始化完成，设备: {device}")
    
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        标准的PyTorch forward方法，调用encode_sequences
        
        Args:
            sequences: 蛋白质序列列表
        Returns:
            features: [batch_size, output_dim] 特征张量
        """
        return self.encode_sequences(sequences)
    
    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        编码蛋白质序列为固定维度特征
        
        Args:
            sequences: 蛋白质序列列表
        Returns:
            features: [batch_size, output_dim] 特征张量
        """
        # 分词和编码
        inputs = self.tokenizer(
            sequences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.config.max_length
        )
        
        # 移到正确设备（与模型一致）
        device = next(self.esm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # ESM-2前向传播
        # 只有在非训练模式时才使用no_grad，训练时需要梯度流动
        if self.training:
            outputs = self.esm_model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
        
        # 获取序列表示 (去除CLS和SEP tokens)
        sequence_repr = outputs.last_hidden_state[:, 1:-1, :]  # [B, T-2, D]
        
        # 创建注意力掩码 (去除CLS和SEP tokens)
        attention_mask = inputs['attention_mask'][:, 1:-1]  # [B, T-2]
        
        # 注意力池化
        if self.config.pooling_method == 'attention':
            pooled_features = self.pooling(sequence_repr, attention_mask)
        else:
            # 简单平均池化 (考虑mask)
            masked_repr = sequence_repr * attention_mask.unsqueeze(-1)
            pooled_features = masked_repr.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # 投影到目标维度
        features = self.projection(pooled_features)
        
        return features
    
    def contrastive_train_step(self, positive_seqs: List[str], negative_seqs: List[str]) -> torch.Tensor:
        """
        对比学习训练步骤
        
        Args:
            positive_seqs: 正样本序列（抗阴性菌）
            negative_seqs: 负样本序列（只抗阳性菌）
        Returns:
            contrastive_loss: 对比损失值
        """
        if not self.config.use_contrastive_learning:
            raise ValueError("对比学习未启用，请在配置中设置use_contrastive_learning=True")
        
        # 编码正负样本
        pos_features = self.encode_sequences(positive_seqs)  # [B, D]
        neg_features = self.encode_sequences(negative_seqs)  # [B, D]
        
        # 计算对比损失
        loss = self.contrastive_loss(pos_features, neg_features)
        
        return loss
    
    def extract_contrastive_features(self, positive_seqs: List[str], negative_seqs: List[str],
                                   batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取对比特征（正负样本）
        
        Args:
            positive_seqs: 正样本序列列表
            negative_seqs: 负样本序列列表
            batch_size: 批处理大小
        Returns:
            positive_features, negative_features: 正负样本特征张量
        """
        logger.info(f"提取对比特征: {len(positive_seqs)}条正样本, {len(negative_seqs)}条负样本")
        
        # 设置为训练模式以保持梯度信息
        self.train()
        
        # 确保序列数量匹配
        min_len = min(len(positive_seqs), len(negative_seqs))
        if len(positive_seqs) != len(negative_seqs):
            logger.warning(f"正负样本数量不匹配，截取到最小长度: {min_len}")
            positive_seqs = positive_seqs[:min_len]
            negative_seqs = negative_seqs[:min_len]
        
        # 直接使用encode_sequences获取特征，保持梯度信息
        pos_features = self.encode_sequences(positive_seqs)
        neg_features = self.encode_sequences(negative_seqs)
        
        return pos_features, neg_features
    
    def extract_auxiliary_features(self, sequences: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        从辅助序列中提取全局特征 (批量处理)
        
        Args:
            sequences: 辅助序列列表
            batch_size: 批处理大小，None使用config默认值
        Returns:
            features: [num_sequences, output_dim] numpy数组
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        self.eval()
        all_features = []
        
        logger.info(f"开始提取{len(sequences)}条序列的ESM-2特征...")
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_features = self.encode_sequences(batch_seqs)
            all_features.append(batch_features.detach().cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"已处理 {i+len(batch_seqs)}/{len(sequences)} 序列")
        
        features = np.vstack(all_features)
        logger.info(f"ESM-2特征提取完成，形状: {features.shape}")
        
        return features

def load_auxiliary_sequences(aux_data_path: str = "data/natureAMP.txt") -> List[str]:
    """加载辅助序列数据"""
    import pandas as pd
    
    if not os.path.exists(aux_data_path):
        raise FileNotFoundError(f"辅助数据文件未找到: {aux_data_path}")
    
    # 读取序列 (假设第一列是序列)
    df = pd.read_csv(aux_data_path, header=None)
    sequences = df.iloc[:, 0].tolist()
    
    # 基本清理
    sequences = [str(seq).upper().strip() for seq in sequences if str(seq).strip()]
    
    logger.info(f"从 {aux_data_path} 加载了 {len(sequences)} 条辅助序列")
    return sequences

def load_contrastive_datasets(base_path: str = "enhanced_architecture") -> Tuple[List[str], List[str], List[str]]:
    """
    加载对比学习数据集
    
    Args:
        base_path: 数据文件所在目录
    Returns:
        main_seqs, positive_seqs, negative_seqs: 主训练集、正样本、负样本
    """
    main_file = os.path.join(base_path, "main_training_sequences.txt")
    pos_file = os.path.join(base_path, "positive_sequences.txt")
    neg_file = os.path.join(base_path, "negative_sequences.txt")
    
    def load_txt_sequences(file_path: str) -> List[str]:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    main_seqs = load_txt_sequences(main_file)
    positive_seqs = load_txt_sequences(pos_file)
    negative_seqs = load_txt_sequences(neg_file)
    
    logger.info(f"加载对比学习数据集:")
    logger.info(f"  主训练集: {len(main_seqs)} 条")
    logger.info(f"  正样本: {len(positive_seqs)} 条")
    logger.info(f"  负样本: {len(negative_seqs)} 条")
    
    return main_seqs, positive_seqs, negative_seqs

def fuse_global_features(features: np.ndarray, method: str = 'clustering', n_clusters: int = 5,
                        negative_features: Optional[np.ndarray] = None) -> np.ndarray:
    """
    融合全局特征 (支持对比学习增强)
    
    Args:
        features: [num_sequences, feature_dim] 正样本特征矩阵
        method: 融合方法 ('mean', 'weighted_mean', 'clustering', 'contrastive')
        n_clusters: 聚类数量 (仅clustering方法使用)
        negative_features: [num_neg_sequences, feature_dim] 负样本特征 (用于对比增强)
    Returns:
        global_features: [n_clusters or 1, feature_dim] 全局特征
    """
    if method == 'mean':
        gf = features.mean(axis=0, keepdims=True)
        logger.info("全局特征融合: 简单平均")
        
    elif method == 'weighted_mean':
        # 这里可以根据序列长度或其他权重进行加权
        # 简化实现：等权重 (可以后续改进)
        weights = np.ones(len(features)) / len(features)
        gf = (features.T @ weights).T[None, ...]
        logger.info("全局特征融合: 加权平均")
        
    elif method == 'clustering':
        from sklearn.cluster import KMeans
        
        if len(features) < n_clusters:
            logger.warning(f"序列数量({len(features)}) < 聚类数({n_clusters})，使用简单平均")
            gf = features.mean(axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            gf = kmeans.fit(features).cluster_centers_
            logger.info(f"全局特征融合: K-means聚类 (k={n_clusters})")
    
    elif method == 'contrastive' and negative_features is not None:
        # 对比增强的特征融合
        from sklearn.cluster import KMeans
        
        # 计算正负样本中心
        pos_center = features.mean(axis=0)
        neg_center = negative_features.mean(axis=0)
        
        # 计算对比方向
        contrast_direction = pos_center - neg_center
        contrast_direction = contrast_direction / (np.linalg.norm(contrast_direction) + 1e-8)
        
        # 在对比方向上增强正样本特征
        enhanced_features = features + 0.1 * contrast_direction[None, :]
        
        # 聚类增强后的特征
        if len(enhanced_features) < n_clusters:
            gf = enhanced_features.mean(axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            gf = kmeans.fit(enhanced_features).cluster_centers_
        
        logger.info(f"全局特征融合: 对比增强聚类 (k={n_clusters})")
    
    else:
        raise ValueError(f"未知的融合方法: {method}")
    
    logger.info(f"融合后全局特征形状: {gf.shape}")
    return gf

def test_esm2_encoder():
    """测试ESM-2编码器功能"""
    logger.info("开始测试ESM-2编码器...")
    
    # 创建测试序列
    test_sequences = [
        "GLFDIVKKVVGALGSLGLVVR",  # 示例AMP序列
        "KWVKAMDGVIDMLFYKMVYK",
        "FLGALFKALAALFVSSSK"
    ]
    
    try:
        # 初始化编码器
        encoder = ESM2AuxiliaryEncoder()
        
        # 编码测试
        features = encoder.encode_sequences(test_sequences)
        logger.info(f"编码测试成功！特征形状: {features.shape}")
        
        # 特征统计
        logger.info(f"特征均值: {features.mean().item():.4f}")
        logger.info(f"特征标准差: {features.std().item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ESM-2编码器测试失败: {e}")
        return False

def test_contrastive_learning():
    """测试对比学习功能"""
    logger.info("开始测试对比学习功能...")
    
    # 创建测试序列
    positive_seqs = [
        "GLFDIVKKVVGALGSLGLVVR",  # 抗阴性菌
        "KWVKAMDGVIDMLFYKMVYK",
        "FLGALFKALAALFVSSSK"
    ]
    
    negative_seqs = [
        "GCWSTVLGGLKKFAKGGLEAIVNPK",  # 只抗阳性菌
        "GFWTTAAEGLKKFAKAGLASILNPK",
        "GLSQGVEPDIGQTYFEESRINQD"
    ]
    
    try:
        # 初始化编码器
        encoder = ESM2AuxiliaryEncoder()
        
        # 测试对比特征提取
        pos_features, neg_features = encoder.extract_contrastive_features(positive_seqs, negative_seqs)
        
        logger.info(f"正样本特征形状: {pos_features.shape}")
        logger.info(f"负样本特征形状: {neg_features.shape}")
        
        # 计算特征距离
        pos_center = pos_features.mean(axis=0)
        neg_center = neg_features.mean(axis=0)
        distance = np.linalg.norm(pos_center - neg_center)
        
        logger.info(f"正负样本中心距离: {distance:.4f}")
        
        # 测试对比增强的全局特征融合
        enhanced_gf = fuse_global_features(
            pos_features, 
            method='contrastive', 
            negative_features=neg_features,
            n_clusters=3
        )
        
        logger.info(f"对比增强全局特征形状: {enhanced_gf.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"对比学习测试失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ESM-2增强辅助编码器 (支持对比学习)")
    parser.add_argument("--test", action="store_true", help="运行基础测试")
    parser.add_argument("--test_contrastive", action="store_true", help="测试对比学习功能")
    parser.add_argument("--extract", action="store_true", help="提取辅助特征")
    parser.add_argument("--extract_contrastive", action="store_true", help="提取对比学习特征")
    parser.add_argument("--aux_data", type=str, default="data/natureAMP.txt", help="辅助数据路径")
    parser.add_argument("--output", type=str, default="esm2_auxiliary_features.npy", help="输出特征文件")
    parser.add_argument("--fusion_method", type=str, default="clustering", 
                       choices=["mean", "weighted_mean", "clustering", "contrastive"],
                       help="特征融合方法")
    
    args = parser.parse_args()
    
    if args.test:
        success = test_esm2_encoder()
        if success:
            logger.info("✅ ESM-2编码器测试通过！")
        else:
            logger.error("❌ ESM-2编码器测试失败！")
    
    if args.test_contrastive:
        success = test_contrastive_learning()
        if success:
            logger.info("✅ 对比学习功能测试通过！")
        else:
            logger.error("❌ 对比学习功能测试失败！")
    
    if args.extract:
        try:
            # 加载辅助序列
            aux_sequences = load_auxiliary_sequences(args.aux_data)
            
            # 初始化编码器
            encoder = ESM2AuxiliaryEncoder()
            
            # 提取特征
            features = encoder.extract_auxiliary_features(aux_sequences)
            
            # 融合全局特征
            global_features = fuse_global_features(features, method=args.fusion_method, n_clusters=5)
            
            # 保存特征
            np.save(args.output, global_features)
            logger.info(f"✅ 全局特征已保存到: {args.output}")
            
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
    
    if args.extract_contrastive:
        try:
            # 加载对比学习数据集
            main_seqs, positive_seqs, negative_seqs = load_contrastive_datasets()
            
            # 初始化编码器
            encoder = ESM2AuxiliaryEncoder()
            
            # 提取对比特征
            pos_features, neg_features = encoder.extract_contrastive_features(positive_seqs, negative_seqs)
            
            # 对比增强的全局特征融合
            if args.fusion_method == "contrastive":
                global_features = fuse_global_features(
                    pos_features, 
                    method='contrastive',
                    negative_features=neg_features,
                    n_clusters=5
                )
            else:
                global_features = fuse_global_features(pos_features, method=args.fusion_method, n_clusters=5)
            
            # 保存特征
            output_name = f"contrastive_{args.fusion_method}_features.npy"
            np.save(output_name, global_features)
            logger.info(f"✅ 对比学习全局特征已保存到: {output_name}")
            
            # 额外保存正负样本特征用于分析
            np.save("positive_features.npy", pos_features)
            np.save("negative_features.npy", neg_features)
            logger.info("✅ 正负样本特征已分别保存")
            
        except Exception as e:
            logger.error(f"❌ 对比学习特征提取失败: {e}")
