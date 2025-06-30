#!/usr/bin/env python3
"""
序列生成服务
为Web应用提供抗菌肽序列生成功能
"""

import os
import torch
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from gram_predictor.config.model_config import get_config
from gram_predictor.esm2_auxiliary_encoder import ESM2AuxiliaryEncoder
from gram_predictor.diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet
from gram_predictor.data_loader import tokens_to_sequence

# 词汇表转换：从21词汇格式转换为22词汇格式（用于与预测服务兼容）
def convert_sequence_for_prediction(sequence: str) -> str:
    """
    将生成的序列转换为预测服务兼容的格式
    生成服务使用21词汇（PAD + 20氨基酸），预测服务使用22词汇（PAD + UNK + 20氨基酸）
    """
    # 序列本身不需要转换，只是确保格式正确
    return sequence.upper().strip()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequenceGenerationService:
    """序列生成服务类"""
    
    def __init__(self, checkpoint_path: str = None, config_name: str = "dual_4090"):
        if checkpoint_path is None:
            # 自动检测模型路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, "models", "best.pt")
        """
        初始化生成服务
        
        Args:
            checkpoint_path: 模型检查点路径
            config_name: 配置名称
        """
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型组件
        self.esm2_encoder = None
        self.diffusion_model = None
        self.config = None
        
        # 加载状态
        self.is_loaded = False
        
        logger.info(f"序列生成服务初始化完成，设备: {self.device}")
    
    def load_models(self):
        """加载生成模型"""
        if self.is_loaded:
            return True
            
        try:
            logger.info(f"加载生成模型: {self.checkpoint_path}")
            
            # 检查文件是否存在
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"模型检查点不存在: {self.checkpoint_path}")
            
            # 加载配置
            self.config = get_config(self.config_name)
            
            # 加载检查点
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # 初始化ESM-2编码器
            self.esm2_encoder = ESM2AuxiliaryEncoder(self.config.esm2)
            self.esm2_encoder.load_state_dict(checkpoint['esm2_encoder_state_dict'])
            self.esm2_encoder.to(self.device)
            self.esm2_encoder.eval()
            
            # 初始化扩散模型
            scheduler = D3PMScheduler(
                num_timesteps=self.config.diffusion.num_timesteps,
                schedule_type=self.config.diffusion.schedule_type,
                vocab_size=self.config.diffusion.vocab_size
            )
            
            unet = D3PMUNet(
                vocab_size=self.config.diffusion.vocab_size,
                max_seq_len=self.config.diffusion.max_seq_len,
                hidden_dim=self.config.diffusion.hidden_dim,
                num_layers=self.config.diffusion.num_layers,
                num_heads=self.config.diffusion.num_heads,
                dropout=self.config.diffusion.dropout
            )
            
            unet.load_state_dict(checkpoint['diffusion_model_state_dict'])
            
            self.diffusion_model = D3PMDiffusion(
                model=unet,
                scheduler=scheduler,
                device=self.device
            )
            
            self.is_loaded = True
            logger.info("✅ 生成模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_sequences(self, 
                          num_sequences: int = 5,
                          seq_length: int = 40,
                          sampling_method: str = "diverse",
                          temperature: float = 1.0,
                          reference_sequences: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        生成抗菌肽序列
        
        Args:
            num_sequences: 生成序列数量
            seq_length: 序列长度
            sampling_method: 采样方法 ("basic", "diverse", "top_k", "nucleus")
            temperature: 采样温度
            reference_sequences: 参考序列（用于条件生成）
            **kwargs: 其他参数
            
        Returns:
            包含生成结果的字典
        """
        if not self.is_loaded:
            if not self.load_models():
                return {
                    "success": False,
                    "error": "模型加载失败",
                    "sequences": []
                }
        
        try:
            logger.info(f"开始生成 {num_sequences} 条序列，方法: {sampling_method}")
            
            # 处理参考序列（条件生成）
            esm_features = None
            if reference_sequences and len(reference_sequences) > 0:
                logger.info(f"使用 {len(reference_sequences)} 条参考序列进行条件生成")
                with torch.no_grad():
                    esm_features = self.esm2_encoder.encode_sequences(reference_sequences)
                    esm_features = esm_features.mean(dim=0, keepdim=True)  # 使用平均特征
            
            # 根据采样方法生成序列
            with torch.no_grad():
                if sampling_method == "basic":
                    generated_tokens = self.diffusion_model.sample(
                        batch_size=num_sequences,
                        seq_len=seq_length,
                        esm_features=esm_features,
                        temperature=temperature
                    )
                
                elif sampling_method == "diverse":
                    diversity_strength = kwargs.get("diversity_strength", 0.3)
                    generated_tokens = self.diffusion_model.diverse_sample(
                        batch_size=num_sequences,
                        seq_len=seq_length,
                        esm_features=esm_features,
                        diversity_strength=diversity_strength,
                        temperature=temperature
                    )
                
                elif sampling_method == "top_k":
                    k = kwargs.get("k", 10)
                    generated_tokens = self.diffusion_model.top_k_sample(
                        batch_size=num_sequences,
                        seq_len=seq_length,
                        esm_features=esm_features,
                        k=k,
                        temperature=temperature
                    )
                
                elif sampling_method == "nucleus":
                    p = kwargs.get("p", 0.9)
                    generated_tokens = self.diffusion_model.nucleus_sample(
                        batch_size=num_sequences,
                        seq_len=seq_length,
                        esm_features=esm_features,
                        p=p,
                        temperature=temperature
                    )
                
                else:
                    raise ValueError(f"未知的采样方法: {sampling_method}")
            
            # 转换为氨基酸序列
            sequences = []
            for i, tokens in enumerate(generated_tokens):
                seq = tokens_to_sequence(tokens.cpu().numpy())
                # 转换为预测服务兼容格式
                converted_seq = convert_sequence_for_prediction(seq)
                sequences.append({
                    "id": f"Generated_Seq_{i+1}",
                    "sequence": converted_seq,
                    "length": len(converted_seq),
                    "method": sampling_method
                })
            
            logger.info(f"✅ 成功生成 {len(sequences)} 条序列")
            
            return {
                "success": True,
                "sequences": sequences,
                "parameters": {
                    "num_sequences": num_sequences,
                    "seq_length": seq_length,
                    "sampling_method": sampling_method,
                    "temperature": temperature,
                    "reference_count": len(reference_sequences) if reference_sequences else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 序列生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "sequences": []
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "device": str(self.device),
            "config_name": self.config_name,
            "vocab_size": self.config.diffusion.vocab_size,
            "max_seq_len": self.config.diffusion.max_seq_len,
            "model_params": {
                "hidden_dim": self.config.diffusion.hidden_dim,
                "num_layers": self.config.diffusion.num_layers,
                "num_heads": self.config.diffusion.num_heads
            }
        }

# 全局生成服务实例
generation_service = None

def get_generation_service() -> SequenceGenerationService:
    """获取全局生成服务实例"""
    global generation_service
    if generation_service is None:
        generation_service = SequenceGenerationService()
    return generation_service