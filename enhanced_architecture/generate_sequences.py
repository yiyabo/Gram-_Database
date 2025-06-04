#!/usr/bin/env python3
"""
训练完成后的序列生成脚本
展示如何使用训练好的模型生成新的抗菌肽序列
"""

import torch
import numpy as np
import argparse
from typing import List, Optional
import os

# 导入项目模块
from config.model_config import get_config
from esm2_auxiliary_encoder import ESM2AuxiliaryEncoder
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet
from data_loader import tokens_to_sequence

class SequenceGenerator:
    """序列生成器"""
    
    def __init__(self, checkpoint_path: str, config_name: str = "dual_4090"):
        """
        初始化生成器
        
        Args:
            checkpoint_path: 训练好的模型检查点路径
            config_name: 配置名称
        """
        self.config = get_config(config_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.load_models(checkpoint_path)
        
        print(f"✅ 序列生成器初始化完成")
        print(f"📱 设备: {self.device}")
        print(f"🧬 词汇表大小: {self.config.diffusion.vocab_size}")
    
    def load_models(self, checkpoint_path: str):
        """加载训练好的模型"""
        print(f"📂 加载检查点: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 加载检查点 - 修复PyTorch 2.6的weights_only问题
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
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
        
        print(f"✅ 模型加载完成")
        print(f"📊 训练轮数: {checkpoint['epoch']}")
        print(f"📈 最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    
    def generate_basic(self, num_sequences: int = 10, seq_length: int = 50, 
                      temperature: float = 1.0) -> List[str]:
        """
        基础序列生成
        
        Args:
            num_sequences: 生成序列数量
            seq_length: 序列长度
            temperature: 采样温度 (1.0=标准, >1.0更随机, <1.0更确定)
        
        Returns:
            生成的氨基酸序列列表
        """
        print(f"🧬 生成 {num_sequences} 条长度为 {seq_length} 的序列...")
        
        with torch.no_grad():
            generated_tokens = self.diffusion_model.sample(
                batch_size=num_sequences,
                seq_len=seq_length,
                temperature=temperature
            )
        
        # 转换为氨基酸序列
        sequences = []
        for tokens in generated_tokens:
            seq = tokens_to_sequence(tokens.cpu().numpy())
            sequences.append(seq)
        
        return sequences
    
    def generate_with_reference(self, reference_sequences: List[str], 
                               num_sequences: int = 10, seq_length: int = 50,
                               temperature: float = 0.8) -> List[str]:
        """
        基于参考序列的条件生成
        
        Args:
            reference_sequences: 参考序列列表
            num_sequences: 生成序列数量
            seq_length: 序列长度
            temperature: 采样温度
        
        Returns:
            生成的氨基酸序列列表
        """
        print(f"🎯 基于 {len(reference_sequences)} 条参考序列生成...")
        
        # 提取ESM-2特征
        with torch.no_grad():
            esm_features = self.esm2_encoder.encode_sequences(reference_sequences)
            # 使用平均特征作为条件
            avg_features = esm_features.mean(dim=0, keepdim=True)
            
            generated_tokens = self.diffusion_model.sample(
                batch_size=num_sequences,
                seq_len=seq_length,
                esm_features=avg_features,
                temperature=temperature
            )
        
        # 转换为氨基酸序列
        sequences = []
        for tokens in generated_tokens:
            seq = tokens_to_sequence(tokens.cpu().numpy())
            sequences.append(seq)
        
        return sequences
    
    def generate_diverse(self, num_sequences: int = 10, seq_length: int = 50,
                        diversity_strength: float = 0.3, temperature: float = 1.0) -> List[str]:
        """
        多样性感知生成
        
        Args:
            num_sequences: 生成序列数量
            seq_length: 序列长度
            diversity_strength: 多样性强度 (0-1)
            temperature: 采样温度
        
        Returns:
            生成的氨基酸序列列表
        """
        print(f"🌈 多样性生成 {num_sequences} 条序列...")
        
        with torch.no_grad():
            generated_tokens = self.diffusion_model.diverse_sample(
                batch_size=num_sequences,
                seq_len=seq_length,
                diversity_strength=diversity_strength,
                temperature=temperature
            )
        
        # 转换为氨基酸序列
        sequences = []
        for tokens in generated_tokens:
            seq = tokens_to_sequence(tokens.cpu().numpy())
            sequences.append(seq)
        
        return sequences
    
    def generate_high_quality(self, num_sequences: int = 10, seq_length: int = 50,
                             method: str = "top_k", **kwargs) -> List[str]:
        """
        高质量序列生成
        
        Args:
            num_sequences: 生成序列数量
            seq_length: 序列长度
            method: 采样方法 ("top_k", "nucleus")
            **kwargs: 采样参数
        
        Returns:
            生成的氨基酸序列列表
        """
        print(f"⭐ 使用 {method} 方法生成高质量序列...")
        
        with torch.no_grad():
            if method == "top_k":
                k = kwargs.get("k", 10)
                temperature = kwargs.get("temperature", 0.7)
                generated_tokens = self.diffusion_model.top_k_sample(
                    batch_size=num_sequences,
                    seq_len=seq_length,
                    k=k,
                    temperature=temperature
                )
            elif method == "nucleus":
                p = kwargs.get("p", 0.9)
                temperature = kwargs.get("temperature", 0.8)
                generated_tokens = self.diffusion_model.nucleus_sample(
                    batch_size=num_sequences,
                    seq_len=seq_length,
                    p=p,
                    temperature=temperature
                )
            else:
                raise ValueError(f"未知的采样方法: {method}")
        
        # 转换为氨基酸序列
        sequences = []
        for tokens in generated_tokens:
            seq = tokens_to_sequence(tokens.cpu().numpy())
            sequences.append(seq)
        
        return sequences
    
    def batch_generate(self, total_sequences: int = 100, batch_size: int = 20,
                      seq_length: int = 50, output_file: str = "generated_sequences.txt") -> List[str]:
        """
        批量生成序列
        
        Args:
            total_sequences: 总序列数量
            batch_size: 批次大小
            seq_length: 序列长度
            output_file: 输出文件路径
        
        Returns:
            生成的氨基酸序列列表
        """
        print(f"🔄 批量生成 {total_sequences} 条序列...")
        
        all_sequences = []
        
        for i in range(0, total_sequences, batch_size):
            current_batch = min(batch_size, total_sequences - i)
            
            # 使用不同的采样策略增加多样性
            if i % 3 == 0:
                sequences = self.generate_basic(current_batch, seq_length)
            elif i % 3 == 1:
                sequences = self.generate_high_quality(current_batch, seq_length, "top_k", k=15)
            else:
                sequences = self.generate_diverse(current_batch, seq_length)
            
            all_sequences.extend(sequences)
            print(f"  已生成: {len(all_sequences)}/{total_sequences}")
        
        # 保存到文件
        with open(output_file, 'w') as f:
            for i, seq in enumerate(all_sequences, 1):
                f.write(f">Generated_Sequence_{i}\n{seq}\n")
        
        print(f"💾 序列已保存到: {output_file}")
        return all_sequences

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="抗菌肽序列生成器")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--config", type=str, default="dual_4090", help="配置名称")
    parser.add_argument("--mode", type=str, default="basic", 
                       choices=["basic", "reference", "diverse", "high_quality", "batch"],
                       help="生成模式")
    parser.add_argument("--num_sequences", type=int, default=10, help="生成序列数量")
    parser.add_argument("--seq_length", type=int, default=50, help="序列长度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--output", type=str, default="generated_sequences.txt", help="输出文件")
    parser.add_argument("--reference", type=str, nargs="+", help="参考序列")
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = SequenceGenerator(args.checkpoint, args.config)
    
    # 根据模式生成序列
    if args.mode == "basic":
        sequences = generator.generate_basic(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length,
            temperature=args.temperature
        )
    
    elif args.mode == "reference":
        if not args.reference:
            # 使用默认参考序列
            reference_seqs = [
                "GLFDIVKKVVGALGSLGLVVR",  # 已知抗阴性菌肽
                "KWVKAMDGVIDMLFYKMVYK",
                "FLGALFKALAALFVSSSK"
            ]
        else:
            reference_seqs = args.reference
        
        sequences = generator.generate_with_reference(
            reference_sequences=reference_seqs,
            num_sequences=args.num_sequences,
            seq_length=args.seq_length,
            temperature=args.temperature
        )
    
    elif args.mode == "diverse":
        sequences = generator.generate_diverse(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length,
            temperature=args.temperature
        )
    
    elif args.mode == "high_quality":
        sequences = generator.generate_high_quality(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length,
            method="top_k",
            k=10,
            temperature=0.7
        )
    
    elif args.mode == "batch":
        sequences = generator.batch_generate(
            total_sequences=args.num_sequences,
            seq_length=args.seq_length,
            output_file=args.output
        )
        return
    
    # 显示生成的序列
    print(f"\n🧬 生成的序列:")
    for i, seq in enumerate(sequences, 1):
        print(f"{i:2d}: {seq}")
    
    # 保存序列
    with open(args.output, 'w') as f:
        for i, seq in enumerate(sequences, 1):
            f.write(f">Generated_Sequence_{i}\n{seq}\n")
    
    print(f"\n💾 序列已保存到: {args.output}")

if __name__ == "__main__":
    main()