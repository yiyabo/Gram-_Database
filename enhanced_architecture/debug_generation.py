#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试扩散模型生成过程
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.model_config import get_config
from main_trainer import EnhancedAMPTrainer
from data_loader import VOCAB_TO_AA

def debug_diffusion_generation():
    """调试扩散模型生成过程"""
    
    print("🔍 开始调试扩散模型生成过程...")
    
    # 加载配置和模型
    config = get_config("production")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载训练好的模型
    checkpoint_path = Path(config.training.output_dir) / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return
    
    # 初始化训练器和模型
    trainer = EnhancedAMPTrainer(config_name="production")
    trainer.initialize_models()
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainer.esm2_encoder.load_state_dict(checkpoint['esm2_encoder_state_dict'])
    trainer.diffusion_model.model.load_state_dict(checkpoint['diffusion_model_state_dict'])
    
    # 设置为评估模式
    trainer.diffusion_model.model.eval()
    
    print("✅ 模型加载完成")
    
    # 调试生成过程
    batch_size = 2
    seq_len = 10
    
    with torch.no_grad():
        print(f"\n🎯 开始生成调试 (batch_size={batch_size}, seq_len={seq_len})")
        
        # 从随机噪声开始
        x = torch.randint(0, trainer.scheduler.vocab_size, (batch_size, seq_len), device=device)
        print(f"初始随机噪声: {x}")
        print(f"初始噪声范围: {x.min().item()} - {x.max().item()}")
        
        # 简化的逆向扩散过程 - 只测试几个步骤
        num_inference_steps = 5
        timesteps = torch.linspace(trainer.scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, device=device)
        
        print(f"\n📊 时间步序列: {timesteps}")
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            print(f"\n--- 步骤 {i+1}/{len(timesteps)}, 时间步 t={t.item()} ---")
            print(f"当前 x: {x}")
            
            # 获取模型预测的logits
            predicted_logits = trainer.diffusion_model.model(x, t_batch)
            print(f"预测logits形状: {predicted_logits.shape}")
            print(f"logits数值范围: {predicted_logits.min().item():.4f} - {predicted_logits.max().item():.4f}")
            
            # 检查每个位置的logits分布
            for pos in range(min(3, seq_len)):  # 只检查前3个位置
                pos_logits = predicted_logits[0, pos, :]  # 第一个样本的第pos个位置
                pos_probs = F.softmax(pos_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(pos_probs, k=5)
                
                print(f"  位置{pos} - Top5概率:")
                for j, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
                    aa = VOCAB_TO_AA[idx.item()]
                    print(f"    {j+1}. {aa} (token {idx.item()}): {prob.item():.4f}")
            
            # 检查是否所有位置都倾向于PAD
            argmax_tokens = torch.argmax(predicted_logits, dim=-1)
            print(f"argmax tokens: {argmax_tokens}")
            
            # 计算PAD token的平均概率
            pad_probs = F.softmax(predicted_logits, dim=-1)[:, :, 0]  # PAD是token 0
            avg_pad_prob = pad_probs.mean().item()
            print(f"PAD token平均概率: {avg_pad_prob:.4f}")
            
            # 更新x用于下一步
            if i < len(timesteps) - 1:
                # 使用多项式采样
                probs = F.softmax(predicted_logits, dim=-1)
                x = torch.multinomial(probs.view(-1, trainer.scheduler.vocab_size), 
                                    num_samples=1).view(batch_size, seq_len)
            else:
                # 最后一步使用argmax
                x = torch.argmax(predicted_logits, dim=-1)
            
            print(f"更新后的 x: {x}")
        
        print(f"\n🏁 最终生成结果:")
        print(f"最终tokens: {x}")
        
        # 转换为氨基酸序列
        from data_loader import tokens_to_sequence
        for i, seq_tokens in enumerate(x):
            seq = tokens_to_sequence(seq_tokens.cpu().numpy())
            print(f"序列 {i+1}: '{seq}' (长度: {len(seq)})")
        
    print("\n✅ 调试完成！")

def check_training_data_distribution():
    """检查训练数据的token分布"""
    print("\n📈 检查训练数据token分布...")
    
    from data_loader import AntimicrobialPeptideDataset
    
    # 加载训练数据
    dataset = AntimicrobialPeptideDataset(
        sequences_file="main_training_sequences.txt",
        max_length=50
    )
    
    # 统计token分布
    token_counts = torch.zeros(21)  # 21个token
    total_tokens = 0
    
    for i in range(min(100, len(dataset))):  # 只检查前100个样本
        sample = dataset[i]
        tokens = sample['input_ids']
        
        for token in tokens:
            if token.item() < 21:  # 确保在词汇表范围内
                token_counts[token.item()] += 1
                total_tokens += 1
    
    # 打印分布
    print("Token分布:")
    for token_id in range(21):
        count = token_counts[token_id].item()
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        aa = VOCAB_TO_AA[token_id]
        print(f"  {aa} (token {token_id}): {count} ({percentage:.2f}%)")
    
    print(f"总token数: {total_tokens}")

if __name__ == "__main__":
    print("🧪 扩散模型生成调试工具")
    print("=" * 50)
    
    # 检查训练数据分布
    check_training_data_distribution()
    
    print("\n" + "=" * 50)
    
    # 调试生成过程
    debug_diffusion_generation()
