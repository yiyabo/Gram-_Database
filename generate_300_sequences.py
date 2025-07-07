#!/usr/bin/env python3
"""
生成300条抗菌肽序列
参数: T=0.8, diversity_strength=1.2
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gram_predictor.generation_service import SequenceGenerationService

def generate_sequences(num_sequences=300, temperature=0.8, diversity_strength=1.2):
    """
    生成指定数量的抗菌肽序列
    
    Args:
        num_sequences: 序列数量
        temperature: 采样温度
        diversity_strength: 多样性强度
    """
    print(f"开始生成 {num_sequences} 条序列")
    print(f"参数: T={temperature}, diversity_strength={diversity_strength}")
    
    # 初始化生成服务
    service = SequenceGenerationService()
    
    # 加载模型
    print("正在加载模型...")
    if not service.load_models():
        print("❌ 模型加载失败")
        return
    
    print("✅ 模型加载成功")
    
    # 生成序列
    start_time = time.time()
    
    result = service.generate_sequences(
        num_sequences=num_sequences,
        seq_length=30,  # 固定长度40
        sampling_method="diverse",
        temperature=temperature,
        diversity_strength=diversity_strength
    )
    
    end_time = time.time()
    
    if not result["success"]:
        print(f"❌ 生成失败: {result['error']}")
        return
    
    # 保存结果
    sequences = result["sequences"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_sequences_300_{timestamp}.fasta"
    
    with open(output_file, "w") as f:
        for seq_data in sequences:
            f.write(f">{seq_data['id']}_T{temperature}_D{diversity_strength}\n")
            f.write(f"{seq_data['sequence']}\n")
    
    print(f"✅ 生成完成！")
    print(f"序列数量: {len(sequences)}")
    print(f"生成时间: {end_time - start_time:.2f}秒")
    print(f"保存到: {output_file}")
    
    # 统计信息
    lengths = [len(seq['sequence']) for seq in sequences]
    print(f"序列长度统计: 最短={min(lengths)}, 最长={max(lengths)}, 平均={sum(lengths)/len(lengths):.1f}")
    
    return output_file

if __name__ == "__main__":
    generate_sequences()