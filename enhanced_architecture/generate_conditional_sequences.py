#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练好的条件扩散模型生成新的肽序列，并使用混合预测器进行评估。
"""

import os
import sys
import torch
import argparse
import random
import subprocess
import logging

# 解决模块导入问题
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入我们的组件
from enhanced_architecture.conditional_esm2_extractor import ConditionalESM2FeatureExtractor
from gram_predictor.diffusion_models.conditional_d3pm import ConditionalD3PMUNet, ConditionalD3PMDiffusion, D3PMScheduler
from gram_predictor.data_loader import tokens_to_sequence

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(config, checkpoint_path, device):
    """加载训练好的条件扩散模型"""
    logger.info(f"从检查点加载模型: {checkpoint_path}")
    
    # 1. 重新创建模型架构
    unet = ConditionalD3PMUNet(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        condition_dim=config["condition_dim"],
        max_seq_len=config["max_seq_len"]
    )
    scheduler = D3PMScheduler(num_timesteps=config["num_timesteps"])
    diffusion_model = ConditionalD3PMDiffusion(unet, scheduler, device)
    
    # 2. 加载权重
    if not os.path.exists(checkpoint_path):
        logger.error(f"检查点文件未找到: {checkpoint_path}")
        raise FileNotFoundError(f"检查点文件未找到: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理DataParallel包装的模型
    model_state_dict = checkpoint['model_state_dict']
    if isinstance(diffusion_model.model, torch.nn.DataParallel):
        diffusion_model.model.module.load_state_dict(model_state_dict)
    else:
        # 如果保存时是DataParallel，但现在不是，需要移除'module.'前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        diffusion_model.model.load_state_dict(new_state_dict)
        
    logger.info("✅ 模型加载成功。")
    return diffusion_model

def get_reference_sequences(files, num_refs):
    """从文件中随机选择参考序列"""
    all_seqs = []
    for file_path in files:
        with open(file_path, 'r') as f:
            all_seqs.extend([line.strip() for line in f if line.strip()])
    
    if len(all_seqs) < num_refs:
        logger.warning(f"数据文件中的序列总数 ({len(all_seqs)}) 少于请求的参考序列数 ({num_refs})。将使用所有可用序列。")
        return all_seqs
        
    return random.sample(all_seqs, num_refs)

def save_sequences_to_fasta(sequences, output_file):
    """将生成的序列保存为FASTA格式"""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">Generated_Seq_{i+1}\n")
            f.write(f"{seq}\n")
    logger.info(f"生成的序列已保存到FASTA文件: {output_file}")

def run_predictor(fasta_file, output_file):
    """调用hybrid_predict.py脚本进行预测"""
    logger.info(f"正在调用 hybrid_predict.py 对 {fasta_file} 进行预测...")
    
    command = [
        "conda", "run", "-n", "drug", "python", "hybrid_predict.py",
        "--fasta_file", fasta_file,
        "--output_file", output_file
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("✅ 预测脚本成功执行。")
        logger.info("预测器输出:\n" + result.stdout)
        if result.stderr:
            logger.warning("预测器标准错误输出:\n" + result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 预测脚本执行失败。返回码: {e.returncode}")
        logger.error("标准输出:\n" + e.stdout)
        logger.error("标准错误:\n" + e.stderr)
    except FileNotFoundError:
        logger.error("❌ 无法执行命令。请确保 'conda' 在您的系统路径中。")

def main():
    parser = argparse.ArgumentParser(description="使用条件扩散模型生成肽序列并进行预测。")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_650M/best_model.pt",
                        help="训练好的模型检查点路径。")
    parser.add_argument("--num_sequences", type=int, default=100,
                        help="要生成的序列数量。")
    parser.add_argument("--num_references", type=int, default=5,
                        help="用于条件的参考序列数量。")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-Free Guidance的指导强度。")
    parser.add_argument("--output_fasta", type=str, default="generated_peptides.fasta",
                        help="保存生成的序列的FASTA文件。")
    parser.add_argument("--prediction_output", type=str, default="predictions/generated_peptides_predictions.txt",
                        help="保存预测结果的文件。")
    
    args = parser.parse_args()

    # --- 1. 设置环境和配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 使用与训练时匹配的服务器配置
    config = {
        "esm_model": "facebook/esm2_t33_650M_UR50D",
        "condition_dim": 512,
        "hidden_dim": 512,
        "num_layers": 8,
        "num_timesteps": 1000,
        "max_seq_len": 100,
    }

    # --- 2. 加载模型和特征提取器 ---
    try:
        diffusion_model = load_trained_model(config, args.checkpoint, device)
        feature_extractor = ConditionalESM2FeatureExtractor(
            model_name=config["esm_model"],
            condition_dim=config["condition_dim"]
        ).to(device)
    except Exception as e:
        logger.critical(f"无法加载模型或特征提取器: {e}")
        return

    # --- 3. 准备生成条件 ---
    logger.info(f"正在准备 {args.num_references} 条参考序列作为生成条件...")
    reference_files = [
        "enhanced_architecture/gram_neg_only.txt",
        "enhanced_architecture/gram_both.txt"
    ]
    ref_sequences = get_reference_sequences(reference_files, args.num_references)
    logger.info("参考序列: " + ", ".join(ref_sequences))
    
    # 提取条件特征，并取平均作为最终条件
    condition_features = feature_extractor.extract_condition_features(ref_sequences)
    final_condition = condition_features.mean(dim=0).unsqueeze(0) # [1, condition_dim]
    # 复制条件以匹配要生成的序列数量
    final_condition = final_condition.repeat(args.num_sequences, 1)

    # --- 4. 生成序列 ---
    logger.info(f"正在生成 {args.num_sequences} 条序列...")
    generated_tokens = diffusion_model.sample(
        batch_size=args.num_sequences,
        seq_len=config["max_seq_len"],
        condition_features=final_condition,
        guidance_scale=args.guidance_scale
    )
    
    generated_sequences = [tokens_to_sequence(tokens) for tokens in generated_tokens]
    logger.info("✅ 序列生成完成。")
    for i, seq in enumerate(generated_sequences[:5]):
        logger.info(f"  样本 {i+1}: {seq}")

    # --- 5. 保存到FASTA文件 ---
    save_sequences_to_fasta(generated_sequences, args.output_fasta)

    # --- 6. 调用预测器 ---
    run_predictor(args.output_fasta, args.prediction_output)

if __name__ == "__main__":
    main()