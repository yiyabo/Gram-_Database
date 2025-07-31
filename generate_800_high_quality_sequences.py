#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高质量抗菌肽序列生成器 - 生成800条序列
使用条件D3PM模型 + ESM-2特征引导 + Classifier-Free Guidance
优化参数设置，确保高质量输出
"""

import os
import sys
import torch
import random
import argparse
import logging
import subprocess
from datetime import datetime

# 解决模块导入问题
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要组件
try:
    from enhanced_architecture.conditional_esm2_extractor import ConditionalESM2FeatureExtractor
    from gram_predictor.diffusion_models.conditional_d3pm import ConditionalD3PMUNet, ConditionalD3PMDiffusion, D3PMScheduler
    from gram_predictor.data_loader import tokens_to_sequence
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'generation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HighQualitySequenceGenerator:
    """高质量序列生成器"""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 使用设备: {self.device}")
        
        # 自动检测检查点路径
        if checkpoint_path is None:
            checkpoint_path = self.find_best_checkpoint()
        
        self.checkpoint_path = checkpoint_path
        self.diffusion_model = None
        self.feature_extractor = None
        
        # 优化的模型配置 - 基于您成功的设置
        self.model_config = {
            "esm_model": "facebook/esm2_t33_650M_UR50D",
            "condition_dim": 512,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_timesteps": 1000,
            "max_seq_len": 100,
        }
        
        # 高质量生成参数
        self.generation_params = {
            "guidance_scale": 5.0,  # 适中的引导强度，避免过拟合
            "temperature": 0.8,     # 稍低的温度，提高质量
            "min_len": 20,
            "max_len": 50,
            "num_references": 10,   # 更多参考序列
        }
        
    def find_best_checkpoint(self):
        """自动查找最佳检查点"""
        possible_paths = [
            "enhanced_architecture/output/checkpoints/best.pt",
            "enhanced_architecture/output/checkpoints/latest.pt",
            "gram_predictor/models/best.pt",
            "checkpoints/best.pt",
            "output/checkpoints/best.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"✅ 找到检查点: {path}")
                return path
        
        # 搜索所有.pt文件
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".pt") and ("best" in file or "checkpoint" in file):
                    full_path = os.path.join(root, file)
                    logger.info(f"🔍 发现检查点: {full_path}")
                    return full_path
        
        raise FileNotFoundError("❌ 未找到任何模型检查点文件！请确保已训练模型。")
    
    def load_models(self):
        """加载模型组件"""
        logger.info(f"📂 加载检查点: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        logger.info(f"📊 检查点信息: Epoch {checkpoint.get('epoch', 'Unknown')}, Loss {checkpoint.get('best_val_loss', 'Unknown')}")
        
        # 1. 初始化ESM-2特征提取器
        self.feature_extractor = ConditionalESM2FeatureExtractor(
            model_name=self.model_config["esm_model"],
            condition_dim=self.model_config["condition_dim"]
        ).to(self.device)
        
        # 2. 初始化条件扩散模型
        unet = ConditionalD3PMUNet(
            hidden_dim=self.model_config["hidden_dim"],
            num_layers=self.model_config["num_layers"],
            condition_dim=self.model_config["condition_dim"],
            max_seq_len=self.model_config["max_seq_len"]
        )
        
        scheduler = D3PMScheduler(num_timesteps=self.model_config["num_timesteps"])
        self.diffusion_model = ConditionalD3PMDiffusion(unet, scheduler, self.device)
        
        # 3. 加载权重
        try:
            # 处理可能的DataParallel包装
            model_state_dict = checkpoint['model_state_dict']
            if list(model_state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:]  # 移除'module.'前缀
                    new_state_dict[name] = v
                model_state_dict = new_state_dict
            
            self.diffusion_model.model.load_state_dict(model_state_dict)
            logger.info("✅ 扩散模型权重加载成功")
            
        except Exception as e:
            logger.error(f"❌ 模型权重加载失败: {e}")
            raise
        
        logger.info("🎯 所有模型组件加载完成")
    
    def get_reference_sequences(self, num_refs: int):
        """获取高质量参考序列"""
        reference_files = [
            "enhanced_architecture/gram_neg_only.txt",
            "enhanced_architecture/gram_both.txt"
        ]
        
        all_seqs = []
        for file_path in reference_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    seqs = [line.strip() for line in f if line.strip() and len(line.strip()) >= 15]
                    all_seqs.extend(seqs)
        
        if not all_seqs:
            logger.warning("⚠️ 未找到参考序列文件，使用默认高质量序列")
            # 提供一些高质量的默认参考序列
            all_seqs = [
                "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
                "GIGKFLHSAKKFGKAFVGEIMNS",
                "KWKLFKKIGAVLKVLTTGLPALIS",
                "FLGALFKALKAA",
                "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
                "GIGKFLHSAKKFGKAFVGEIMNS",
                "KWKLFKKIGAVLKVLTTGLPALIS",
                "FLGALFKALKAA"
            ]
        
        if len(all_seqs) < num_refs:
            # 如果序列不够，重复使用
            all_seqs = all_seqs * ((num_refs // len(all_seqs)) + 1)
        
        selected = random.sample(all_seqs, min(num_refs, len(all_seqs)))
        logger.info(f"📋 选择了 {len(selected)} 条参考序列")
        return selected
    
    def generate_sequences(self, num_sequences: int = 800):
        """生成高质量序列"""
        logger.info(f"🎯 开始生成 {num_sequences} 条高质量序列...")
        
        # 分批生成以避免内存问题
        batch_size = 50  # 每批生成50条
        all_sequences = []
        
        # 获取参考序列
        ref_sequences = self.get_reference_sequences(self.generation_params["num_references"])
        logger.info("参考序列示例: " + ", ".join(ref_sequences[:3]) + "...")
        
        # 提取条件特征
        with torch.no_grad():
            condition_features = self.feature_extractor.extract_condition_features(ref_sequences)
            final_condition = condition_features.mean(dim=0).unsqueeze(0)  # [1, condition_dim]
        
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_sequences - batch_idx * batch_size)
            logger.info(f"🔄 生成批次 {batch_idx + 1}/{num_batches} ({current_batch_size} 条序列)")
            
            # 复制条件以匹配批次大小
            batch_condition = final_condition.repeat(current_batch_size, 1)
            
            # 随机选择序列长度
            target_lengths = [random.randint(self.generation_params["min_len"], 
                                           self.generation_params["max_len"]) 
                            for _ in range(current_batch_size)]
            max_length = max(target_lengths)
            
            # 生成序列
            with torch.no_grad():
                generated_tokens = self.diffusion_model.sample(
                    batch_size=current_batch_size,
                    seq_len=max_length,
                    condition_features=batch_condition,
                    guidance_scale=self.generation_params["guidance_scale"],
                    temperature=self.generation_params["temperature"]
                )
            
            # 转换为氨基酸序列并截断到目标长度
            for i, (tokens, target_len) in enumerate(zip(generated_tokens, target_lengths)):
                full_sequence = tokens_to_sequence(tokens.cpu().numpy())
                truncated_sequence = full_sequence[:target_len]
                
                # 基本质量检查
                if len(truncated_sequence) >= 10 and truncated_sequence.count('X') == 0:
                    all_sequences.append({
                        'id': f'HighQuality_Seq_{len(all_sequences) + 1:04d}',
                        'sequence': truncated_sequence,
                        'length': len(truncated_sequence),
                        'batch': batch_idx + 1
                    })
        
        logger.info(f"✅ 成功生成 {len(all_sequences)} 条序列")
        return all_sequences
    
    def save_sequences(self, sequences, output_prefix="high_quality_800_sequences"):
        """保存生成的序列"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为FASTA格式
        fasta_file = f"{output_prefix}_{timestamp}.fasta"
        with open(fasta_file, 'w') as f:
            for seq_data in sequences:
                f.write(f">{seq_data['id']} | Length={seq_data['length']} | Batch={seq_data['batch']}\n")
                f.write(f"{seq_data['sequence']}\n")
        
        # 保存为CSV格式
        csv_file = f"{output_prefix}_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("ID,Sequence,Length,Batch\n")
            for seq_data in sequences:
                f.write(f"{seq_data['id']},{seq_data['sequence']},{seq_data['length']},{seq_data['batch']}\n")
        
        logger.info(f"💾 序列已保存:")
        logger.info(f"   FASTA: {fasta_file}")
        logger.info(f"   CSV: {csv_file}")
        
        return fasta_file, csv_file
    
    def run_prediction(self, fasta_file):
        """运行预测分析"""
        logger.info("🔍 开始预测分析...")
        
        prediction_output = fasta_file.replace('.fasta', '_predictions.txt')
        
        # 尝试调用预测脚本
        try:
            command = [
                sys.executable, "hybrid_predict.py",
                "--fasta_file", fasta_file,
                "--output_file", prediction_output
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
            logger.info("✅ 预测完成")
            logger.info(f"📊 预测结果保存到: {prediction_output}")
            
            if result.stdout:
                logger.info("预测输出摘要:")
                for line in result.stdout.split('\n')[-10:]:  # 显示最后10行
                    if line.strip():
                        logger.info(f"   {line}")
                        
        except subprocess.TimeoutExpired:
            logger.error("❌ 预测超时（30分钟）")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 预测失败: {e}")
            if e.stdout:
                logger.error(f"标准输出: {e.stdout}")
            if e.stderr:
                logger.error(f"标准错误: {e.stderr}")
        except FileNotFoundError:
            logger.warning("⚠️ 未找到 hybrid_predict.py，跳过预测步骤")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高质量抗菌肽序列生成器")
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径（可选，自动检测）")
    parser.add_argument("--num_sequences", type=int, default=800, help="生成序列数量")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="引导强度")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--skip_prediction", action="store_true", help="跳过预测步骤")
    
    args = parser.parse_args()
    
    try:
        # 初始化生成器
        generator = HighQualitySequenceGenerator(args.checkpoint)
        
        # 更新参数
        if args.guidance_scale != 5.0:
            generator.generation_params["guidance_scale"] = args.guidance_scale
        if args.temperature != 0.8:
            generator.generation_params["temperature"] = args.temperature
        
        # 加载模型
        generator.load_models()
        
        # 生成序列
        sequences = generator.generate_sequences(args.num_sequences)
        
        if not sequences:
            logger.error("❌ 未生成任何序列")
            return
        
        # 保存序列
        fasta_file, csv_file = generator.save_sequences(sequences)
        
        # 运行预测（可选）
        if not args.skip_prediction:
            generator.run_prediction(fasta_file)
        
        # 生成统计报告
        lengths = [seq['length'] for seq in sequences]
        logger.info("📈 生成统计:")
        logger.info(f"   总序列数: {len(sequences)}")
        logger.info(f"   平均长度: {sum(lengths)/len(lengths):.1f}")
        logger.info(f"   长度范围: {min(lengths)}-{max(lengths)}")
        logger.info(f"   引导强度: {generator.generation_params['guidance_scale']}")
        logger.info(f"   采样温度: {generator.generation_params['temperature']}")
        
        logger.info("🎉 高质量序列生成完成！")
        
    except Exception as e:
        logger.error(f"❌ 生成过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()