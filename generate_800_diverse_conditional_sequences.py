#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高质量抗菌肽序列生成器 - 800条序列
结合条件D3PM + ESM-2引导 + 多样性采样逻辑
"""

import os
import sys
import torch
import random
import argparse
import logging
import subprocess
from datetime import datetime
import torch.nn.functional as F

# 解决模块导入问题
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要组件
try:
    from enhanced_architecture.conditional_esm2_extractor import ConditionalESM2FeatureExtractor
    from gram_predictor.diffusion_models.conditional_d3pm import ConditionalD3PMUNet, ConditionalD3PMDiffusion, D3PMScheduler
    from gram_predictor.data_loader import tokens_to_sequence, AMINO_ACID_VOCAB
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'diverse_generation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiverseConditionalD3PMDiffusion(ConditionalD3PMDiffusion):
    """扩展条件D3PM，添加多样性采样功能"""
    
    @torch.no_grad()
    def diverse_sample(self,
                      batch_size: int,
                      seq_len: int,
                      condition_features: torch.Tensor = None,
                      guidance_scale: float = 5.0,
                      temperature: float = 1.0,
                      diversity_strength: float = 0.3) -> torch.Tensor:
        """
        多样性感知的条件采样：结合CFG和多样性控制
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            condition_features: 条件特征 [batch_size, condition_dim]
            guidance_scale: 指导强度
            temperature: 采样温度
            diversity_strength: 多样性强度 (0-1)
        """
        self.model.eval()
        
        # 设置目标氨基酸分布（基于训练数据）
        target_distribution = {
            'K': 0.1136, 'G': 0.1045, 'L': 0.0897, 'R': 0.0888, 'A': 0.0719,
            'I': 0.0612, 'V': 0.0577, 'P': 0.0569, 'S': 0.0500, 'C': 0.0459,
            'F': 0.0436, 'T': 0.0362, 'N': 0.0320, 'Q': 0.0248, 'D': 0.0247,
            'E': 0.0238, 'W': 0.0216, 'Y': 0.0209, 'H': 0.0198, 'M': 0.0122
        }
        
        # 转换为token分布
        target_token_probs = torch.zeros(self.scheduler.vocab_size, device=self.device)
        for aa, prob in target_distribution.items():
            if aa in AMINO_ACID_VOCAB:
                token_id = AMINO_ACID_VOCAB[aa]
                target_token_probs[token_id] = prob
        target_token_probs = target_token_probs / target_token_probs.sum()
        
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
            guided_logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            
            # 屏蔽PAD token的概率，避免生成PAD
            guided_logits[:, :, 0] = float('-inf')
            
            # 应用多样性调整
            if diversity_strength > 0:
                # 计算当前序列的氨基酸分布
                diversity_adjustment = torch.zeros_like(guided_logits)
                for b in range(batch_size):
                    current_counts = torch.bincount(x[b], minlength=self.scheduler.vocab_size).float()
                    current_dist = current_counts / (current_counts.sum() + 1e-8)
                    
                    for pos in range(seq_len):
                        # 惩罚过度出现的氨基酸
                        overpresented = current_dist > target_token_probs * 1.5
                        diversity_adjustment[b, pos, overpresented] = -diversity_strength * 3
                        
                        # 奖励不足的氨基酸
                        underpresented = current_dist < target_token_probs * 0.5
                        diversity_adjustment[b, pos, underpresented] = diversity_strength * 2
                
                guided_logits = guided_logits + diversity_adjustment
            
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

class DiverseConditionalSequenceGenerator:
    """多样性条件序列生成器"""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 使用设备: {self.device}")
        
        # 自动检测检查点路径
        if checkpoint_path is None:
            checkpoint_path = self.find_best_checkpoint()
        
        self.checkpoint_path = checkpoint_path
        self.diffusion_model = None
        self.feature_extractor = None
        
        # 模型配置
        self.model_config = {
            "esm_model": "facebook/esm2_t33_650M_UR50D",
            "condition_dim": 512,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_timesteps": 1000,
            "max_seq_len": 100,
        }
        
        # 生成参数 - 您要求的diverse采样参数
        self.generation_params = {
            "guidance_scale": 5.0,        # CFG引导强度
            "temperature": 0.8,           # 采样温度
            "diversity_strength": 1.2,    # 多样性强度 (与generate_300_sequences.py一致)
            "min_len": 20,
            "max_len": 50,
            "num_references": 10,
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
        
        raise FileNotFoundError("❌ 未找到任何模型检查点文件！")
    
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
        
        # 2. 初始化扩展的条件扩散模型（带多样性采样）
        unet = ConditionalD3PMUNet(
            hidden_dim=self.model_config["hidden_dim"],
            num_layers=self.model_config["num_layers"],
            condition_dim=self.model_config["condition_dim"],
            max_seq_len=self.model_config["max_seq_len"]
        )
        
        scheduler = D3PMScheduler(num_timesteps=self.model_config["num_timesteps"])
        self.diffusion_model = DiverseConditionalD3PMDiffusion(unet, scheduler, self.device)
        
        # 3. 加载权重 - 智能检测键名
        try:
            # 检测可能的键名
            possible_keys = ['model_state_dict', 'diffusion_model_state_dict', 'unet_state_dict', 'state_dict']
            model_state_dict = None
            
            logger.info("检查点包含的键:")
            for key in checkpoint.keys():
                logger.info(f"  {key}")
            
            # 尝试找到正确的模型状态字典
            for key in possible_keys:
                if key in checkpoint:
                    model_state_dict = checkpoint[key]
                    logger.info(f"✅ 使用键名: {key}")
                    break
            
            if model_state_dict is None:
                # 如果没有找到标准键名，尝试直接使用检查点
                if hasattr(checkpoint, 'state_dict'):
                    model_state_dict = checkpoint.state_dict()
                    logger.info("✅ 使用 checkpoint.state_dict()")
                else:
                    raise KeyError(f"未找到模型状态字典。可用键: {list(checkpoint.keys())}")
            
            # 处理DataParallel包装
            if list(model_state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[7:]  # 移除'module.'前缀
                    new_state_dict[name] = v
                model_state_dict = new_state_dict
                logger.info("✅ 移除了DataParallel前缀")
            
            self.diffusion_model.model.load_state_dict(model_state_dict)
            logger.info("✅ 扩散模型权重加载成功")
            
        except Exception as e:
            logger.error(f"❌ 模型权重加载失败: {e}")
            logger.error(f"检查点文件结构: {list(checkpoint.keys())}")
            raise
        
        logger.info("🎯 所有模型组件加载完成")
    
    def get_reference_sequences(self, num_refs: int):
        """获取参考序列"""
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
            logger.warning("⚠️ 未找到参考序列文件，使用默认序列")
            all_seqs = [
                "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
                "GIGKFLHSAKKFGKAFVGEIMNS",
                "KWKLFKKIGAVLKVLTTGLPALIS",
                "FLGALFKALKAA"
            ] * 3  # 重复以确保有足够的序列
        
        if len(all_seqs) < num_refs:
            all_seqs = all_seqs * ((num_refs // len(all_seqs)) + 1)
        
        selected = random.sample(all_seqs, min(num_refs, len(all_seqs)))
        logger.info(f"📋 选择了 {len(selected)} 条参考序列")
        return selected
    
    def generate_sequences(self, num_sequences: int = 800):
        """生成序列 - 使用多样性条件采样"""
        logger.info(f"🎯 开始生成 {num_sequences} 条序列 (多样性条件采样)")
        logger.info(f"📊 参数: guidance_scale={self.generation_params['guidance_scale']}, "
                   f"temperature={self.generation_params['temperature']}, "
                   f"diversity_strength={self.generation_params['diversity_strength']}")
        
        # 分批生成
        batch_size = 50
        all_sequences = []
        
        # 获取参考序列和条件特征
        ref_sequences = self.get_reference_sequences(self.generation_params["num_references"])
        logger.info("参考序列示例: " + ", ".join(ref_sequences[:3]) + "...")
        
        with torch.no_grad():
            condition_features = self.feature_extractor.extract_condition_features(ref_sequences)
            final_condition = condition_features.mean(dim=0).unsqueeze(0)
        
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_sequences - batch_idx * batch_size)
            logger.info(f"🔄 生成批次 {batch_idx + 1}/{num_batches} ({current_batch_size} 条序列)")
            
            batch_condition = final_condition.repeat(current_batch_size, 1)
            
            # 随机序列长度
            target_lengths = [random.randint(self.generation_params["min_len"], 
                                           self.generation_params["max_len"]) 
                            for _ in range(current_batch_size)]
            max_length = max(target_lengths)
            
            # 使用多样性条件采样
            with torch.no_grad():
                generated_tokens = self.diffusion_model.diverse_sample(
                    batch_size=current_batch_size,
                    seq_len=max_length,
                    condition_features=batch_condition,
                    guidance_scale=self.generation_params["guidance_scale"],
                    temperature=self.generation_params["temperature"],
                    diversity_strength=self.generation_params["diversity_strength"]
                )
            
            # 转换为氨基酸序列
            for i, (tokens, target_len) in enumerate(zip(generated_tokens, target_lengths)):
                full_sequence = tokens_to_sequence(tokens.cpu().numpy())
                truncated_sequence = full_sequence[:target_len]
                
                if len(truncated_sequence) >= 10 and truncated_sequence.count('X') == 0:
                    all_sequences.append({
                        'id': f'DiverseConditional_Seq_{len(all_sequences) + 1:04d}',
                        'sequence': truncated_sequence,
                        'length': len(truncated_sequence),
                        'batch': batch_idx + 1
                    })
        
        logger.info(f"✅ 成功生成 {len(all_sequences)} 条序列")
        return all_sequences
    
    def save_sequences(self, sequences, output_prefix="diverse_conditional_800_sequences"):
        """保存生成的序列"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为FASTA格式
        fasta_file = f"{output_prefix}_{timestamp}.fasta"
        with open(fasta_file, 'w') as f:
            for seq_data in sequences:
                f.write(f">{seq_data['id']} | Length={seq_data['length']} | "
                       f"CFG={self.generation_params['guidance_scale']} | "
                       f"T={self.generation_params['temperature']} | "
                       f"D={self.generation_params['diversity_strength']}\n")
                f.write(f"{seq_data['sequence']}\n")
        
        logger.info(f"💾 序列已保存到: {fasta_file}")
        return fasta_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多样性条件抗菌肽序列生成器")
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径")
    parser.add_argument("--num_sequences", type=int, default=800, help="生成序列数量")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG引导强度")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--diversity_strength", type=float, default=1.2, help="多样性强度")
    
    args = parser.parse_args()
    
    try:
        # 初始化生成器
        generator = DiverseConditionalSequenceGenerator(args.checkpoint)
        
        # 更新参数
        generator.generation_params.update({
            "guidance_scale": args.guidance_scale,
            "temperature": args.temperature,
            "diversity_strength": args.diversity_strength
        })
        
        # 加载模型
        generator.load_models()
        
        # 生成序列
        sequences = generator.generate_sequences(args.num_sequences)
        
        if not sequences:
            logger.error("❌ 未生成任何序列")
            return
        
        # 保存序列
        fasta_file = generator.save_sequences(sequences)
        
        # 统计报告
        lengths = [seq['length'] for seq in sequences]
        logger.info("📈 生成统计:")
        logger.info(f"   总序列数: {len(sequences)}")
        logger.info(f"   平均长度: {sum(lengths)/len(lengths):.1f}")
        logger.info(f"   长度范围: {min(lengths)}-{max(lengths)}")
        logger.info(f"   CFG引导强度: {generator.generation_params['guidance_scale']}")
        logger.info(f"   采样温度: {generator.generation_params['temperature']}")
        logger.info(f"   多样性强度: {generator.generation_params['diversity_strength']}")
        
        logger.info("🎉 多样性条件生成完成！")
        
    except Exception as e:
        logger.error(f"❌ 生成过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()