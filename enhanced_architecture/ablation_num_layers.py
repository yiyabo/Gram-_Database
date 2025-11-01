#!/usr/bin/env python3
"""
消融实验：Transformer层数对生成质量的影响
测试 num_layers = 4, 6, 8, 12 四种配置
"""

import os
import sys
import torch
import json
import logging
from datetime import datetime
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gram_predictor.config.model_config import get_default_config
from enhanced_architecture.main_trainer import EnhancedAMPTrainer
from gram_predictor.data_loader import tokens_to_sequence

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ablation_num_layers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NumLayersAblationExperiment:
    """Transformer层数消融实验"""
    
    def __init__(self, base_output_dir: str = "ablation_results/num_layers"):
        """
        初始化消融实验
        
        Args:
            base_output_dir: 实验结果输出目录
        """
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        
        # 定义要测试的层数配置
        self.layer_configs = [4, 6, 8, 12]
        
        # 存储实验结果
        self.results = {}
        
        logger.info(f"消融实验初始化完成")
        logger.info(f"测试配置: num_layers = {self.layer_configs}")
        logger.info(f"输出目录: {base_output_dir}")
    
    def create_config_for_layers(self, num_layers: int):
        """
        为指定层数创建配置
        
        Args:
            num_layers: Transformer层数
            
        Returns:
            配置对象
        """
        # 从默认配置开始
        config = get_default_config()
        
        # === 统一的基础配置（确保公平对比） ===
        
        # ESM-2配置（所有实验使用相同的ESM-2）
        config.esm2.model_name = "facebook/esm2_t12_35M_UR50D"  # 35M模型
        config.esm2.feature_dim = 480
        config.esm2.freeze_esm = True  # 冻结ESM-2，只训练扩散模型
        
        # 扩散模型配置
        config.diffusion.vocab_size = 21
        config.diffusion.hidden_dim = 512  # 固定隐藏层维度
        config.diffusion.num_layers = num_layers  # 唯一变化的参数
        config.diffusion.num_heads = 8
        config.diffusion.max_seq_len = 100
        config.diffusion.dropout = 0.1
        config.diffusion.num_timesteps = 1000
        config.diffusion.schedule_type = 'cosine'
        
        # 训练配置（统一）
        config.training.num_epochs = 100  # 消融实验用较少轮数以节省时间
        config.training.learning_rate = 1e-4
        config.training.batch_size = 32
        config.training.gradient_accumulation_steps = 1
        config.training.warmup_epochs = 5
        config.training.validation_frequency = 5
        config.training.save_frequency = 10
        config.training.use_mixed_precision = True
        
        # 数据配置
        config.data.batch_size = 32
        config.data.max_sequence_length = 100
        config.data.num_workers = 4
        
        # 评估配置
        config.evaluation.num_samples = 100  # 每次评估生成100条序列
        config.evaluation.sample_batch_size = 20
        
        # 输出目录（每个配置单独目录）
        config.training.output_dir = os.path.join(
            self.base_output_dir, 
            f"layers_{num_layers}"
        )
        os.makedirs(config.training.output_dir, exist_ok=True)
        
        logger.info(f"创建配置: num_layers={num_layers}")
        logger.info(f"  输出目录: {config.training.output_dir}")
        
        return config
    
    def train_single_config(self, num_layers: int):
        """
        训练单个配置
        
        Args:
            num_layers: Transformer层数
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"开始训练: num_layers = {num_layers}")
        logger.info(f"{'='*80}\n")
        
        try:
            # 创建配置
            config = self.create_config_for_layers(num_layers)
            
            # 保存配置到文件
            config_path = os.path.join(config.training.output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'num_layers': num_layers,
                    'hidden_dim': config.diffusion.hidden_dim,
                    'num_heads': config.diffusion.num_heads,
                    'num_epochs': config.training.num_epochs,
                    'learning_rate': config.training.learning_rate,
                    'batch_size': config.data.batch_size
                }, f, indent=2)
            
            # 创建临时配置类（绕过config_name限制）
            from gram_predictor.config.model_config import ModelConfig
            
            # 直接创建训练器并传入配置对象
            trainer = EnhancedAMPTrainer(config_name="default")
            trainer.config = config  # 覆盖默认配置
            
            # 重新初始化模型（使用新配置）
            trainer.initialize_models()
            trainer.setup_data_loaders()
            trainer.setup_optimizers()
            trainer.setup_monitoring()
            
            # 开始训练
            logger.info(f"开始训练 {num_layers} 层模型...")
            trainer.train()
            
            # 记录成功
            self.results[num_layers] = {
                'status': 'success',
                'output_dir': config.training.output_dir,
                'config_path': config_path
            }
            
            logger.info(f"✅ num_layers={num_layers} 训练完成")
            
        except Exception as e:
            logger.error(f"❌ num_layers={num_layers} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            self.results[num_layers] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_all_models(self):
        """
        评估所有训练好的模型
        生成序列并比较质量
        """
        logger.info(f"\n{'='*80}")
        logger.info("开始评估所有模型")
        logger.info(f"{'='*80}\n")
        
        evaluation_results = {}
        
        for num_layers in self.layer_configs:
            if self.results.get(num_layers, {}).get('status') != 'success':
                logger.warning(f"跳过 num_layers={num_layers}（训练失败或未完成）")
                continue
            
            logger.info(f"\n评估 num_layers={num_layers} 的模型...")
            
            try:
                # 加载模型检查点
                checkpoint_path = os.path.join(
                    self.base_output_dir,
                    f"layers_{num_layers}",
                    "checkpoints",
                    "best.pt"
                )
                
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"检查点不存在: {checkpoint_path}")
                    continue
                
                # 从检查点加载配置和模型
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # 生成测试序列
                from gram_predictor.generation_service import SequenceGenerationService
                
                # 创建临时生成服务
                gen_service = SequenceGenerationService(checkpoint_path=checkpoint_path)
                gen_service.load_models()
                
                # 生成100条序列用于评估
                result = gen_service.generate_sequences(
                    num_sequences=100,
                    seq_length=40,
                    sampling_method='diverse',
                    temperature=1.0,
                    diversity_strength=0.5
                )
                
                if result['success']:
                    sequences = result['sequences']
                    
                    # 评估序列质量
                    from enhanced_architecture.evaluation.evaluator import ModelEvaluator, SequenceAnalyzer
                    
                    analyzer = SequenceAnalyzer()
                    
                    # 计算多样性指标
                    diversity = analyzer.calculate_amino_acid_diversity(sequences)
                    similarity = analyzer.calculate_sequence_similarity(sequences)
                    
                    # 统计有效序列
                    valid_count = sum(1 for seq in sequences if analyzer.is_valid_sequence(seq))
                    valid_ratio = valid_count / len(sequences)
                    
                    # 保存结果
                    evaluation_results[num_layers] = {
                        'num_sequences': len(sequences),
                        'valid_sequences': valid_count,
                        'valid_ratio': valid_ratio,
                        'amino_acid_diversity': diversity,
                        'sequence_similarity': similarity,
                        'diversity_score': 1.0 - similarity,
                        'checkpoint_size_mb': os.path.getsize(checkpoint_path) / (1024*1024)
                    }
                    
                    logger.info(f"  有效序列: {valid_count}/{len(sequences)} ({valid_ratio:.2%})")
                    logger.info(f"  氨基酸多样性: {diversity:.3f}")
                    logger.info(f"  序列多样性: {1.0 - similarity:.3f}")
                    
                else:
                    logger.error(f"生成序列失败: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存评估结果
        results_path = os.path.join(self.base_output_dir, "ablation_results.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"\n评估结果已保存到: {results_path}")
        
        # 生成对比报告
        self.generate_comparison_report(evaluation_results)
        
        return evaluation_results
    
    def generate_comparison_report(self, results: Dict):
        """生成对比报告"""
        report = """
# Transformer层数消融实验报告

## 实验设置
- 变化参数: num_layers (4, 6, 8, 12)
- 固定参数: hidden_dim=512, num_heads=8, ESM-2=35M
- 训练轮数: 100 epochs
- 评估序列数: 100条/配置

## 实验结果

| num_layers | 有效序列比例 | 氨基酸多样性 | 序列多样性 | 模型大小(MB) |
|-----------|-------------|-------------|-----------|-------------|
"""
        
        for num_layers in sorted(results.keys()):
            r = results[num_layers]
            report += f"| {num_layers} | {r['valid_ratio']:.2%} | {r['amino_acid_diversity']:.3f} | {r['diversity_score']:.3f} | {r['checkpoint_size_mb']:.1f} |\n"
        
        report += """
## 结论

基于实验结果：

1. **最优层数**: [需要根据结果填写]
2. **性能-效率权衡**: [需要根据结果分析]
3. **建议**: [实际应用建议]

## 详细分析

[根据实验结果补充详细分析]
"""
        
        report_path = os.path.join(self.base_output_dir, "ablation_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"对比报告已保存到: {report_path}")
        print(report)
    
    def run_full_experiment(self):
        """运行完整的消融实验"""
        logger.info("\n" + "="*80)
        logger.info("开始 Transformer 层数消融实验")
        logger.info("="*80 + "\n")
        
        # 训练所有配置
        for num_layers in self.layer_configs:
            self.train_single_config(num_layers)
        
        # 评估所有模型
        logger.info("\n" + "="*80)
        logger.info("训练完成，开始评估")
        logger.info("="*80 + "\n")
        
        evaluation_results = self.evaluate_all_models()
        
        # 总结
        logger.info("\n" + "="*80)
        logger.info("消融实验完成")
        logger.info("="*80)
        logger.info(f"\n实验结果保存在: {self.base_output_dir}")
        logger.info(f"  - 训练日志: ablation_num_layers_*.log")
        logger.info(f"  - 评估结果: ablation_results.json")
        logger.info(f"  - 对比报告: ablation_report.md")
        
        return evaluation_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transformer层数消融实验")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_results/num_layers",
        help="实验结果输出目录"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        default=[4, 6, 8, 12],
        help="要测试的层数列表，如: --layers 4 8 12"
    )
    parser.add_argument(
        "--eval_only",
        action='store_true',
        help="仅评估已训练的模型，不重新训练"
    )
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = NumLayersAblationExperiment(base_output_dir=args.output_dir)
    experiment.layer_configs = args.layers
    
    if args.eval_only:
        # 仅评估
        logger.info("仅评估模式：跳过训练，直接评估已有模型")
        experiment.evaluate_all_models()
    else:
        # 完整实验（训练+评估）
        experiment.run_full_experiment()


if __name__ == "__main__":
    main()

