"""
主训练脚本：整合ESM-2辅助编码器和D3PM扩散模型的统一训练管道
实现针对革兰氏阴性菌的抗菌肽生成模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
import json

# 导入项目模块
from config.model_config import get_config
from esm2_auxiliary_encoder import ESM2AuxiliaryEncoder, ContrastiveLoss
from diffusion_models.d3pm_diffusion import D3PMDiffusion, D3PMScheduler, D3PMUNet
from data_loader import AntimicrobialPeptideDataset, ContrastiveAMPDataset, collate_contrastive_batch
from evaluation.evaluator import ModelEvaluator

class EnhancedAMPTrainer:
    """增强型抗菌肽生成模型训练器"""
    
    def __init__(self, config_name: str = "default"):
        """
        初始化训练器
        
        Args:
            config_name: 配置名称 ('default', 'quick_test', 'production')
        """
        self.config = get_config(config_name)
        # 强制使用CPU以避免MPS兼容性问题
        # self.device = torch.device("cpu")
        # 如果在服务器上，可以改为：
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型组件
        self.esm2_encoder = None
        self.diffusion_model = None
        self.scheduler = None
        self.contrastive_loss = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.contrastive_loader = None
        
        # 优化器和调度器
        self.esm_optimizer = None
        self.diffusion_optimizer = None
        self.lr_scheduler = None
        
        # 监控工具
        self.writer = None
        self.evaluator = None
        
        self.logger.info(f"训练器初始化完成，使用配置: {config_name}")
        self.logger.info(f"设备: {self.device}")
    
    def setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = os.path.join(self.config.training.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize_models(self):
        """初始化所有模型组件"""
        self.logger.info("初始化模型组件...")
        
        # 1. ESM-2辅助编码器
        self.esm2_encoder = ESM2AuxiliaryEncoder(self.config.esm2).to(self.device)
        
        # 2. D3PM扩散模型和调度器
        self.scheduler = D3PMScheduler(
            num_timesteps=self.config.diffusion.num_timesteps,
            schedule_type=self.config.diffusion.schedule_type,
            vocab_size=self.config.diffusion.vocab_size
        )
        
        # 创建D3PM UNet模型
        d3pm_unet = D3PMUNet(
            vocab_size=self.config.diffusion.vocab_size,
            max_seq_len=self.config.diffusion.max_seq_len,
            hidden_dim=self.config.diffusion.hidden_dim,
            num_layers=self.config.diffusion.num_layers,
            num_heads=self.config.diffusion.num_heads,
            dropout=self.config.diffusion.dropout
        )
        
        self.diffusion_model = D3PMDiffusion(
            model=d3pm_unet,
            scheduler=self.scheduler,
            device=self.device
        )
        
        # 3. 对比学习损失函数
        if self.config.esm2.use_contrastive_learning:
            self.contrastive_loss = ContrastiveLoss(
                temperature=self.config.esm2.contrastive_temperature
            )
        
        # 4. 模型评估器
        self.evaluator = ModelEvaluator(self.config.evaluation)
        
        # 输出模型信息
        esm_params = sum(p.numel() for p in self.esm2_encoder.parameters() if p.requires_grad)
        diffusion_params = sum(p.numel() for p in self.diffusion_model.model.parameters())
        
        self.logger.info(f"ESM-2编码器可训练参数: {esm_params:,}")
        self.logger.info(f"扩散模型参数: {diffusion_params:,}")
        self.logger.info(f"总参数量: {esm_params + diffusion_params:,}")
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        self.logger.info("设置数据加载器...")
        
        # 1. 主训练数据集 (用于扩散模型训练)
        main_dataset = AntimicrobialPeptideDataset(
            sequences_file=self.config.data.main_sequences_path,
            max_length=self.config.data.max_sequence_length
        )
        
        # 2. 对比学习数据集 (用于ESM-2特征学习)
        if self.config.esm2.use_contrastive_learning:
            contrastive_dataset = ContrastiveAMPDataset(
                positive_file=self.config.data.positive_sequences_path,
                negative_file=self.config.data.negative_sequences_path,
                max_length=self.config.data.max_sequence_length,
                negative_sample_ratio=self.config.esm2.negative_sample_ratio
            )
            
            self.contrastive_loader = DataLoader(
                contrastive_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                collate_fn=collate_contrastive_batch
            )
        
        # 3. 数据集分割
        total_size = len(main_dataset)
        train_size = int(self.config.data.train_ratio * total_size)
        val_size = int(self.config.data.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            main_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 4. 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        self.logger.info(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
        if self.contrastive_loader:
            self.logger.info(f"对比学习数据集大小: {len(contrastive_dataset)}")
    
    def setup_optimizers(self):
        """设置优化器和学习率调度器"""
        self.logger.info("设置优化器...")
        
        # ESM-2编码器优化器 (学习率较低)
        esm_params = [p for p in self.esm2_encoder.parameters() if p.requires_grad]
        if esm_params:
            self.esm_optimizer = optim.AdamW(
                esm_params,
                lr=self.config.training.esm_learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999)
            )
        
        # 扩散模型优化器
        self.diffusion_optimizer = optim.AdamW(
            self.diffusion_model.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        if self.config.training.use_lr_scheduler:
            total_steps = len(self.train_loader) * self.config.training.num_epochs
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.diffusion_optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )
    
    def setup_monitoring(self):
        """设置监控工具"""
        if self.config.training.use_tensorboard:
            log_dir = os.path.join(self.config.training.output_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir)
        
        if self.config.training.use_wandb:
            wandb.init(
                project=self.config.training.wandb_project,
                config=self.config.__dict__,
                name=f"enhanced_amp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def contrastive_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        对比学习训练步骤
        
        Args:
            batch: 包含正负样本的批次数据
            
        Returns:
            损失字典
        """
        if not self.config.esm2.use_contrastive_learning:
            return {}
        
        # 提取正负样本特征
        positive_features, negative_features = self.esm2_encoder.extract_contrastive_features(
            batch['positive_sequences'], batch['negative_sequences']
        )
        
        # 计算对比学习损失
        contrastive_loss = self.contrastive_loss(
            positive_features, negative_features
        )
        
        # 反向传播
        if self.esm_optimizer:
            self.esm_optimizer.zero_grad()
            contrastive_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.esm2_encoder.parameters(),
                self.config.training.max_grad_norm
            )
            self.esm_optimizer.step()
        
        return {'contrastive_loss': contrastive_loss.item()}
    
    def diffusion_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        扩散模型训练步骤
        
        Args:
            batch: 包含序列数据的批次
            
        Returns:
            损失字典
        """
        sequences = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 暂时不使用ESM-2特征（先让基本训练工作）
        # TODO: 后续可以添加ESM-2特征作为条件信息
        
        # 扩散模型训练损失
        loss = self.diffusion_model.training_loss(x_start=sequences)
        
        # 反向传播
        self.diffusion_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.diffusion_model.model.parameters(),
            self.config.training.max_grad_norm
        )
        self.diffusion_optimizer.step()
        
        if self.lr_scheduler:
            self.lr_scheduler.step()
        
        return {'diffusion_loss': loss.item()}
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.esm2_encoder.train()
        self.diffusion_model.model.train()  # 设置扩散模型为训练模式
        
        epoch_losses = {'diffusion_loss': [], 'contrastive_loss': []}
        
        # 创建进度条
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        )
        
        for batch_idx, batch in pbar:
            self.global_step += 1
            
            # 1. 扩散模型训练步骤
            diffusion_losses = self.diffusion_training_step(batch)
            epoch_losses['diffusion_loss'].append(diffusion_losses['diffusion_loss'])
            
            # 2. 对比学习训练步骤 (如果启用)
            if self.config.esm2.use_contrastive_learning and self.contrastive_loader:
                try:
                    contrastive_batch = next(iter(self.contrastive_loader))
                    contrastive_losses = self.contrastive_training_step(contrastive_batch)
                    if contrastive_losses:
                        epoch_losses['contrastive_loss'].append(contrastive_losses['contrastive_loss'])
                except StopIteration:
                    pass
            
            # 更新进度条
            current_losses = {k: v[-1] if v else 0.0 for k, v in epoch_losses.items()}
            pbar.set_postfix(current_losses)
            
            # 记录到tensorboard/wandb
            if self.global_step % self.config.training.log_interval == 0:
                self.log_metrics(current_losses, prefix="train")
        
        # 计算epoch平均损失
        avg_losses = {}
        for loss_name, loss_values in epoch_losses.items():
            if loss_values:
                avg_losses[f"avg_{loss_name}"] = np.mean(loss_values)
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.esm2_encoder.eval()
        self.diffusion_model.model.eval()  # 设置扩散模型为评估模式
        
        val_losses = {'diffusion_loss': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                sequences = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 暂时不使用ESM-2特征（先让基本训练工作）
                # TODO: 后续可以添加ESM-2特征作为条件信息
                
                # 扩散模型损失
                loss = self.diffusion_model.training_loss(x_start=sequences)
                
                val_losses['diffusion_loss'].append(loss.item())
        
        # 计算平均损失
        avg_val_loss = np.mean(val_losses['diffusion_loss'])
        
        return {'val_diffusion_loss': avg_val_loss}
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """记录指标到监控工具"""
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{name}", value, self.global_step)
        
        if self.config.training.use_wandb:
            wandb_metrics = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=self.global_step)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'esm2_encoder_state_dict': self.esm2_encoder.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.model.state_dict(),  # 使用model.state_dict()
            'esm_optimizer_state_dict': self.esm_optimizer.state_dict() if self.esm_optimizer else None,
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新检查点
        checkpoint_dir = os.path.join(self.config.training.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
        
        self.logger.info(f"保存检查点到: {latest_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        self.logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.esm2_encoder.load_state_dict(checkpoint['esm2_encoder_state_dict'])
        self.diffusion_model.model.load_state_dict(checkpoint['diffusion_model_state_dict'])  # 使用model.load_state_dict()
        
        if self.esm_optimizer and checkpoint['esm_optimizer_state_dict']:
            self.esm_optimizer.load_state_dict(checkpoint['esm_optimizer_state_dict'])
        
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        
        if self.lr_scheduler and checkpoint['lr_scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"检查点加载完成，从epoch {self.current_epoch}继续训练")
    
    def generate_samples(self, num_samples: int = 10, max_length: int = 50) -> List[str]:
        """生成样本序列"""
        self.diffusion_model.model.eval()  # 修复：使用model.eval()而不是diffusion_model.eval()
        
        with torch.no_grad():
            # 生成随机噪声作为起点 - 修复：使用batch_size和seq_len参数
            generated_sequences = self.diffusion_model.sample(
                batch_size=num_samples,
                seq_len=max_length,
                num_inference_steps=self.config.diffusion.num_inference_steps
            )
            
            # 转换为氨基酸序列
            sequences = []
            for seq_tokens in generated_sequences:
                # 导入序列转换函数
                from data_loader import tokens_to_sequence
                seq_str = tokens_to_sequence(seq_tokens.cpu().numpy())
                sequences.append(seq_str)
        
        return sequences
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        
        # 初始化所有组件
        self.initialize_models()
        self.setup_data_loaders()
        self.setup_optimizers()
        self.setup_monitoring()
        
        # 训练循环
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_losses = self.train_epoch()
            
            # 验证阶段
            if epoch % self.config.training.val_interval == 0:
                val_losses = self.validate_epoch()
                
                # 记录验证指标
                self.log_metrics(val_losses, prefix="val")
                
                # 检查是否为最佳模型
                current_val_loss = val_losses['val_diffusion_loss']
                is_best = current_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_val_loss
                
                self.logger.info(
                    f"Epoch {epoch + 1}: "
                    f"训练损失: {train_losses.get('avg_diffusion_loss', 0):.4f}, "
                    f"验证损失: {current_val_loss:.4f}, "
                    f"最佳验证损失: {self.best_val_loss:.4f}"
                )
                
                # 保存检查点
                if epoch % self.config.training.save_interval == 0:
                    self.save_checkpoint(epoch, is_best)
                
                # 生成样本 (可选)
                if epoch % self.config.training.sample_interval == 0:
                    samples = self.generate_samples(num_samples=5)
                    self.logger.info("生成的样本序列:")
                    for i, seq in enumerate(samples, 1):
                        self.logger.info(f"  {i}: {seq}")
        
        # 训练完成
        self.logger.info("训练完成!")
        
        # 最终保存
        self.save_checkpoint(self.config.training.num_epochs - 1, False)
        
        # 关闭监控工具
        if self.writer:
            self.writer.close()
        
        if self.config.training.use_wandb:
            wandb.finish()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强型抗菌肽生成模型训练")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        choices=["default", "quick_test", "production"],
        help="配置名称"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="从检查点恢复训练的路径"
    )
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = EnhancedAMPTrainer(config_name=args.config)
    
    # 如果指定了恢复路径，加载检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
