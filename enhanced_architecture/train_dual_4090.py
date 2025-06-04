#!/usr/bin/env python3
"""
专门针对双4090配置的训练脚本
修复多GPU支持和显存优化
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

class Dual4090Trainer:
    """专门针对双4090的训练器"""
    
    def __init__(self, config_name: str = "dual_4090"):
        """初始化训练器"""
        self.config = get_config(config_name)
        
        # 设置设备和多GPU
        self.setup_device()
        
        # 设置日志
        self.setup_logging()
        
        # 初始化变量
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
        
        # 优化器
        self.esm_optimizer = None
        self.diffusion_optimizer = None
        self.lr_scheduler = None
        
        # 监控工具
        self.writer = None
        self.evaluator = None
        
        self.logger.info(f"双4090训练器初始化完成，配置: {config_name}")
    
    def setup_device(self):
        """设置设备和多GPU支持"""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.device = torch.device("cuda:0")
            
            print(f"🔍 检测到 {self.num_gpus} 个GPU:")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 清理GPU缓存
            for i in range(self.num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            
            # 设置多GPU策略
            if self.num_gpus >= 2:
                self.use_multi_gpu = True
                print(f"✅ 启用多GPU训练 (使用 {self.num_gpus} 个GPU)")
                
                # 设置CUDA环境变量优化
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
            else:
                self.use_multi_gpu = False
                print("📱 使用单GPU训练")
        else:
            self.device = torch.device("cpu")
            self.num_gpus = 0
            self.use_multi_gpu = False
            print("⚠️ 未检测到GPU，使用CPU训练")
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = os.path.join(self.config.training.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"dual_4090_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        """初始化模型组件"""
        self.logger.info("🚀 初始化模型组件...")
        
        # 1. ESM-2辅助编码器
        self.logger.info(f"加载ESM-2模型: {self.config.esm2.model_name}")
        self.esm2_encoder = ESM2AuxiliaryEncoder(self.config.esm2)
        
        # 2. 扩散模型
        self.logger.info("初始化D3PM扩散模型...")
        self.scheduler = D3PMScheduler(
            num_timesteps=self.config.diffusion.num_timesteps,
            schedule_type=self.config.diffusion.schedule_type,
            vocab_size=self.config.diffusion.vocab_size
        )
        
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
        
        # 3. 移动到GPU
        self.esm2_encoder = self.esm2_encoder.to(self.device)
        self.diffusion_model.model = self.diffusion_model.model.to(self.device)
        
        # 4. 多GPU包装
        if self.use_multi_gpu:
            self.logger.info(f"🔗 启用多GPU DataParallel...")
            
            # ESM-2编码器多GPU（如果不冻结）
            if not self.config.esm2.freeze_esm:
                self.esm2_encoder = nn.DataParallel(self.esm2_encoder)
                self.logger.info("✅ ESM-2编码器已启用DataParallel")
            
            # 扩散模型多GPU
            self.diffusion_model.model = nn.DataParallel(self.diffusion_model.model)
            self.logger.info("✅ 扩散模型已启用DataParallel")
        
        # 5. 对比学习损失
        if self.config.esm2.use_contrastive_learning:
            self.contrastive_loss = ContrastiveLoss(
                temperature=self.config.esm2.contrastive_temperature
            )
        
        # 6. 评估器
        self.evaluator = ModelEvaluator(self.config.evaluation)
        
        # 输出模型信息
        esm_params = sum(p.numel() for p in self.esm2_encoder.parameters() if p.requires_grad)
        diffusion_params = sum(p.numel() for p in self.diffusion_model.model.parameters())
        
        self.logger.info(f"📊 模型参数统计:")
        self.logger.info(f"  ESM-2编码器可训练参数: {esm_params:,}")
        self.logger.info(f"  扩散模型参数: {diffusion_params:,}")
        self.logger.info(f"  总参数量: {esm_params + diffusion_params:,}")
        
        # 显存使用情况
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"  GPU {i} 显存: {allocated:.2f}GB 已分配, {reserved:.2f}GB 已保留, {total:.2f}GB 总计")
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        self.logger.info("📁 设置数据加载器...")
        
        # 主训练数据集
        main_dataset = AntimicrobialPeptideDataset(
            sequences_file=self.config.data.main_sequences_path,
            max_length=self.config.data.max_sequence_length
        )
        
        # 对比学习数据集
        if self.config.esm2.use_contrastive_learning:
            contrastive_dataset = ContrastiveAMPDataset(
                positive_file=self.config.data.positive_sequences_path,
                negative_file=self.config.data.negative_sequences_path,
                max_length=self.config.data.max_sequence_length,
                negative_sample_ratio=self.config.esm2.negative_sample_ratio
            )
            
            self.contrastive_loader = DataLoader(
                contrastive_dataset,
                batch_size=self.config.esm2.batch_size,  # 使用ESM-2专用批次大小
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory,
                collate_fn=collate_contrastive_batch
            )
        
        # 数据集分割
        total_size = len(main_dataset)
        train_size = int(self.config.data.train_ratio * total_size)
        val_size = int(self.config.data.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            main_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
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
        
        self.logger.info(f"📊 数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
        if self.contrastive_loader:
            self.logger.info(f"📊 对比学习数据集大小: {len(contrastive_dataset)}")
    
    def setup_optimizers(self):
        """设置优化器"""
        self.logger.info("⚙️ 设置优化器...")
        
        # ESM-2优化器
        esm_params = [p for p in self.esm2_encoder.parameters() if p.requires_grad]
        if esm_params:
            self.esm_optimizer = optim.AdamW(
                esm_params,
                lr=self.config.training.esm_learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999)
            )
            self.logger.info(f"  ESM-2优化器: AdamW, lr={self.config.training.esm_learning_rate}")
        
        # 扩散模型优化器
        self.diffusion_optimizer = optim.AdamW(
            self.diffusion_model.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999)
        )
        self.logger.info(f"  扩散模型优化器: AdamW, lr={self.config.training.learning_rate}")
        
        # 学习率调度器
        if self.config.training.use_lr_scheduler:
            total_steps = len(self.train_loader) * self.config.training.num_epochs
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.diffusion_optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )
            self.logger.info(f"  学习率调度器: CosineAnnealingLR")
    
    def setup_monitoring(self):
        """设置监控工具"""
        if self.config.training.use_tensorboard:
            log_dir = os.path.join(self.config.training.output_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir)
            self.logger.info(f"📈 TensorBoard日志: {log_dir}")
        
        if self.config.training.use_wandb:
            wandb.init(
                project=self.config.training.wandb_project,
                config=self.config.__dict__,
                name=f"dual_4090_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.logger.info("📊 WandB监控已启用")
    
    def train_step(self, batch):
        """单个训练步骤"""
        sequences = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 扩散模型训练
        loss = self.diffusion_model.training_loss(x_start=sequences)
        
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
    
    def contrastive_step(self, batch):
        """对比学习步骤"""
        if not self.config.esm2.use_contrastive_learning:
            return {}
        
        try:
            positive_features, negative_features = self.esm2_encoder.extract_contrastive_features(
                batch['positive_sequences'], batch['negative_sequences']
            )
            
            contrastive_loss = self.contrastive_loss(positive_features, negative_features)
            
            if self.esm_optimizer:
                self.esm_optimizer.zero_grad()
                contrastive_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.esm2_encoder.parameters(),
                    self.config.training.max_grad_norm
                )
                self.esm_optimizer.step()
            
            return {'contrastive_loss': contrastive_loss.item()}
        except Exception as e:
            self.logger.warning(f"对比学习步骤出错: {e}")
            return {}
    
    def train_epoch(self):
        """训练一个epoch"""
        self.esm2_encoder.train()
        self.diffusion_model.model.train()
        
        epoch_losses = {'diffusion_loss': [], 'contrastive_loss': []}
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        )
        
        for batch_idx, batch in pbar:
            self.global_step += 1
            
            # 扩散模型训练
            diffusion_losses = self.train_step(batch)
            epoch_losses['diffusion_loss'].append(diffusion_losses['diffusion_loss'])
            
            # 对比学习训练
            if self.config.esm2.use_contrastive_learning and self.contrastive_loader:
                try:
                    contrastive_batch = next(iter(self.contrastive_loader))
                    contrastive_losses = self.contrastive_step(contrastive_batch)
                    if contrastive_losses:
                        epoch_losses['contrastive_loss'].append(contrastive_losses['contrastive_loss'])
                except StopIteration:
                    pass
            
            # 更新进度条
            current_losses = {k: v[-1] if v else 0.0 for k, v in epoch_losses.items()}
            pbar.set_postfix(current_losses)
            
            # 记录指标
            if self.global_step % self.config.training.log_interval == 0:
                self.log_metrics(current_losses, prefix="train")
                
                # 显存监控
                if torch.cuda.is_available() and self.global_step % (self.config.training.log_interval * 10) == 0:
                    for i in range(self.num_gpus):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        self.logger.info(f"GPU {i} 显存使用: {allocated:.2f}GB")
        
        # 计算平均损失
        avg_losses = {}
        for loss_name, loss_values in epoch_losses.items():
            if loss_values:
                avg_losses[f"avg_{loss_name}"] = np.mean(loss_values)
        
        return avg_losses
    
    def validate_epoch(self):
        """验证epoch"""
        self.esm2_encoder.eval()
        self.diffusion_model.model.eval()
        
        val_losses = {'diffusion_loss': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                sequences = batch['input_ids'].to(self.device)
                loss = self.diffusion_model.training_loss(x_start=sequences)
                val_losses['diffusion_loss'].append(loss.item())
        
        avg_val_loss = np.mean(val_losses['diffusion_loss'])
        return {'val_diffusion_loss': avg_val_loss}
    
    def log_metrics(self, metrics, prefix=""):
        """记录指标"""
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{name}", value, self.global_step)
        
        if self.config.training.use_wandb:
            wandb_metrics = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=self.global_step)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        # 获取原始模型（去除DataParallel包装）
        esm2_state = self.esm2_encoder.module.state_dict() if isinstance(self.esm2_encoder, nn.DataParallel) else self.esm2_encoder.state_dict()
        diffusion_state = self.diffusion_model.model.module.state_dict() if isinstance(self.diffusion_model.model, nn.DataParallel) else self.diffusion_model.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'esm2_encoder_state_dict': esm2_state,
            'diffusion_model_state_dict': diffusion_state,
            'esm_optimizer_state_dict': self.esm_optimizer.state_dict() if self.esm_optimizer else None,
            'diffusion_optimizer_state_dict': self.diffusion_optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_dir = os.path.join(self.config.training.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 保存最佳模型: {best_path}")
        
        self.logger.info(f"💾 保存检查点: {latest_path}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("🚀 开始双4090训练...")
        
        # 初始化所有组件
        self.initialize_models()
        self.setup_data_loaders()
        self.setup_optimizers()
        self.setup_monitoring()
        
        # 训练循环
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            if epoch % self.config.training.val_interval == 0:
                val_losses = self.validate_epoch()
                self.log_metrics(val_losses, prefix="val")
                
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
        
        # 训练完成
        self.logger.info("✅ 训练完成!")
        self.save_checkpoint(self.config.training.num_epochs - 1, False)
        
        if self.writer:
            self.writer.close()
        if self.config.training.use_wandb:
            wandb.finish()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="双4090优化训练")
    parser.add_argument("--config", type=str, default="dual_4090", help="配置名称")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练路径")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = Dual4090Trainer(config_name=args.config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()