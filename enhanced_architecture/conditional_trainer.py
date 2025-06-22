#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
条件扩散模型的统一训练器
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import sys
from typing import List, Dict
from tqdm import tqdm

# 导入我们创建的组件
from gram_predictor.data.conditional_dataset import ConditionalDataset, load_sequences_from_file
from gram_predictor.diffusion_models.conditional_d3pm import ConditionalD3PMUNet, ConditionalD3PMDiffusion, D3PMScheduler
from enhanced_architecture.conditional_esm2_extractor import ConditionalESM2FeatureExtractor
from gram_predictor.data_loader import sequence_to_tokens # 用于将序列转换为token ID

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConditionalTrainer:
    """用于训练条件扩散模型的训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')
        self.output_dir = self.config.get("output_dir", "output/checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型将保存到: {self.output_dir}")
        
        # 初始化所有组件
        self._init_components()
        
    def _init_components(self):
        """初始化模型、数据加载器、优化器等"""
        logger.info("正在初始化组件...")
        
        # 1. 特征提取器
        self.feature_extractor = ConditionalESM2FeatureExtractor(
            model_name=self.config.get("esm_model", "facebook/esm2_t8_215M_UR50D"),
            condition_dim=self.config["condition_dim"],
            use_layers=self.config.get("esm_layers", [2, 4, 6])
        ).to(self.device)
        
        # 2. 扩散模型
        unet = ConditionalD3PMUNet(
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            condition_dim=self.config["condition_dim"]
        )
        scheduler = D3PMScheduler(num_timesteps=self.config["num_timesteps"])
        self.diffusion_model = ConditionalD3PMDiffusion(unet, scheduler, self.device)
        
        # 检查并应用DataParallel以支持多GPU
        if torch.cuda.device_count() > 1:
            logger.info(f"检测到 {torch.cuda.device_count()} 个GPU，使用 DataParallel。")
            self.diffusion_model.model = torch.nn.DataParallel(self.diffusion_model.model)
        
        # 3. 数据集和数据加载器
        sequences = load_sequences_from_file(self.config["data_path"])
        dataset = ConditionalDataset(
            sequences,
            pairing_strategy=self.config.get("pairing_strategy", "random"),
            num_references=self.config.get("num_references", 1)
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # 初始化验证数据加载器 (如果提供了验证数据)
        if self.config.get("val_data_path"):
            val_sequences = load_sequences_from_file(self.config["val_data_path"])
            val_dataset = ConditionalDataset(
                val_sequences,
                pairing_strategy=self.config.get("pairing_strategy", "random"),
                num_references=self.config.get("num_references", 1)
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                collate_fn=self.collate_fn
            )
        else:
            self.val_loader = None
        
        # 4. 优化器和学习率调度器
        self.optimizer = optim.AdamW(
            self.diffusion_model.model.parameters(),
            lr=self.config["learning_rate"]
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9
        )
        
        logger.info("✅ 所有组件初始化完成。")

    def collate_fn(self, batch: List[Dict[str, List[str]]]) -> Dict[str, torch.Tensor]:
        """
        自定义的collate函数，用于处理批次数据。
        1. 提取条件特征。
        2. 将目标序列转换为token张量。
        """
        target_seqs = [item['target_sequence'][0] for item in batch]
        ref_seqs_list = [item['reference_sequences'] for item in batch]
        
        # 1. 提取条件特征
        # 注意：这里为了简化，对每个样本的多个参考序列取平均特征
        batch_condition_features = []
        for ref_seqs in ref_seqs_list:
            if ref_seqs:
                # 提取特征并取平均
                features = self.feature_extractor.extract_condition_features(ref_seqs)
                features = features.mean(dim=0)
                batch_condition_features.append(features)
            else:
                # 如果没有参考序列，使用零向量或特殊的null特征
                batch_condition_features.append(torch.zeros(self.config["condition_dim"], device=self.device))
        
        condition_features = torch.stack(batch_condition_features).to(self.device)
        
        # 2. 处理目标序列
        # 假设所有序列长度相同，或者需要进行填充
        # 这里我们使用简单的截断/填充到固定长度
        max_len = self.config.get("max_seq_len", 50)
        target_tokens_list = []
        for seq in target_seqs:
            tokens = sequence_to_tokens(seq, max_length=max_len)
            target_tokens_list.append(tokens)
            
        target_tokens = torch.stack(target_tokens_list).to(self.device)
        
        return {
            'target_tokens': target_tokens,
            'condition_features': condition_features
        }

    def train(self):
        """主训练循环"""
        logger.info("🚀 开始训练...")
        
        for epoch in range(self.config["epochs"]):
            self.diffusion_model.model.train()
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            total_loss = 0.0
            
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                target_tokens = batch['target_tokens']
                condition_features = batch['condition_features']
                
                # 计算损失
                loss = self.diffusion_model.training_loss(target_tokens, condition_features)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = total_loss / len(self.train_loader)
            self.lr_scheduler.step()
            
            logger.info(f"Epoch {epoch+1} 完成 | 平均训练损失: {avg_loss:.4f} | 当前学习率: {self.lr_scheduler.get_last_lr()[0]:.6f}")
            
            # 运行验证
            if self.val_loader:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch+1} | 验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
            
            # 定期保存最新模型
            if (epoch + 1) % self.config.get("save_interval", 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
        logger.info("✅ 训练完成！")

    def validate(self):
        """在验证集上评估模型"""
        self.diffusion_model.model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                target_tokens = batch['target_tokens']
                condition_features = batch['condition_features']
                
                loss = self.diffusion_model.training_loss(target_tokens, condition_features)
                total_val_loss += loss.item()
                
        return total_val_loss / len(self.val_loader)

    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        # 只保存模型的state_dict
        torch.save({
            'model_state_dict': self.diffusion_model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        
        logger.info(f"模型已保存到: {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """加载模型检查点"""
        checkpoint_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(checkpoint_path):
            logger.warning(f"检查点文件未找到: {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.diffusion_model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"模型已从 {checkpoint_path} 加载。")


if __name__ == '__main__':
    # 根据操作系统自动选择配置
    if sys.platform == "darwin":
        # macOS (本地) 配置: 使用小模型和CPU
        logger.info("检测到macOS系统，使用本地测试配置。")
        config = {
            "esm_model": "facebook/esm2_t6_8M_UR50D",
            "condition_dim": 320,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_timesteps": 100,
            "max_seq_len": 30,
            "data_path": "mock_train_sequences.txt",
            "val_data_path": "mock_val_sequences.txt",
            "output_dir": "test_checkpoints_local",
            "pairing_strategy": "random",
            "num_references": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "epochs": 2,
            "save_interval": 1,
        }
    else:
        # Linux (服务器) 配置: 使用大模型和GPU
        logger.info("检测到Linux系统，使用服务器训练配置。")
        config = {
            "esm_model": "facebook/esm2_t33_650M_UR50D",
            "condition_dim": 512,
            "hidden_dim": 512,
            "num_layers": 8,
            "num_timesteps": 1000,
            "max_seq_len": 100,
            "data_path": "data/main_training_sequences.txt", # 假设这是真实数据路径
            "val_data_path": "data/validation_sequences.txt", # 假设这是真实数据路径
            "output_dir": "checkpoints_650M",
            "pairing_strategy": "random", # 后续可以改为更高级的策略
            "num_references": 3,
            "batch_size": 16, # 针对4090D可以设置更大的batch size
            "learning_rate": 5e-5,
            "epochs": 50,
            "save_interval": 5,
        }

    # --- 以下为测试执行逻辑 ---
    
    # 如果是本地测试，创建模拟数据文件
    if sys.platform == "darwin":
        mock_train_sequences = [("A" * 25) for _ in range(10)]
        mock_val_sequences = [("V" * 25) for _ in range(5)]
        
        with open(config["data_path"], 'w') as f:
            for seq in mock_train_sequences:
                f.write(seq + "\n")
                
        with open(config["val_data_path"], 'w') as f:
            for seq in mock_val_sequences:
                f.write(seq + "\n")
    else:
        # 在服务器上，我们假设真实数据文件已存在
        # 可以添加检查确保文件存在
        if not os.path.exists(config["data_path"]) or not os.path.exists(config["val_data_path"]):
            logger.error(f"服务器模式下，数据文件未找到: {config['data_path']} 或 {config['val_data_path']}")
            logger.error("请确保数据文件已准备好，或在本地模式下运行以使用模拟数据。")
            sys.exit(1) # 退出脚本
            
    # 运行训练器
    mock_train_sequences = [("A" * 25) for _ in range(10)]
    mock_val_sequences = [("V" * 25) for _ in range(5)]
    
    with open(config["data_path"], 'w') as f:
        for seq in mock_train_sequences:
            f.write(seq + "\n")
            
    with open(config["val_data_path"], 'w') as f:
        for seq in mock_val_sequences:
            f.write(seq + "\n")
            
    # 运行训练器
    try:
        trainer = ConditionalTrainer(config)
        trainer.train()
        
        # 验证检查点是否已创建
        assert os.path.exists(os.path.join(config["output_dir"], "best_model.pt"))
        assert os.path.exists(os.path.join(config["output_dir"], "checkpoint_epoch_2.pt"))
        logger.info("✅ 检查点文件创建成功！")
        
        # 测试加载功能
        logger.info("测试加载检查点...")
        new_trainer = ConditionalTrainer(config)
        new_trainer.load_checkpoint("best_model.pt")
        logger.info("✅ 检查点加载成功！")
        
    except Exception as e:
        logger.error(f"❌ 训练器测试失败: {e}", exc_info=True)
    finally:
        # 清理模拟文件和目录
        import shutil
        if os.path.exists(config["data_path"]):
            os.remove(config["data_path"])
        if os.path.exists(config["val_data_path"]):
            os.remove(config["val_data_path"])
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])
        logger.info("已清理所有模拟文件和目录。")