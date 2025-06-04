"""
配置文件：定义增强型蛋白质生成模型的所有参数
包括扩散模型、ESM-2辅助编码器和训练参数
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class DiffusionConfig:
    """扩散模型配置"""
    # 基础参数
    vocab_size: int = 21  # 20个氨基酸 + PAD token
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 100
    dropout: float = 0.1
    
    # 扩散参数
    num_timesteps: int = 1000
    schedule_type: str = 'cosine'  # 'linear' or 'cosine'
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # 采样参数
    num_inference_steps: int = 50
    ddim_eta: float = 0.0
    
    def __post_init__(self):
        """验证配置参数"""
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim必须能被num_heads整除"
        assert self.schedule_type in ['linear', 'cosine'], "不支持的调度类型"

@dataclass
class ESM2Config:
    """ESM-2辅助编码器配置"""
    # ESM-2模型参数
    model_name: str = "facebook/esm2_t12_35M_UR50D"  # ESM-2 35M参数模型（推荐用于双4090）
    freeze_esm: bool = True  # 是否冻结ESM-2参数
    
    # 特征提取参数
    extract_layers: List[int] = None  # 提取哪些层的特征，None表示最后一层
    pooling_method: str = 'attention'  # 'mean', 'max', 'attention', 'cls'
    feature_dim: int = 480  # ESM-2 35M的特征维度
    
    # 特征融合参数
    fusion_method: str = 'contrastive'  # 'mean', 'weighted_mean', 'clustering', 'contrastive'
    projection_dim: int = 512  # 投影到扩散模型hidden_dim
    
    # 对比学习参数
    use_contrastive_learning: bool = True
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.1
    negative_sample_ratio: float = 1.0
    
    # 训练参数
    max_length: int = 512  # 最大序列长度
    batch_size: int = 8    # 批处理大小（小模型用小批次）
    
    # 缓存参数
    cache_features: bool = True
    cache_dir: str = "./cache/esm_features"
    
    def __post_init__(self):
        if self.extract_layers is None:
            self.extract_layers = [-1]  # 默认使用最后一层

@dataclass
class DataConfig:
    """数据集配置"""
    # 数据路径
    gram_negative_path: str = "../data/Gram-.fasta"
    gram_positive_path: str = "../data/Gram+.fasta"
    gram_both_path: str = "../data/Gram+-.fasta"
    
    # 处理后的数据路径
    main_sequences_path: str = "./main_training_sequences.txt"
    positive_sequences_path: str = "./positive_sequences.txt"
    negative_sequences_path: str = "./negative_sequences.txt"
    
    # 数据参数
    max_sequence_length: int = 100
    min_sequence_length: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 数据增强
    use_data_augmentation: bool = True
    augmentation_ratio: float = 0.2
    
    # 批处理参数
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "训练/验证/测试比例之和必须为1"

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 100
    learning_rate: float = 1e-4
    esm_learning_rate: float = 1e-5  # ESM-2的学习率（较低）
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 学习率调度
    use_lr_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'exponential'
    warmup_steps: int = 1000
    
    # 优化器参数
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # 损失函数权重
    diffusion_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    reconstruction_loss_weight: float = 0.05
    
    # 验证和保存
    val_interval: int = 10  # 每多少个epoch验证一次
    save_interval: int = 20  # 每多少个epoch保存一次
    sample_interval: int = 50  # 每多少个epoch生成样本
    log_interval: int = 100  # 每多少步记录日志
    early_stopping_patience: int = 20
    
    # 混合精度训练
    use_mixed_precision: bool = True
    
    # 分布式训练
    use_distributed: bool = False
    local_rank: int = 0
    
    # 输出和监控
    output_dir: str = "./output"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "enhanced-amp-generation"

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 生成参数
    num_samples: int = 1000
    sample_batch_size: int = 100
    
    # 评估指标
    evaluate_diversity: bool = True
    evaluate_validity: bool = True
    evaluate_novelty: bool = True
    evaluate_activity: bool = True
    
    # 活性预测模型路径（如果有的话）
    activity_model_path: Optional[str] = None
    
    # 输出设置
    save_generated_sequences: bool = True
    output_dir: str = "./results"
    
    # 比较基线
    baseline_sequences_path: Optional[str] = None

@dataclass
class ModelConfig:
    """完整模型配置"""
    # 子配置
    diffusion: DiffusionConfig
    esm2: ESM2Config
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    
    # 设备和随机种子
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    seed: int = 42
    deterministic: bool = True
    
    # 模型保存
    model_save_dir: str = "./checkpoints"
    experiment_name: str = "enhanced_amp_generator"
    
    # 日志
    use_wandb: bool = False
    wandb_project: str = "amp-generation"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """后处理：自动设置设备等"""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 确保投影维度匹配
        self.esm2.projection_dim = self.diffusion.hidden_dim
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(
            diffusion=DiffusionConfig(**config_dict.get('diffusion', {})),
            esm2=ESM2Config(**config_dict.get('esm2', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['diffusion', 'esm2', 'data', 'training', 'evaluation']}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return {
            'diffusion': dataclasses.asdict(self.diffusion),
            'esm2': dataclasses.asdict(self.esm2),
            'data': dataclasses.asdict(self.data),
            'training': dataclasses.asdict(self.training),
            'evaluation': dataclasses.asdict(self.evaluation),
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'model_save_dir': self.model_save_dir,
            'experiment_name': self.experiment_name,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity
        }


# 预定义配置
def get_default_config() -> ModelConfig:
    """获取默认配置"""
    return ModelConfig(
        diffusion=DiffusionConfig(),
        esm2=ESM2Config(),
        data=DataConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig()
    )

def get_quick_test_config() -> ModelConfig:
    """获取快速测试配置（小模型，少数据）"""
    config = get_default_config()
    
    # 缩小模型
    config.diffusion.hidden_dim = 256
    config.diffusion.num_layers = 4
    config.diffusion.num_timesteps = 100
    config.diffusion.num_inference_steps = 20
    
    # 减少训练轮数
    config.training.num_epochs = 5
    config.training.validation_frequency = 2
    config.training.save_frequency = 5
    
    # 小批次
    config.data.batch_size = 8
    
    # 少样本评估
    config.evaluation.num_samples = 50
    config.evaluation.sample_batch_size = 10
    
    # 禁用ESM-2缓存以避免测试时的依赖
    config.esm2.cache_features = False
    
    return config

def get_production_config() -> ModelConfig:
    """获取生产环境配置（大模型，完整训练）"""
    config = get_default_config()
    
    # 更大的模型
    config.diffusion.hidden_dim = 768
    config.diffusion.num_layers = 12
    config.diffusion.num_heads = 12
    
    # 更多训练
    config.training.num_epochs = 200
    config.training.learning_rate = 5e-5
    
    # 更大批次
    config.data.batch_size = 64
    
    # 启用更多功能
    config.training.use_mixed_precision = True
    config.esm2.cache_features = True
    config.training.use_wandb = True
    
    return config

def get_dual_4090_config() -> ModelConfig:
    """获取双4090优化配置（48GB显存，高性能训练）"""
    config = get_default_config()
    
    # 使用更大的ESM-2模型
    config.esm2.model_name = "facebook/esm2_t30_150M_UR50D"  # 150M参数，最佳性能
    config.esm2.feature_dim = 640  # ESM-2 150M的特征维度
    config.esm2.freeze_esm = False  # 解冻ESM-2进行微调（显存充足时）
    config.esm2.batch_size = 16  # 适合大模型的批次大小
    
    # 扩散模型配置
    config.diffusion.hidden_dim = 1024  # 更大的隐藏层
    config.diffusion.num_layers = 16    # 更深的网络
    config.diffusion.num_heads = 16     # 更多注意力头
    config.diffusion.max_seq_len = 150  # 支持更长序列
    
    # 训练配置
    config.training.num_epochs = 300
    config.training.learning_rate = 3e-5
    config.training.esm_learning_rate = 5e-6  # ESM-2微调用更小学习率
    config.training.use_mixed_precision = True
    config.training.use_distributed = True  # 启用多GPU训练
    
    # 数据配置
    config.data.batch_size = 32  # 充分利用显存
    config.data.max_sequence_length = 150
    config.data.num_workers = 8  # 更多数据加载进程
    
    # 启用所有高级功能
    config.esm2.cache_features = True
    config.training.use_wandb = True
    config.training.use_tensorboard = True
    
    return config


def get_config(config_name: str = "default") -> ModelConfig:
    """
    根据配置名称获取配置
    
    Args:
        config_name: 配置名称 ('default', 'quick_test', 'production', 'dual_4090')
        
    Returns:
        模型配置
    """
    if config_name == "default":
        return get_default_config()
    elif config_name == "quick_test":
        return get_quick_test_config()
    elif config_name == "production":
        return get_production_config()
    elif config_name == "dual_4090":
        return get_dual_4090_config()
    else:
        raise ValueError(f"未知的配置名称: {config_name}. 可用选项: 'default', 'quick_test', 'production', 'dual_4090'")


if __name__ == "__main__":
    # 测试配置
    print("Testing configuration classes...")
    
    # 默认配置
    config = get_default_config()
    print(f"Default device: {config.device}")
    print(f"Diffusion hidden_dim: {config.diffusion.hidden_dim}")
    print(f"ESM2 projection_dim: {config.esm2.projection_dim}")
    
    # 快速测试配置
    test_config = get_quick_test_config()
    print(f"Test config epochs: {test_config.training.num_epochs}")
    print(f"Test config batch_size: {test_config.data.batch_size}")
    
    # 转换为字典并恢复
    config_dict = config.to_dict()
    restored_config = ModelConfig.from_dict(config_dict)
    print(f"Config serialization test: {config.device == restored_config.device}")
    
    print("Configuration test completed!")
