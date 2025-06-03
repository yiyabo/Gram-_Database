# 🚀 增强型抗菌肽生成模型 - 完整训练指南

## 📋 训练前准备

### 1. 环境检查与依赖安装

首先确保安装了所有必要的依赖：

```bash
# 安装/更新依赖
pip install -r requirements.txt

# 如果使用conda环境，建议创建新环境
conda create -n amp_generation python=3.9
conda activate amp_generation
pip install -r requirements.txt
```

### 2. 系统测试

在开始完整训练前，先运行系统测试：

```bash
cd enhanced_architecture
python start_training.py --test-only --config production
```

## 🎯 训练配置选择

### 配置对比

| 配置 | 用途 | 训练轮数 | 批次大小 | 模型大小 | 预计时间 |
|------|------|----------|----------|----------|----------|
| `quick_test` | 快速验证 | 5 | 8 | 小 | 10-30分钟 |
| `default` | 标准训练 | 100 | 32 | 中 | 2-6小时 |
| `production` | 生产级别 | 200 | 64 | 大 | 6-24小时 |

### 推荐训练策略

**阶段1: 快速验证** (推荐先运行)
```bash
python start_training.py --config quick_test
```

**阶段2: 完整训练**
```bash
python start_training.py --config production
```

## 🔧 生产级训练配置详解

### 模型参数
- **扩散模型**: 768维隐藏层，12层Transformer，12个注意力头
- **ESM-2模型**: facebook/esm2_t6_8M_UR50D (8M参数)
- **训练轮数**: 200 epochs
- **批次大小**: 64
- **学习率**: 5e-5 (扩散模型), 1e-5 (ESM-2)

### 高级功能
- ✅ 混合精度训练 (节省显存)
- ✅ ESM-2特征缓存 (加速训练)
- ✅ WandB监控 (可选)
- ✅ 梯度裁剪 (稳定训练)
- ✅ 学习率调度 (余弦退火)

## 🚀 开始完整训练

### 方法1: 标准训练 (推荐)

```bash
# 进入项目目录
cd enhanced_architecture

# 开始生产级训练
python start_training.py --config production
```

### 方法2: 自定义监控

如果要启用WandB监控：

```bash
# 首先登录WandB (如果还没有)
wandb login

# 开始训练
python start_training.py --config production
```

### 方法3: 从检查点恢复

如果训练中断，可以从检查点恢复：

```bash
python start_training.py --config production --resume output/checkpoints/latest.pt
```

## 📊 训练监控

### 1. 终端输出
训练过程中会显示：
- 每个epoch的训练/验证损失
- 生成的样本序列
- 训练进度和预计剩余时间

### 2. Tensorboard监控
```bash
# 在另一个终端启动Tensorboard
tensorboard --logdir output/tensorboard
# 然后在浏览器打开 http://localhost:6006
```

### 3. WandB监控 (如果启用)
- 自动上传到WandB云端
- 实时查看损失曲线
- 比较不同实验结果

## 📁 输出文件结构

训练过程中会生成以下文件：

```
output/
├── checkpoints/
│   ├── latest.pt          # 最新检查点
│   ├── best.pt           # 最佳模型
│   └── epoch_*.pt        # 定期保存的检查点
├── logs/
│   └── training_*.log    # 训练日志
├── tensorboard/          # Tensorboard日志
└── generated_samples/    # 生成的样本序列
```

## ⚡ 性能优化建议

### 硬件要求
- **最低配置**: 8GB RAM, CPU训练
- **推荐配置**: 16GB+ RAM, GPU (8GB+ VRAM)
- **理想配置**: 32GB+ RAM, 多GPU

### 训练加速技巧

1. **使用GPU** (如果可用):
   修改 `main_trainer.py` 第39行：
   ```python
   # 将这行
   self.device = torch.device("cpu")
   # 改为
   self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **调整批次大小**:
   根据显存大小调整 `config.data.batch_size`

3. **启用混合精度**:
   生产配置已默认启用

## 🔍 训练过程解析

### 双路径训练机制

1. **扩散模型主路径**:
   - 学习从噪声生成抗菌肽序列
   - 使用D3PM离散扩散算法
   - 目标：生成有效的氨基酸序列

2. **ESM-2辅助路径**:
   - 提取蛋白质语言特征
   - 对比学习区分阴性菌vs阳性菌特异性
   - 目标：增强序列的生物学意义

### 关键训练指标

- **扩散损失**: 序列重构质量
- **对比损失**: 特异性区分能力
- **验证损失**: 模型泛化能力
- **生成质量**: 序列有效性和多样性

## 🎯 预期训练结果

### 训练收敛指标
- 扩散损失: 从 ~3.0 降至 ~1.5
- 对比损失: 从 ~2.0 降至 ~0.8
- 验证损失: 稳定在训练损失附近

### 生成质量提升
- 序列有效性: >95%
- 抗菌活性预测: >70%
- 序列多样性: 高
- 革兰氏阴性菌特异性: 显著提升

## 🚨 常见问题与解决方案

### 1. 内存不足
```bash
# 减小批次大小
# 在config中修改 batch_size = 16 或 8
```

### 2. 训练速度慢
```bash
# 启用GPU训练
# 减少序列最大长度
# 使用更小的ESM-2模型
```

### 3. 损失不收敛
```bash
# 降低学习率
# 增加warmup步数
# 检查数据质量
```

### 4. 生成质量差
```bash
# 增加训练轮数
# 调整采样温度
# 使用不同采样策略
```

## 📞 获取帮助

如果遇到问题：
1. 查看训练日志: `output/logs/training_*.log`
2. 检查系统状态: `python start_training.py --test-only`
3. 查看详细错误信息和堆栈跟踪

---

**准备好开始你的完整训练之旅了吗？🎯**

使用以下命令开始：
```bash
cd enhanced_architecture
python start_training.py --config production
```
