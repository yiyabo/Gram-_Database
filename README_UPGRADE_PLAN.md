# 🚀 Gram-_Database 升级计划：VAE → 扩散模型 + ESM-2增强

## 📋 项目概述

本项目旨在将现有的双路径VAE抗菌肽生成模型升级为**扩散模型 + ESM-2增强辅助路径**的新一代架构，预期将性能从当前的70-90分提升至90分。

## 🎯 核心升级目标

- **主路径**：VAE → 离散扩散模型 (D3PM/CDCD)
- **辅助路径**：从头训练Transformer → ESM-2预训练模型 + 多尺度增强
- **性能目标**：70-80分 → 90分
- **保持优势**：双路径架构、全局特征融合、k-mer分词

## 🏗️ 技术架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    新架构设计                                │
├─────────────────────────────────────────────────────────────┤
│ Phase-A: ESM-2增强辅助路径                                   │
│ ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│ │ 辅助序列        │→│ ESM-2预训练   │→│ 多尺度CNN融合    │  │
│ │ (natureAMP.txt) │  │ + 注意力池化  │  │ + 全局特征聚类   │  │
│ └─────────────────┘  └──────────────┘  └─────────────────┘  │
│                                                ↓             │
│ Phase-B: 扩散模型主路径                     全局特征         │
│ ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│ │ 主训练序列      │→│ 离散扩散模型   │←│ 条件注入机制     │  │
│ │ (Train_Main.txt)│  │ (D3PM/CDCD)   │  │ (全局特征引导)   │  │
│ └─────────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📅 分阶段实施计划

### 🚀 阶段1：ESM-2辅助路径升级 (第1-2周)

**目标**: 用ESM-2替换现有从头训练的Transformer编码器

#### 第1周任务
- [ ] **环境准备**
  - 安装transformers库: `pip install transformers torch`
  - 测试ESM-2模型加载和推理
  - 验证与现有代码的兼容性

- [ ] **核心开发**
  - 创建 `esm2_auxiliary_encoder.py`
  - 实现ESM-2特征提取函数
  - 替换 `TransformerAE.py` 中的 `train_aux_encoder` 函数

#### 第2周任务
- [ ] **集成测试**
  - 修改 `TransformerAE.py` 主流程
  - 测试ESM-2提取的全局特征质量
  - 对比ESM-2 vs 原始Transformer的特征表示效果

**预期成果**: ESM-2版本的辅助特征提取器，特征质量显著提升

### 🔄 阶段2：扩散模型主路径开发 (第3-5周)

**目标**: 实现D3PM离散扩散模型替换VAE

#### 第3周任务
- [ ] **扩散模型基础**
  - 研究并选择扩散模型框架 (推荐: D3PM)
  - 创建 `diffusion_models.py` 模块
  - 实现基础的噪声添加和去噪过程

#### 第4周任务
- [ ] **架构实现**
  - 实现条件扩散UNet网络
  - 设计全局特征条件注入机制
  - 创建 `protein_diffusion.py` 主模型文件

#### 第5周任务
- [ ] **训练和优化**
  - 实现扩散模型训练循环
  - 集成现有的数据预处理pipeline
  - 初步训练和调试

**预期成果**: 可训练的条件扩散模型，初步生成效果

### ⚡ 阶段3：系统整合和优化 (第6-7周)

**目标**: 完整集成双路径架构并优化性能

#### 第6周任务
- [ ] **端到端集成**
  - 创建 `enhanced_protein_generator.py` 主文件
  - 集成ESM-2辅助路径 + 扩散主路径
  - 实现完整的训练和推理pipeline

#### 第7周任务
- [ ] **性能优化**
  - 超参数调优
  - 推理速度优化 (DDIM/DPM-Solver)
  - 批量生成和长度自适应

**预期成果**: 完整的新架构系统，性能达到预期目标

### 🎯 阶段4：高级功能和部署 (第8周+)

- [ ] **高级功能**
  - 多尺度CNN特征融合
  - 对比学习增强
  - 生物学约束和后处理

- [ ] **评估和部署**
  - 全面性能评估
  - 与现有VAE系统对比
  - 更新预测和生成脚本

## 📁 新增文件结构

```
enhanced_architecture/
├── esm2_auxiliary_encoder.py      # ESM-2辅助编码器
├── diffusion_models/
│   ├── __init__.py
│   ├── d3pm.py                    # D3PM扩散模型实现
│   ├── schedulers.py              # 噪声调度器
│   └── utils.py                   # 扩散模型工具函数
├── protein_diffusion.py          # 蛋白质扩散主模型
├── enhanced_protein_generator.py # 新架构主文件
├── config/
│   ├── esm2_config.py            # ESM-2配置
│   └── diffusion_config.py       # 扩散模型配置
└── evaluation/
    ├── compare_architectures.py  # 架构对比评估
    └── metrics.py                # 评估指标
```

## 🔧 技术选型说明

### ESM-2模型选择
```python
# 推荐配置 (平衡性能和效率)
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"  # 8M参数
# 备选项
# ESM_MODEL = "facebook/esm2_t12_35M_UR50D"  # 35M参数 (更强但更慢)
```

### 扩散模型选择
```python
# 主要候选
DIFFUSION_TYPE = "D3PM"      # 离散扩散 (推荐)
# DIFFUSION_TYPE = "CDCD"    # 条件离散扩散 (备选)
# DIFFUSION_TYPE = "DiT"     # Diffusion Transformer (高级选项)
```

## 📊 预期性能提升

| 组件 | 当前架构 | 新架构 | 预期提升 |
|------|----------|--------|----------|
| 辅助特征提取 | 从头训练Transformer | ESM-2预训练 | +15-20分 |
| 主生成模型 | β-VAE | 离散扩散模型 | +10-15分 |
| 整体性能 | 70-80分 | **90分** | **+10-20分** |

## 🎮 快速开始

### 1. 环境准备
```bash
# 安装新依赖
pip install transformers torch accelerate
pip install diffusers  # 如果使用HuggingFace的扩散模型库

# 验证ESM-2可用性
python -c "from transformers import EsmModel; print('ESM-2 ready!')"
```

### 2. 运行ESM-2测试
```bash
# 测试ESM-2特征提取
python enhanced_architecture/esm2_auxiliary_encoder.py --test
```

### 3. 开始升级
```bash
# 阶段1：ESM-2升级
python enhanced_architecture/enhanced_protein_generator.py --phase esm2_upgrade

# 阶段2：扩散模型开发
python enhanced_architecture/enhanced_protein_generator.py --phase diffusion_dev
```

## 📈 监控和评估

### 关键指标
- **生成质量**: 序列有效性、多样性、新颖性
- **条件遵循**: 全局特征引导效果
- **训练效率**: 收敛速度、稳定性
- **推理速度**: 生成时间、内存占用

### 对比基线
- 现有VAE系统 (70-80分)
- 简单Transformer生成器
- 其他开源AMP生成模型

## 🚨 风险评估和缓解

### 主要风险
1. **计算资源需求增加**: ESM-2 + 扩散模型
   - 缓解: 使用较小的ESM-2版本，优化批处理
2. **推理速度变慢**: 多步扩散过程
   - 缓解: DDIM加速采样，模型蒸馏
3. **架构复杂度增加**: 调试和维护难度
   - 缓解: 模块化设计，充分测试

### 回退方案
- 保留现有VAE系统作为backup
- 分阶段升级，每个阶段可独立验证
- 渐进式性能提升，降低风险

## 🤝 开发规范

### 代码规范
- 保持与现有代码风格一致
- 充分的注释和文档
- 模块化设计，便于测试和维护

### 版本控制
- 为每个阶段创建独立分支
- 重要milestone打tag
- 定期备份模型检查点

---

## 📞 联系信息

- **项目负责人**: Xinxiang Wang
- **技术问题**: 在issue中讨论
- **进度更新**: 每周更新此README
