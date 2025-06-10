# 高质量抗菌肽序列生成器

## 功能说明

这个独立的生成系统可以：

1. **智能生成** - 使用训练好的扩散模型和ESM-2编码器生成抗菌肽序列
2. **质量筛选** - 用混合预测模型筛选出概率 > 0.95 的高质量序列
3. **智能去重** - 去除重复序列，包括与现有数据库的重复
4. **批量处理** - 支持大规模批量生成和筛选

## 文件说明

- `generate_high_quality_sequences.py` - 主生成脚本
- `test_generation_setup.py` - 环境检查脚本
- `run_generation.sh` - 一键运行脚本
- `GENERATION_README.md` - 本说明文档

## 使用方法

### 1. 环境检查

首先运行环境检查脚本：

```bash
python test_generation_setup.py
```

确保所有依赖和文件都正确配置。

### 2. 快速开始

使用默认参数生成序列：

```bash
# 方法1: 使用bash脚本 (推荐)
chmod +x run_generation.sh
./run_generation.sh

# 方法2: 直接运行Python脚本
python generate_high_quality_sequences.py
```

### 3. 自定义参数

```bash
python generate_high_quality_sequences.py \
    --num_batches 20 \        # 生成批次数
    --batch_size 100 \        # 每批次序列数量
    --seq_length 40 \         # 序列长度
    --threshold 0.95 \        # 预测概率阈值
    --temperature 1.0 \       # 采样温度 (控制随机性)
    --diversity_strength 1.5 \ # 多样性强度
    --output "my_sequences.fasta"  # 输出文件名
```

## 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_batches` | 10 | 生成批次数，总序列数 = 批次数 × 批次大小 |
| `--batch_size` | 50 | 每批次生成的序列数量 |
| `--seq_length` | 40 | 生成序列的长度 |
| `--threshold` | 0.95 | 预测概率阈值，只保留 ≥ 此值的序列 |
| `--temperature` | 1.0 | 采样温度，越高越随机 |
| `--diversity_strength` | 1.5 | 多样性强度，越高生成的序列越多样 |
| `--output` | 自动生成 | 输出FASTA文件路径 |

## 输出结果

### 生成的文件

1. **FASTA文件** - 包含筛选后的高质量序列
2. **日志文件** - 详细的生成过程日志

### FASTA格式示例

```
>Generated_AMP_000001
Probability: 0.9876 | Length: 40 | Generated: 2025-06-10 15:30:45
KLWLRLLWQVLVQVLVQLWVKLWLVQVLVQLWVKLWLVQV

>Generated_AMP_000002  
Probability: 0.9654 | Length: 38 | Generated: 2025-06-10 15:30:45
RVWQRVWQRVWQRVWQRVWQRVWQRVWQRVWQRVWQRV
```

## 性能优化

### 推荐硬件配置

- **内存**: 至少 8GB RAM
- **GPU**: NVIDIA GPU (可选，但推荐)
- **存储**: 至少 5GB 可用空间

### 批量大小建议

| 硬件配置 | 推荐批次大小 | 预期速度 |
|----------|--------------|----------|
| CPU only | 20-50 | 慢 |
| GPU + 8GB RAM | 50-100 | 中等 |
| GPU + 16GB+ RAM | 100-200 | 快 |

## 质量控制

### 筛选标准

1. **长度检查** - 序列长度 > 5 个氨基酸
2. **有效性检查** - 只包含标准氨基酸
3. **预测筛选** - 预测概率 ≥ 阈值
4. **去重检查** - 去除重复序列

### 预期产出

以默认参数运行 (10批次 × 50序列):
- **生成序列**: ~500个
- **有效序列**: ~400个 (80%)
- **高质量序列**: ~20-50个 (5-10%)
- **最终唯一序列**: ~15-40个

## 故障排除

### 常见问题

1. **导入错误**
   ```
   解决: pip install torch tensorflow scikit-learn biopython peptides transformers
   ```

2. **文件未找到**
   ```
   解决: 确保在项目根目录运行，检查模型文件路径
   ```

3. **内存不足**
   ```
   解决: 减小batch_size参数
   ```

4. **GPU相关错误**
   ```
   解决: 确保CUDA正确安装，或使用CPU模式
   ```

### 日志级别

- **INFO**: 正常进度信息
- **WARNING**: 非关键警告
- **ERROR**: 错误信息
- **DEBUG**: 详细调试信息

## 高级用法

### 大规模生成

生成大量序列时的建议：

```bash
# 生成10000个候选序列，期望得到500-1000个高质量序列
python generate_high_quality_sequences.py \
    --num_batches 100 \
    --batch_size 100 \
    --threshold 0.92 \
    --output "large_scale_amp.fasta"
```

### 参数调优

- **提高质量**: 增加 `threshold` (0.96-0.99)
- **增加多样性**: 增加 `diversity_strength` (1.5-2.5)
- **控制长度**: 调整 `seq_length` (20-60)
- **平衡质量和数量**: 调整 `temperature` (0.8-1.2)

## 注意事项

1. **运行时间** - 大规模生成可能需要数小时
2. **磁盘空间** - 确保有足够空间存储结果
3. **网络连接** - 首次运行需要下载ESM-2模型
4. **随机性** - 每次运行结果会略有不同

## 技术原理

1. **生成模型** - 基于D3PM扩散模型的序列生成
2. **辅助编码** - ESM-2蛋白质语言模型提供语义特征
3. **质量预测** - 混合LSTM+MLP模型进行活性预测
4. **多样采样** - 使用diverse sampling增加序列多样性

---

如有问题，请检查日志文件或联系开发者。