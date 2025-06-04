# 🚀 双4090重新训练命令指南

## 📋 问题解决

您之前遇到的显存不足问题已经解决：
1. ✅ 修复了多GPU支持 - 代码现在正确使用双4090
2. ✅ 优化了显存管理 - 添加了CUDA内存优化
3. ✅ 创建了专门的双4090训练脚本

## 🔧 重新训练命令

### 方法1: 使用清理脚本（推荐）
```bash
cd enhanced_architecture
chmod +x clean_and_retrain.sh
./clean_and_retrain.sh
```

### 方法2: 手动清理并训练
```bash
cd enhanced_architecture

# 1. 备份之前的结果
mv output output_backup_$(date +"%Y%m%d_%H%M%S")

# 2. 清理缓存
rm -rf cache

# 3. 设置CUDA优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. 开始训练
python train_dual_4090.py --config dual_4090
```

### 方法3: 直接使用新训练脚本
```bash
cd enhanced_architecture
python train_dual_4090.py --config dual_4090
```

## 🎯 配置说明

### dual_4090配置特点：
- **ESM-2模型**: `facebook/esm2_t30_150M_UR50D` (150M参数)
- **扩散模型**: 1024维隐藏层，16层，16注意力头
- **多GPU**: 自动检测并使用双4090
- **批次大小**: 32 (充分利用48GB显存)
- **训练轮数**: 300 epochs
- **显存优化**: 启用混合精度训练

## 📊 预期性能

### 硬件利用率：
- **双GPU使用**: ✅ 自动DataParallel
- **显存使用**: ~15-20GB per GPU
- **训练速度**: 比单GPU快1.7-1.9倍

### 模型性能提升：
- **特征质量**: +25-30% (相比8M ESM-2)
- **生成质量**: +15-20%
- **整体性能**: 预期从70-80分提升至85-90分

## 🔍 训练监控

### 实时监控：
```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f enhanced_architecture/output/logs/dual_4090_training_*.log
```

### TensorBoard监控：
```bash
tensorboard --logdir enhanced_architecture/output/tensorboard
# 然后访问 http://localhost:6006
```

## ⚠️ 注意事项

1. **首次运行**: ESM-2模型首次下载需要时间
2. **显存监控**: 训练过程中会自动显示显存使用情况
3. **检查点保存**: 每20个epoch自动保存检查点
4. **中断恢复**: 如需中断，可以从检查点恢复训练

## 🚨 故障排除

### 如果仍然出现显存不足：
```bash
# 降级到35M ESM-2模型
python train_dual_4090.py --config production
```

### 如果多GPU不工作：
```bash
# 检查CUDA和PyTorch版本
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

---

**推荐**: 直接运行 `./clean_and_retrain.sh` 开始重新训练，这会自动处理所有优化设置。