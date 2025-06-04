#!/bin/bash

# 清理并重新训练脚本
# 用于服务器环境的完全重新训练

echo "🧹 清理之前的训练结果..."

# 备份之前的结果（可选）
if [ -d "output" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    echo "📦 备份之前的结果到 output_backup_$timestamp"
    mv output "output_backup_$timestamp"
fi

# 清理缓存文件
if [ -d "cache" ]; then
    echo "🗑️  清理ESM-2特征缓存..."
    rm -rf cache
fi

# 清理临时文件
echo "🗑️  清理临时文件..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ 清理完成！"
echo ""
echo "🚀 开始重新训练..."
echo "使用专门的双4090优化训练脚本"
echo ""

# 检查GPU状态
echo "📊 GPU状态检查:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "⏰ 开始训练时间: $(date)"
echo ""

# 设置CUDA优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# 开始训练 - 使用专门的双4090训练脚本
python train_dual_4090.py --config dual_4090

echo ""
echo "⏰ 训练结束时间: $(date)"