#!/bin/bash

# 高质量抗菌肽序列生成脚本
# 使用方法: ./run_generation.sh

echo "🚀 开始生成高质量抗菌肽序列..."
echo "📊 当前设置:"
echo "  - 生成批次: 20"
echo "  - 每批次数量: 100" 
echo "  - 预测阈值: 0.95"
echo "  - 序列长度: 40"
echo "  - 多样性强度: 1.5"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请确保Python已安装"
    exit 1
fi

# 检查必要文件
if [ ! -f "enhanced_architecture/output/checkpoints/best.pt" ]; then
    echo "❌ 生成模型文件未找到: enhanced_architecture/output/checkpoints/best.pt"
    exit 1
fi

if [ ! -f "model/hybrid_classifier_best_tuned.keras" ]; then
    echo "❌ 预测模型文件未找到: model/hybrid_classifier_best_tuned.keras"
    exit 1
fi

if [ ! -f "data/Gram+-.fasta" ]; then
    echo "❌ 数据库文件未找到: data/Gram+-.fasta"
    exit 1
fi

# 创建输出目录
mkdir -p generated_sequences

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="generated_sequences/high_quality_amp_${TIMESTAMP}.fasta"

echo "📝 输出文件: $OUTPUT_FILE"
echo ""

# 运行生成脚本
python generate_high_quality_sequences.py \
    --num_batches 20 \
    --batch_size 100 \
    --seq_length 40 \
    --threshold 0.95 \
    --temperature 1.0 \
    --diversity_strength 1.5 \
    --output "$OUTPUT_FILE" \
    --checkpoint "enhanced_architecture/output/checkpoints/best.pt" \
    --predict_model "model/hybrid_classifier_best_tuned.keras" \
    --scaler "model/hybrid_model_scaler.pkl" \
    --database "data/Gram+-.fasta"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 生成完成！"
    echo "📁 结果文件: $OUTPUT_FILE"
    
    # 显示文件信息
    if [ -f "$OUTPUT_FILE" ]; then
        SEQ_COUNT=$(grep -c "^>" "$OUTPUT_FILE")
        echo "📊 生成序列数量: $SEQ_COUNT"
        echo ""
        echo "📋 文件预览 (前5个序列):"
        head -10 "$OUTPUT_FILE"
    fi
else
    echo "❌ 生成失败，请检查日志文件"
fi