#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理预测结果，将高分序列添加到数据库
筛选得分≥0.9的序列，添加到database.fasta中，并标记来源
"""

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os

def process_predictions_and_update_database(predictions_file, original_fasta, database_fasta, threshold=0.9, source_tag="Amplify-syth"):
    """
    处理预测结果并更新数据库
    
    Args:
        predictions_file: 预测结果文件路径
        original_fasta: 原始FASTA文件路径 
        database_fasta: 数据库FASTA文件路径
        threshold: 阈值，默认0.9
        source_tag: 来源标记，默认"Amplify-syth"
    """
    
    # 读取预测结果
    print(f"📊 读取预测结果: {predictions_file}")
    df_predictions = pd.read_csv(predictions_file, sep='\t')
    print(f"总预测序列数: {len(df_predictions)}")
    
    # 筛选高分序列
    high_score_df = df_predictions[df_predictions['Probability'] >= threshold]
    print(f"🎯 得分≥{threshold}的序列数: {len(high_score_df)}")
    
    if len(high_score_df) == 0:
        print("❌ 没有序列达到阈值要求")
        return
    
    # 显示高分序列统计
    print("\n📈 高分序列统计:")
    for _, row in high_score_df.iterrows():
        print(f"  {row['Sequence_ID']}: {row['Probability']:.4f}")
    
    # 读取原始FASTA文件，建立ID到序列的映射
    print(f"\n🧬 读取原始序列: {original_fasta}")
    original_sequences = {}
    for record in SeqIO.parse(original_fasta, "fasta"):
        original_sequences[record.id] = str(record.seq)
    
    print(f"原始文件中的序列数: {len(original_sequences)}")
    
    # 准备要添加的高分序列
    high_score_sequences = []
    for _, row in high_score_df.iterrows():
        seq_id = row['Sequence_ID']
        probability = row['Probability']
        
        if seq_id in original_sequences:
            sequence = original_sequences[seq_id]
            # 创建新的序列ID，包含来源和得分信息
            new_id = f"{seq_id}|{source_tag}|Score_{probability:.4f}"
            
            # 创建SeqRecord对象
            seq_record = SeqRecord(
                Seq(sequence),
                id=new_id,
                description=f"Source: {source_tag}, Prediction Score: {probability:.4f}"
            )
            high_score_sequences.append(seq_record)
        else:
            print(f"⚠️  警告: 在原始文件中未找到序列 {seq_id}")
    
    print(f"\n✅ 准备添加 {len(high_score_sequences)} 个高分序列到数据库")
    
    # 检查database.fasta是否存在
    if os.path.exists(database_fasta):
        print(f"📄 数据库文件已存在: {database_fasta}")
        # 读取现有序列数量
        existing_count = len(list(SeqIO.parse(database_fasta, "fasta")))
        print(f"现有序列数: {existing_count}")
    else:
        print(f"📄 创建新的数据库文件: {database_fasta}")
        existing_count = 0
    
    # 将高分序列追加到database.fasta
    with open(database_fasta, "a") as output_handle:
        SeqIO.write(high_score_sequences, output_handle, "fasta")
    
    # 验证添加结果
    final_count = len(list(SeqIO.parse(database_fasta, "fasta")))
    added_count = final_count - existing_count
    
    print(f"\n🎉 成功完成!")
    print(f"  - 添加序列数: {added_count}")
    print(f"  - 数据库总序列数: {final_count}")
    print(f"  - 来源标记: {source_tag}")
    print(f"  - 得分阈值: {threshold}")
    
    return high_score_sequences

if __name__ == "__main__":
    # 文件路径
    predictions_file = "predictions/temp_predictions.txt"
    original_fasta = "data/temp.fasta"
    database_fasta = "data/database.fasta"
    
    print("🚀 开始处理预测结果并更新数据库")
    print("=" * 50)
    
    # 处理预测结果
    high_score_sequences = process_predictions_and_update_database(
        predictions_file=predictions_file,
        original_fasta=original_fasta,
        database_fasta=database_fasta,
        threshold=0.9,
        source_tag="Amplify-syth"
    )
    
    print("\n" + "=" * 50)
    print("✅ 任务完成!")
