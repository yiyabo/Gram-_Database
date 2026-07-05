#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
升级版：
- 提取更多抗菌肽理化属性 (带Sequence)
- 筛选并打标 (Positive/Negative)
- 导出Positive-only的FASTA
- 保存美化版直方图和筛选总结
- 新增Positive数量和比例统计
- 扩充丰富属性字段 (Aromaticity, Boman Index, Aliphatic Index, etc.)
"""

from ast import arg
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from peptides import Peptide
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# ====== 1. 提取特征 ======

def extract_features(fasta_file):
    records = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        peptide_id = record.id
        sequence = str(record.seq)

        if not sequence or any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence.upper()):
            continue

        try:
            pep = Peptide(sequence)
            charge = pep.charge(pH=7.4)

            hydrophobicity = pep.hydrophobicity(scale="Eisenberg")
            amphipathicity = pep.hydrophobic_moment(window=11)
            if amphipathicity is None: amphipathicity = 0

            # 新增属性
            hydrophilicity = pep.hydrophobicity(scale="HoppWoods")
            hydropathicity = pep.hydrophobicity(scale="KyteDoolittle")
            aliphatic_index = pep.aliphatic_index()
            instability_index = pep.instability_index()
            cysteine_content = sequence.upper().count('C') / len(sequence)

            records.append({
                'ID': peptide_id,
                'Sequence': sequence,
                'Length': len(sequence),
                # 'pI': round(pI, 2),
                'Charge': round(charge, 2), #电荷
                'Hydrophobicity': round(hydrophobicity, 2), # 平均疏水性（Eisenberg尺度）
                'Hydrophobic_Moment': round(amphipathicity, 2), # 两性性（Hydrophobic Moment）
                'Hydrophilicity': round(hydrophilicity, 2), # 平均亲水性（Hopp-Woods尺度）
                'Hydropathicity': round(hydropathicity, 2), # 疏水性指数（Kyte-Doolittle尺度）
                'Aliphatic_Index': round(aliphatic_index, 2), # 脂肪族氨基酸指数
                'Instability_Index': round(instability_index, 2), # 蛋白质稳定性预测
                'Cysteine_Content': round(cysteine_content, 4) # Cysteine含量百分比
            })

        except Exception as e:
            print(f"Error extracting features for {peptide_id}: {e}")

    return pd.DataFrame(records)


# ====== 3. 绘制直方图 ======

def plot_distribution(df, column, output_dir):
    plt.figure(figsize=(8,6))
    sns.histplot(df[column], kde=True, color='skyblue', edgecolor='black', bins=50)
    plt.title(f'{column} Distribution', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"))
    plt.close()

# ====== 5. 统计P5-P95范围 ======

def summarize_ranges(df, output_dir):
    summary_lines = []
    for column in ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment', 'Hydrophilicity', 'Hydropathicity', 'Aliphatic_Index', 'Instability_Index', 'Cysteine_Content']:
        p5 = np.percentile(df[column], 5)
        p95 = np.percentile(df[column], 95)
        summary_lines.append(f"{column} range (P5–P95): {round(p5,2)} to {round(p95,2)}")

    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        for line in summary_lines:
            f.write(line + "\n")

    print("\n".join(summary_lines))

# ====== 6. 统计序列信息 ======

def summarize_statistics(df, output_dir):
    total = len(df)
    
    summary = [
        f"Total sequences after processing: {total}",
        f"Average length: {round(df['Length'].mean(), 2)}",
        f"Average charge: {round(df['Charge'].mean(), 2)}",
        f"Average hydrophobicity: {round(df['Hydrophobicity'].mean(), 2)}",
        f"Average hydrophobic moment: {round(df['Hydrophobic_Moment'].mean(), 2)}",
        f"Average hydrophilicity: {round(df['Hydrophilicity'].mean(), 2)}",
        f"Average hydropathicity: {round(df['Hydropathicity'].mean(), 2)}",
        f"Average aliphatic index: {round(df['Aliphatic_Index'].mean(), 2)}",
        f"Average instability index: {round(df['Instability_Index'].mean(), 2)}",
        f"Average cysteine content: {round(df['Cysteine_Content'].mean(), 4)}",
    ]

    with open(os.path.join(output_dir, "sequence_statistics.txt"), 'w') as f:
        for line in summary:
            f.write(line + "\n")

    print("\n".join(summary))

# ====== 7. 主程序入口 ======

def main(args):
    print(f"Processing FASTA file: {args.input}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = extract_features(args.input)
    print(f"Total valid sequences: {len(df)}")

    plot_distribution(df, 'Charge', args.output_dir)
    plot_distribution(df, 'Hydrophobicity', args.output_dir)
    plot_distribution(df, 'Hydrophobic_Moment', args.output_dir)
    plot_distribution(df, 'Hydrophilicity', args.output_dir)
    plot_distribution(df, 'Hydropathicity', args.output_dir)
    plot_distribution(df, 'Aliphatic_Index', args.output_dir)
    plot_distribution(df, 'Instability_Index', args.output_dir)
    plot_distribution(df, 'Cysteine_Content', args.output_dir)
    summarize_ranges(df, args.output_dir)

    df_processed = df
    df_processed.to_csv(os.path.join(args.output_dir, "processed_attributes.csv"), index=False)
    print(f"Processed attributes saved to: {os.path.join(args.output_dir, 'processed_attributes.csv')}")
    summarize_statistics(df_processed, args.output_dir)

    print("Pipeline completed.")

# ====== 8. 命令行参数 ======

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full AMP Screening Pipeline with extended property extraction.")
    parser.add_argument('--input', required=True, help='Input FASTA file path')
    parser.add_argument('--output_dir', required=True, help='Output directory path')

    args = parser.parse_args()

    main(args)
