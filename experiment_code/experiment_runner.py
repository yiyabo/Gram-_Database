#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实验执行脚本：按照EXPERIMENT_PLAN.md执行所有实验
生成论文所需的图表、表格和分析结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import Counter
import logging
from datetime import datetime
import json
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 使用更兼容的样式设置
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        print("使用默认matplotlib样式")

sns.set_palette("husl")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """实验执行器"""
    
    def __init__(self):
        """初始化实验执行器"""
        self.results_dir = Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "tables").mkdir(exist_ok=True)
        (self.results_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"实验结果将保存到: {self.results_dir}")
        
        # 数据文件路径
        self.data_paths = {
            'gram_neg': 'data/Gram-.fasta',
            'gram_pos': 'data/Gram+.fasta', 
            'gram_both': 'data/Gram+-.fasta',
            'features': 'data/peptide_features.csv'
        }
        
        # 验证文件存在
        for name, path in self.data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据文件不存在: {path}")
        
        logger.info("所有数据文件验证通过")
    
    def analyze_fasta_file(self, fasta_path, label):
        """分析单个FASTA文件"""
        sequences = []
        lengths = []
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            # 只保留标准氨基酸
            if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
                sequences.append(seq)
                lengths.append(len(seq))
        
        stats = {
            'label': label,
            'count': len(sequences),
            'mean_length': np.mean(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0
        }
        
        return stats, sequences, lengths
    
    def section1_experimental_setup(self):
        """第1部分：实验设置 - 数据集统计和可视化"""
        logger.info("=" * 60)
        logger.info("执行第1部分：实验设置 (Experimental Setup)")
        logger.info("=" * 60)
        
        # 1.1 数据集统计
        logger.info("1.1 分析数据集统计信息...")
        
        dataset_stats = []
        all_lengths = []
        all_labels = []
        
        # 分析每个数据集
        for key, path in [('gram_neg', self.data_paths['gram_neg']),
                         ('gram_pos', self.data_paths['gram_pos']),
                         ('gram_both', self.data_paths['gram_both'])]:
            
            label_map = {
                'gram_neg': 'Gram- only',
                'gram_pos': 'Gram+ only', 
                'gram_both': 'Gram+/-'
            }
            
            stats, sequences, lengths = self.analyze_fasta_file(path, label_map[key])
            dataset_stats.append(stats)
            
            # 收集所有长度用于整体分析
            all_lengths.extend(lengths)
            all_labels.extend([label_map[key]] * len(lengths))
            
            logger.info(f"{label_map[key]}: {stats['count']} 条序列, "
                       f"平均长度: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        
        # 保存统计表格
        stats_df = pd.DataFrame(dataset_stats)
        stats_df.to_csv(self.results_dir / "tables" / "dataset_statistics.csv", index=False)
        logger.info("数据集统计表格已保存")
        
        # 1.2 序列长度分布可视化
        logger.info("1.2 生成序列长度分布图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 整体长度分布直方图
        axes[0, 0].hist(all_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Sequence Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 按类别分组的长度分布
        length_df = pd.DataFrame({'Length': all_lengths, 'Category': all_labels})
        sns.boxplot(data=length_df, x='Category', y='Length', ax=axes[0, 1])
        axes[0, 1].set_title('Sequence Length Distribution by Category')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 长度分布小提琴图
        sns.violinplot(data=length_df, x='Category', y='Length', ax=axes[1, 0])
        axes[1, 0].set_title('Sequence Length Distribution (Violin Plot)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 累积分布函数
        for category in length_df['Category'].unique():
            cat_lengths = length_df[length_df['Category'] == category]['Length']
            sorted_lengths = np.sort(cat_lengths)
            y = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
            axes[1, 1].plot(sorted_lengths, y, label=category, linewidth=2)
        
        axes[1, 1].set_xlabel('Sequence Length')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / "sequence_length_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 1.3 氨基酸组成分析
        logger.info("1.3 分析氨基酸组成...")
        
        # 收集所有序列的氨基酸组成
        aa_composition = Counter()
        total_residues = 0
        
        for path in [self.data_paths['gram_neg'], self.data_paths['gram_pos'], self.data_paths['gram_both']]:
            for record in SeqIO.parse(path, "fasta"):
                seq = str(record.seq).upper()
                if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
                    aa_composition.update(seq)
                    total_residues += len(seq)
        
        # 计算氨基酸频率
        aa_freq = {aa: count/total_residues for aa, count in aa_composition.items()}
        
        # 绘制氨基酸组成图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 氨基酸频率柱状图
        amino_acids = sorted(aa_freq.keys())
        frequencies = [aa_freq[aa] * 100 for aa in amino_acids]  # 转换为百分比
        
        bars = ax1.bar(amino_acids, frequencies, color='lightcoral', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Amino Acid')
        ax1.set_ylabel('Frequency (%)')
        ax1.set_title('Amino Acid Composition in All Sequences')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, freq in zip(bars, frequencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{freq:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 氨基酸理化性质分类
        hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
        polar = ['S', 'T', 'N', 'Q']
        charged_pos = ['K', 'R', 'H']
        charged_neg = ['D', 'E']
        special = ['C', 'G', 'P']
        
        categories = []
        for aa in amino_acids:
            if aa in hydrophobic:
                categories.append('Hydrophobic')
            elif aa in polar:
                categories.append('Polar')
            elif aa in charged_pos:
                categories.append('Positive')
            elif aa in charged_neg:
                categories.append('Negative')
            elif aa in special:
                categories.append('Special')
            else:
                categories.append('Other')
        
        # 按理化性质分组的饼图
        category_freq = {}
        for aa, cat in zip(amino_acids, categories):
            category_freq[cat] = category_freq.get(cat, 0) + aa_freq[aa] * 100
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        wedges, texts, autotexts = ax2.pie(category_freq.values(), labels=category_freq.keys(),
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Amino Acid Distribution by Physicochemical Properties')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "figures" / "amino_acid_composition.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存氨基酸组成数据
        aa_comp_df = pd.DataFrame([
            {'Amino_Acid': aa, 'Frequency_Percent': freq * 100, 'Category': cat}
            for aa, freq, cat in zip(amino_acids, [aa_freq[aa] for aa in amino_acids], categories)
        ])
        aa_comp_df.to_csv(self.results_dir / "tables" / "amino_acid_composition.csv", index=False)
        
        logger.info("第1部分完成：实验设置")
        
        return {
            'dataset_stats': stats_df,
            'length_distribution': length_df,
            'aa_composition': aa_comp_df,
            'total_sequences': sum(stats['count'] for stats in dataset_stats),
            'total_residues': total_residues
        }
    
    def section2_classifier_evaluation(self):
        """第2部分：混合分类器性能评估"""
        logger.info("=" * 60)
        logger.info("执行第2部分：混合分类器性能评估")
        logger.info("=" * 60)
        
        # 检查是否存在训练好的模型
        model_path = "model/hybrid_classifier_best_tuned.keras"
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            logger.info("请先运行 hybrid_classifier.py 训练模型")
            return None
        
        # 加载特征数据
        if os.path.exists(self.data_paths['features']):
            features_df = pd.read_csv(self.data_paths['features'])
            logger.info(f"加载特征数据: {len(features_df)} 条记录")
            
            # 分析特征分布
            feature_cols = [col for col in features_df.columns 
                           if col not in ['ID', 'Sequence', 'Label', 'Source']]
            
            logger.info(f"特征维度: {len(feature_cols)}")
            
            # 生成特征相关性热图
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 选择主要特征进行可视化
            main_features = ['Length', 'Charge', 'Hydrophobicity', 'Hydrophobic_Moment',
                           'Instability_Index', 'Isoelectric_Point', 'Aliphatic_Index']
            
            if all(feat in features_df.columns for feat in main_features):
                corr_matrix = features_df[main_features].corr()
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax, fmt='.2f')
                ax.set_title('Feature Correlation Matrix')
                
                plt.tight_layout()
                plt.savefig(self.results_dir / "figures" / "feature_correlation.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("特征相关性分析完成")
            
            # 按标签分析特征分布
            if 'Label' in features_df.columns:
                positive_data = features_df[features_df['Label'] == 1]
                negative_data = features_df[features_df['Label'] == 0]
                
                logger.info(f"正例数量: {len(positive_data)}, 负例数量: {len(negative_data)}")
                
                # 生成特征分布对比图
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()
                
                for i, feature in enumerate(main_features):
                    if i < len(axes) and feature in features_df.columns:
                        ax = axes[i]
                        
                        # 绘制分布直方图
                        ax.hist(positive_data[feature].dropna(), bins=20, alpha=0.6, 
                               label=f'Anti-Gram-negative (n={len(positive_data)})', 
                               color='green', density=True)
                        ax.hist(negative_data[feature].dropna(), bins=20, alpha=0.6,
                               label=f'Non-Anti-Gram-negative (n={len(negative_data)})', 
                               color='red', density=True)
                        
                        ax.set_xlabel(feature)
                        ax.set_ylabel('Density')
                        ax.set_title(f'{feature} Distribution')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                
                # 隐藏多余的子图
                for i in range(len(main_features), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(self.results_dir / "figures" / "feature_distribution_comparison.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("特征分布对比分析完成")
        
        logger.info("第2部分完成：分类器评估（基础分析）")
        return {'features_analyzed': True}
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("开始执行完整实验流程...")
        
        results = {}
        
        try:
            # 第1部分：实验设置
            results['section1'] = self.section1_experimental_setup()
            
            # 第2部分：分类器评估
            results['section2'] = self.section2_classifier_evaluation()
            
            # 保存实验结果摘要
            summary = {
                'experiment_date': datetime.now().isoformat(),
                'total_sequences': results['section1']['total_sequences'],
                'total_residues': results['section1']['total_residues'],
                'sections_completed': ['section1', 'section2']
            }
            
            with open(self.results_dir / "experiment_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("=" * 60)
            logger.info("实验执行完成！")
            logger.info(f"结果保存在: {self.results_dir}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"实验执行出错: {e}")
            raise

def main():
    """主函数"""
    try:
        runner = ExperimentRunner()
        results = runner.run_all_experiments()
        
        print("\n" + "="*60)
        print("实验执行成功！")
        print(f"总序列数: {results['section1']['total_sequences']}")
        print(f"总氨基酸残基数: {results['section1']['total_residues']}")
        print(f"结果保存在: {runner.results_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"实验执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())