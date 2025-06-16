#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实验第3部分：特征分析与模型可解释性 (SHAP分析)
生成SHAP特征重要性分析和可解释性图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# 导入SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("警告: SHAP库未安装，将跳过SHAP分析")
    print("请运行: pip install shap")
    SHAP_AVAILABLE = False

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """SHAP特征重要性分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.results_dir = Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 创建SHAP结果目录
        self.shap_dir = self.results_dir / "shap_analysis"
        self.shap_dir.mkdir(exist_ok=True)
        
        logger.info(f"SHAP分析结果将保存到: {self.shap_dir}")
    
    def load_and_prepare_data(self):
        """加载和准备数据"""
        logger.info("加载特征数据...")
        
        # 加载特征数据
        features_path = "data/peptide_features.csv"
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"特征文件不存在: {features_path}")
        
        df = pd.read_csv(features_path)
        logger.info(f"加载了 {len(df)} 条记录")
        
        # 选择特征列
        feature_cols = [col for col in df.columns 
                       if col not in ['ID', 'Sequence', 'Label', 'Source']]
        
        # 处理缺失值
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # 分离特征和标签
        X = df[feature_cols].values
        y = df['Label'].values
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        logger.info(f"正例比例: 训练集 {np.mean(y_train):.3f}, 测试集 {np.mean(y_test):.3f}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': scaler,
            'original_df': df
        }
    
    def train_interpretable_models(self, data):
        """训练可解释的模型"""
        logger.info("训练可解释模型...")
        
        models = {}
        
        # 1. 随机森林
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(data['X_train'], data['y_train'])
        models['random_forest'] = rf_model
        
        # 2. 逻辑回归
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(data['X_train'], data['y_train'])
        models['logistic_regression'] = lr_model
        
        # 评估模型性能
        for name, model in models.items():
            train_score = model.score(data['X_train'], data['y_train'])
            test_score = model.score(data['X_test'], data['y_test'])
            logger.info(f"{name}: 训练准确率 {train_score:.3f}, 测试准确率 {test_score:.3f}")
        
        return models
    
    def analyze_feature_importance(self, data, models):
        """分析特征重要性"""
        logger.info("分析特征重要性...")
        
        feature_importance_results = {}
        
        # 1. 随机森林特征重要性
        rf_importance = models['random_forest'].feature_importances_
        feature_importance_results['random_forest'] = rf_importance
        
        # 2. 逻辑回归系数
        lr_coef = np.abs(models['logistic_regression'].coef_[0])
        feature_importance_results['logistic_regression'] = lr_coef
        
        # 3. 排列重要性
        perm_importance = permutation_importance(
            models['random_forest'], 
            data['X_test'], 
            data['y_test'],
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        feature_importance_results['permutation'] = perm_importance.importances_mean
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': data['feature_names'],
            'RandomForest': rf_importance,
            'LogisticRegression': lr_coef,
            'Permutation': perm_importance.importances_mean
        })
        
        # 计算平均重要性排名
        importance_df['Average_Rank'] = (
            importance_df['RandomForest'].rank(ascending=False) +
            importance_df['LogisticRegression'].rank(ascending=False) +
            importance_df['Permutation'].rank(ascending=False)
        ) / 3
        
        importance_df = importance_df.sort_values('Average_Rank')
        
        # 保存结果
        importance_df.to_csv(self.shap_dir / "feature_importance_comparison.csv", index=False)
        
        return importance_df
    
    def perform_shap_analysis(self, data, models):
        """执行SHAP分析"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP库不可用，跳过SHAP分析")
            return None
        
        logger.info("执行SHAP分析...")
        
        shap_results = {}
        
        # 1. 随机森林SHAP分析
        logger.info("分析随机森林模型...")
        try:
            rf_explainer = shap.TreeExplainer(models['random_forest'])
            rf_shap_values = rf_explainer.shap_values(data['X_test'])
            
            # 如果是二分类，取正类的SHAP值
            if isinstance(rf_shap_values, list):
                rf_shap_values = rf_shap_values[1]  # 正类
            
            # 确保形状正确
            if len(rf_shap_values.shape) > 2:
                rf_shap_values = rf_shap_values[:, :, 0] if rf_shap_values.shape[2] == 1 else rf_shap_values[:, :, 1]
            
            shap_results['random_forest'] = {
                'explainer': rf_explainer,
                'shap_values': rf_shap_values
            }
            logger.info(f"随机森林SHAP值形状: {rf_shap_values.shape}")
            
        except Exception as rf_error:
            logger.warning(f"随机森林SHAP分析失败: {rf_error}")
            shap_results['random_forest'] = None
        
        # 2. 逻辑回归SHAP分析
        logger.info("分析逻辑回归模型...")
        try:
            lr_explainer = shap.LinearExplainer(models['logistic_regression'], data['X_train'])
            lr_shap_values = lr_explainer.shap_values(data['X_test'])
            
            # 确保形状正确
            if len(lr_shap_values.shape) > 2:
                lr_shap_values = lr_shap_values[:, :, 0] if lr_shap_values.shape[2] == 1 else lr_shap_values[:, :, 1]
            
            shap_results['logistic_regression'] = {
                'explainer': lr_explainer,
                'shap_values': lr_shap_values
            }
            logger.info(f"逻辑回归SHAP值形状: {lr_shap_values.shape}")
            
        except Exception as lr_error:
            logger.warning(f"逻辑回归SHAP分析失败: {lr_error}")
            shap_results['logistic_regression'] = None
        
        return shap_results
    
    def generate_shap_plots(self, data, shap_results):
        """生成SHAP可视化图表"""
        if not SHAP_AVAILABLE or not shap_results:
            return
        
        logger.info("生成SHAP可视化图表...")
        
        # 1. 特征重要性摘要图 (随机森林)
        if shap_results.get('random_forest') is not None:
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_results['random_forest']['shap_values'],
                    data['X_test'],
                    feature_names=data['feature_names'],
                    plot_type="bar",
                    show=False
                )
                plt.title('SHAP Feature Importance (Random Forest)')
                plt.tight_layout()
                plt.savefig(self.shap_dir / "shap_feature_importance_rf.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"随机森林SHAP特征重要性图生成失败: {e}")
        
        # 2. SHAP摘要图 (随机森林)
        if shap_results.get('random_forest') is not None:
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_results['random_forest']['shap_values'],
                    data['X_test'],
                    feature_names=data['feature_names'],
                    show=False
                )
                plt.title('SHAP Summary Plot (Random Forest)')
                plt.tight_layout()
                plt.savefig(self.shap_dir / "shap_summary_plot_rf.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"随机森林SHAP摘要图生成失败: {e}")
        
        # 3. 逻辑回归SHAP摘要图
        if shap_results.get('logistic_regression') is not None:
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_results['logistic_regression']['shap_values'],
                    data['X_test'],
                    feature_names=data['feature_names'],
                    plot_type="bar",
                    show=False
                )
                plt.title('SHAP Feature Importance (Logistic Regression)')
                plt.tight_layout()
                plt.savefig(self.shap_dir / "shap_feature_importance_lr.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"逻辑回归SHAP特征重要性图生成失败: {e}")
        
        # 4. 瀑布图 (展示单个样本的预测解释)
        if len(data['X_test']) > 0 and shap_results.get('random_forest') is not None:
            try:
                plt.figure(figsize=(10, 6))
                rf_shap_values = shap_results['random_forest']['shap_values']
                rf_explainer = shap_results['random_forest']['explainer']
                
                # 获取期望值
                expected_value = rf_explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                
                # 创建SHAP Explanation对象
                explanation = shap.Explanation(
                    values=rf_shap_values[0],
                    base_values=expected_value,
                    data=data['X_test'][0],
                    feature_names=data['feature_names']
                )
                shap.waterfall_plot(explanation, show=False)
                plt.title('SHAP Waterfall Plot (Sample Prediction Explanation)')
                plt.tight_layout()
                plt.savefig(self.shap_dir / "shap_waterfall_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("SHAP瀑布图生成成功")
            except Exception as waterfall_error:
                logger.warning(f"瀑布图生成失败: {waterfall_error}")
                # 生成替代的条形图
                try:
                    plt.figure(figsize=(10, 6))
                    feature_importance = shap_results['random_forest']['shap_values'][0]
                    sorted_idx = np.argsort(np.abs(feature_importance))[-10:]  # Top 10
                    
                    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                    plt.yticks(range(len(sorted_idx)), [data['feature_names'][i] for i in sorted_idx])
                    plt.xlabel('SHAP Value')
                    plt.title('SHAP Feature Importance (Single Sample)')
                    plt.tight_layout()
                    plt.savefig(self.shap_dir / "shap_single_sample_importance.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("SHAP单样本重要性图生成成功")
                except Exception as bar_error:
                    logger.warning(f"替代条形图生成也失败: {bar_error}")
    
    def generate_traditional_plots(self, data, models, importance_df):
        """生成传统的特征重要性图表"""
        logger.info("生成传统特征重要性图表...")
        
        # 1. 特征重要性对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 15 特征
        top_features = importance_df.head(15)
        
        # 随机森林重要性
        axes[0, 0].barh(range(len(top_features)), top_features['RandomForest'], color='forestgreen')
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['Feature'])
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_title('Random Forest Feature Importance')
        axes[0, 0].invert_yaxis()
        
        # 逻辑回归系数
        axes[0, 1].barh(range(len(top_features)), top_features['LogisticRegression'], color='steelblue')
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['Feature'])
        axes[0, 1].set_xlabel('Coefficient Magnitude')
        axes[0, 1].set_title('Logistic Regression Feature Importance')
        axes[0, 1].invert_yaxis()
        
        # 排列重要性
        axes[1, 0].barh(range(len(top_features)), top_features['Permutation'], color='darkorange')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['Feature'])
        axes[1, 0].set_xlabel('Permutation Importance')
        axes[1, 0].set_title('Permutation Feature Importance')
        axes[1, 0].invert_yaxis()
        
        # 综合排名
        axes[1, 1].barh(range(len(top_features)), 1/top_features['Average_Rank'], color='purple')
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features['Feature'])
        axes[1, 1].set_xlabel('Inverse Average Rank')
        axes[1, 1].set_title('Combined Feature Ranking')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.shap_dir / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特征重要性相关性分析
        fig, ax = plt.subplots(figsize=(8, 6))
        
        importance_corr = importance_df[['RandomForest', 'LogisticRegression', 'Permutation']].corr()
        sns.heatmap(importance_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.3f')
        ax.set_title('Feature Importance Method Correlation')
        
        plt.tight_layout()
        plt.savefig(self.shap_dir / "importance_method_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_biological_significance(self, importance_df):
        """分析特征的生物学意义"""
        logger.info("分析特征的生物学意义...")
        
        # 特征分类
        feature_categories = {
            'Physical Properties': ['Length', 'Charge', 'Isoelectric_Point'],
            'Hydrophobicity': ['Hydrophobicity', 'Hydrophobic_Moment', 'Hydrophilicity'],
            'Structural Properties': ['Instability_Index', 'Aliphatic_Index'],
            'Amino Acid Composition': [col for col in importance_df['Feature'] if col.startswith('AA_')]
        }
        
        # 计算每个类别的平均重要性
        category_importance = {}
        for category, features in feature_categories.items():
            category_features = importance_df[importance_df['Feature'].isin(features)]
            if not category_features.empty:
                avg_importance = category_features['RandomForest'].mean()
                category_importance[category] = avg_importance
        
        # 生成类别重要性图
        if category_importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            categories = list(category_importance.keys())
            importances = list(category_importance.values())
            
            bars = ax.bar(categories, importances, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax.set_xlabel('Feature Category')
            ax.set_ylabel('Average Feature Importance')
            ax.set_title('Feature Importance by Biological Category')
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{imp:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.shap_dir / "biological_category_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 保存生物学意义分析
        bio_analysis = {
            'top_10_features': importance_df.head(10)['Feature'].tolist(),
            'category_importance': category_importance,
            'biological_insights': {
                'most_important_physical': importance_df[importance_df['Feature'].isin(feature_categories['Physical Properties'])].head(1)['Feature'].tolist(),
                'most_important_hydrophobic': importance_df[importance_df['Feature'].isin(feature_categories['Hydrophobicity'])].head(1)['Feature'].tolist(),
                'most_important_aa': importance_df[importance_df['Feature'].str.startswith('AA_')].head(3)['Feature'].tolist()
            }
        }
        
        # 保存为JSON
        import json
        with open(self.shap_dir / "biological_significance_analysis.json", 'w') as f:
            json.dump(bio_analysis, f, indent=2, default=str)
        
        return bio_analysis
    
    def run_complete_analysis(self):
        """运行完整的SHAP分析"""
        logger.info("开始执行第3部分：特征分析与模型可解释性")
        
        try:
            # 1. 加载数据
            data = self.load_and_prepare_data()
            
            # 2. 训练模型
            models = self.train_interpretable_models(data)
            
            # 3. 特征重要性分析
            importance_df = self.analyze_feature_importance(data, models)
            
            # 4. SHAP分析
            shap_results = self.perform_shap_analysis(data, models)
            
            # 5. 生成SHAP图表
            self.generate_shap_plots(data, shap_results)
            
            # 6. 生成传统图表
            self.generate_traditional_plots(data, models, importance_df)
            
            # 7. 生物学意义分析
            bio_analysis = self.analyze_biological_significance(importance_df)
            
            logger.info("第3部分完成：特征分析与模型可解释性")
            logger.info(f"结果保存在: {self.shap_dir}")
            
            return {
                'importance_df': importance_df,
                'bio_analysis': bio_analysis,
                'models_performance': {
                    name: model.score(data['X_test'], data['y_test']) 
                    for name, model in models.items()
                }
            }
            
        except Exception as e:
            logger.error(f"SHAP分析执行出错: {e}")
            raise

def main():
    """主函数"""
    try:
        analyzer = SHAPAnalyzer()
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*60)
        print("第3部分：特征分析与模型可解释性 - 执行成功！")
        print(f"Top 5 重要特征:")
        for i, feature in enumerate(results['importance_df'].head(5)['Feature'], 1):
            print(f"  {i}. {feature}")
        print(f"结果保存在: {analyzer.shap_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"分析执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())