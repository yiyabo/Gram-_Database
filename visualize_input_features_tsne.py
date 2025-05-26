import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_tsne_on_input_features(X_path='./data/X_train.npy', 
                                     y_path='./data/y_train.npy', 
                                     output_dir='./visualization',
                                     output_filename='input_features_tsne.png',
                                     perplexity=30,
                                     n_iter=1000,
                                     random_state=42,
                                     max_samples=None): # 新增max_samples参数
    """
    对输入特征进行t-SNE降维并可视化。

    Args:
        X_path (str): 特征数据 .npy 文件路径。
        y_path (str): 标签数据 .npy 文件路径。
        output_dir (str): 可视化结果输出目录。
        output_filename (str): 输出图像文件名。
        perplexity (float): t-SNE的perplexity参数。
        n_iter (int): t-SNE的迭代次数。
        random_state (int): 随机种子。
        max_samples (int, optional): 用于t-SNE的最大样本数，以防数据过大。默认为None（使用所有样本）。
    """
    logger.info(f"开始对输入特征进行t-SNE可视化...")
    logger.info(f"从 {X_path} 和 {y_path} 加载数据。")

    try:
        X = np.load(X_path)
        y = np.load(y_path)
    except FileNotFoundError:
        logger.error(f"错误：数据文件未找到。请确保 {X_path} 和 {y_path} 存在。")
        logger.error("您可能需要先运行 gram_classification.py 来生成这些数据文件。")
        return

    if X.ndim == 1: # 处理一维X的情况 (例如只有一个样本)
        logger.warning(f"特征数据X只有一维 (shape: {X.shape})，无法进行t-SNE。至少需要两个样本。")
        return
    if len(X) != len(y):
        logger.error(f"特征和标签数量不匹配: X有{len(X)}个样本, y有{len(y)}个样本。")
        return
    if len(X) == 0:
        logger.error("特征数据为空，无法进行t-SNE。")
        return

    # 数据标准化 (与训练时保持一致)
    logger.info("对特征数据进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 如果指定了max_samples，则进行子采样
    if max_samples is not None and len(X_scaled) > max_samples:
        logger.info(f"数据量较大 ({len(X_scaled)}个样本)，将随机采样 {max_samples} 个样本进行t-SNE。")
        indices = np.random.choice(len(X_scaled), max_samples, replace=False)
        X_subset = X_scaled[indices]
        y_subset = y[indices]
    else:
        X_subset = X_scaled
        y_subset = y

    if len(X_subset) < perplexity + 2 : # t-SNE对样本量有要求
        logger.warning(f"用于t-SNE的样本量 ({len(X_subset)}) 过小，无法满足perplexity ({perplexity}) 的要求。跳过t-SNE。")
        return

    logger.info(f"对 {X_subset.shape[0]} 个样本进行t-SNE降维 (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(n_components=2, 
                random_state=random_state, 
                perplexity=float(perplexity), # 确保是float
                n_iter=int(n_iter),
                init='pca', # 使用PCA初始化更稳定
                learning_rate='auto') # 自动学习率
    
    try:
        X_tsne = tsne.fit_transform(X_subset)
    except Exception as e:
        logger.error(f"t-SNE计算过程中出错: {e}")
        logger.error("这可能是由于perplexity相对于样本数量过大，或者数据本身存在问题。")
        logger.error("您可以尝试调整perplexity值或检查数据。")
        return

    logger.info("t-SNE降维完成。开始绘图...")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(y_subset)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, label_val in enumerate(unique_labels):
        indices = (y_subset == label_val)
        class_name = f'Class {int(label_val)}' # 假设标签是0和1
        if int(label_val) == 0: class_name = 'Non-AMP (Class 0)'
        if int(label_val) == 1: class_name = 'AMP (Class 1)'
            
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                    label=class_name, 
                    color=colors[i % len(colors)], 
                    alpha=0.7, s=50)

    plt.title(f't-SNE Visualization of Input Features (Perplexity: {perplexity})', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300)
        logger.info(f"t-SNE可视化图像已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存图像时出错: {e}")
    plt.close()

if __name__ == '__main__':
    # 可以通过命令行参数调整这些值，或者直接修改
    # 为了简单起见，这里直接调用
    # 注意：如果您的 X_train.npy 非常大，t-SNE可能会很慢或消耗大量内存
    # 可以考虑使用 max_samples 参数进行子采样，例如 max_samples=5000
    visualize_tsne_on_input_features(perplexity=30, n_iter=1000, max_samples=5000) 
    # 对于非常大的数据集，perplexity可能需要调整，例如 5 到 50 之间
    # n_iter 至少为 250，通常 1000 或更高效果更好但更慢