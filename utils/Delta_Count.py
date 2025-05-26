#!/usr/bin/env python3
"""
$ python simple_delta_cli.py natureAMP.fasta \
    --k 10 --kmer 5 --sample 32 --out stats.json
"""

import argparse
import json
import sys
import os
import random
from pathlib import Path
from itertools import combinations

import networkx as nx
from Bio import SeqIO
from networkx.algorithms import approximation as apx
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 序列相似度计算函数
# ---------------------------------------------------------------------------

def calculate_kmer_similarity(seq1, seq2, k=5):
    """计算两个序列的k-mer相似度"""
    # 获取序列的字符串表示
    seq1_str = str(seq1.seq) if hasattr(seq1, 'seq') else str(seq1)
    seq2_str = str(seq2.seq) if hasattr(seq2, 'seq') else str(seq2)
    
    # 如果序列太短，返回最小相似度
    if len(seq1_str) < k or len(seq2_str) < k:
        return 0
    
    # 生成k-mers集合
    kmers1 = set(seq1_str[i:i+k] for i in range(len(seq1_str)-k+1))
    kmers2 = set(seq2_str[i:i+k] for i in range(len(seq2_str)-k+1))
    
    # 计算Jaccard相似度
    intersection = len(kmers1.intersection(kmers2))
    union = len(kmers1.union(kmers2))
    
    if union == 0:
        return 0
    return intersection / union

def build_knn_graph(sequences, k=10, kmer_size=5):
    """构建k-近邻图"""
    n = len(sequences)
    print(f"[INFO] 开始为{n}个序列构建{k}-NN图")
    
    # 存储所有序列对的相似度
    similarities = {}
    
    # 计算序列对相似度
    for i in tqdm(range(n), desc="计算序列相似度"):
        for j in range(i+1, n):
            sim = calculate_kmer_similarity(sequences[i], sequences[j], k=kmer_size)
            similarities[(i, j)] = sim
            similarities[(j, i)] = sim  # 对称性
    
    # 构建图
    G = nx.Graph()
    
    # 添加节点 (以序列ID为名称)
    for i, seq in enumerate(sequences):
        G.add_node(i, id=seq.id if hasattr(seq, 'id') else f"seq_{i}")
    
    # 对每个序列，添加到k个最相似序列的边
    for i in range(n):
        # 获取与序列i的所有相似度
        edges = [(j, similarities.get((i, j), 0)) for j in range(n) if j != i]
        # 按相似度排序，选择前k个
        top_k = sorted(edges, key=lambda x: x[1], reverse=True)[:k]
        
        # 添加边 (转换相似度为距离: 距离 = 1 - 相似度)
        for j, sim in top_k:
            if sim <= 0:
                continue                    # 跳过无相似度的"假邻居"
            dist = max(1 - sim, 1e-6)      # 保证权重 > 0，防止零环
            G.add_edge(i, j, weight=dist)
    
    # 确保图是连通的（取最大连通分量）
    if not nx.is_connected(G):
        print("[WARN] 图不是连通的，提取最大连通分量")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G

def simple_delta_estimate(G, sample_size=32):
    """
    最简化版本的delta-hyperbolicity估计
    仅对有限数量的四元组进行采样
    """
    print(f"[INFO] 使用简化算法估计δ-hyperbolicity (采样大小={sample_size})")
    
    # 获取图的所有节点
    nodes = list(G.nodes())
    if len(nodes) < 4:
        print("[WARN] 图中节点少于4个，无法计算δ-hyperbolicity")
        return 0.0
    
    # 计算最短路径
    print("[INFO] 计算最短路径...")
    
    # 随机选择sample_size个节点进行采样
    sample_nodes = random.sample(nodes, min(sample_size, len(nodes)))
    
    # 计算这些节点之间的所有最短路径长度
    distances = {}
    for u in tqdm(sample_nodes, desc="计算节点间距离"):
        distances[u] = {}
        for v in sample_nodes:
            if u != v:
                try:
                    # 使用nx.shortest_path_length可能更快
                    distances[u][v] = nx.shortest_path_length(G, u, v, weight='weight')
                except nx.NetworkXNoPath:
                    # 如果没有路径，设置为无穷大
                    distances[u][v] = float('inf')
    
    # 计算有限数量的四元组的delta值
    print("[INFO] 计算四元组delta值...")
    delta_values = []
    
    # 从采样节点中选择四元组
    quad_total = len(list(combinations(sample_nodes, 4)))
    num_samples = min(1000, quad_total)
    
    if num_samples == quad_total:
        four_tuples = combinations(sample_nodes, 4)  # 全部遍历无需random.sample
    else:
        four_tuples = random.sample(list(combinations(sample_nodes, 4)), num_samples)
    
    for a, b, c, d in tqdm(four_tuples, desc="计算四元组delta"):
        try:
            # 计算三种可能的距离和
            sum1 = distances[a][b] + distances[c][d]
            sum2 = distances[a][c] + distances[b][d]
            sum3 = distances[a][d] + distances[b][c]
            
            # 跳过包含无穷距离的四元组
            if any(s == float('inf') for s in (sum1, sum2, sum3)):
                continue
                
            # 排序并取最大差值的一半
            sums = sorted([sum1, sum2, sum3], reverse=True)
            delta = (sums[0] - sums[1]) / 2
            delta_values.append(delta)
        except KeyError:
            # 跳过出现错误的四元组
            continue
    
    # 如果没有有效的delta值，返回0
    if not delta_values:
        print("[WARN] 未能计算出任何有效的delta值")
        return 0.0
    
    # 返回最大delta值（这是Gromov delta的定义）
    return max(delta_values)

def compute_delta(G, sample_size=32):
    """计算delta-hyperbolicity、直径及比率"""
    print("[INFO] 估算图的δ-hyperbolicity")
    
    # 使用我们的简化算法
    delta = simple_delta_estimate(G, sample_size)
    
    # 图直径（近似）
    print("[INFO] 计算图直径...")
    try:
        diameter = apx.diameter(G)
    except Exception as e:
        print(f"[WARN] 计算直径时出错: {str(e)}")
        print("[INFO] 使用采样法估计直径...")
        # 采样法估计直径
        sample_nodes = random.sample(list(G.nodes()), min(sample_size, len(G.nodes())))
        max_distance = 0
        for u in tqdm(sample_nodes, desc="估算直径"):
            lengths = nx.single_source_dijkstra_path_length(G, u, weight='weight')
            for v, dist in lengths.items():
                max_distance = max(max_distance, dist)
        diameter = max_distance if max_distance > 0 else float('inf')
    
    # delta/diameter比率
    ratio = delta / diameter if diameter > 0 else float('inf')
    
    return delta, diameter, ratio

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    # 固定随机种子，确保结果可重现
    random.seed(42)
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="估计FASTA蛋白质/肽集合的δ-hyperbolicity"
    )
    parser.add_argument("fasta", help="输入FASTA文件")
    parser.add_argument("--k", type=int, default=10, help="k-NN图中的k (默认: 10)")
    parser.add_argument("--kmer", type=int, default=5, help="k-mer大小 (默认: 5)")
    parser.add_argument("--sample", type=int, default=32, 
                        help="用于δ近似的样本数量 (默认: 32)")
    parser.add_argument("--out", type=str, default=None,
                        help="可选的JSON文件以保存结果")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机数种子 (默认: 42)")
    
    args = parser.parse_args()
    
    # 设置用户指定的随机种子
    random.seed(args.seed)
    
    # 检查输入文件
    fasta_path = Path(args.fasta).resolve()
    if not fasta_path.exists():
        sys.stderr.write(f"[ERROR] FASTA文件不存在: {fasta_path}\n")
        sys.exit(1)
    
    # 读取序列
    print(f"[INFO] 从{fasta_path}读取序列")
    try:
        sequences = list(SeqIO.parse(fasta_path, "fasta"))
        if not sequences:
            sys.stderr.write(f"[ERROR] 文件中未找到序列 {fasta_path}\n")
            sys.exit(1)
        print(f"[INFO] 读取了{len(sequences)}个序列")
    except Exception as e:
        sys.stderr.write(f"[ERROR] 读取FASTA文件时出错: {str(e)}\n")
        sys.exit(1)
    
    # 构建图
    G = build_knn_graph(sequences, k=args.k, kmer_size=args.kmer)
    print(f"[INFO] 构建的图: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    
    # 计算hyperbolicity
    delta, diameter, ratio = compute_delta(G, sample_size=args.sample)
    
    # 输出结果
    print("\n======= 结果 =======")
    print(f"δ-hyperbolicity (近似): {delta:.4f}")
    print(f"图直径 (近似):          {diameter}")
    print(f"δ/diameter 比率:        {ratio:.4f}")
    
    # 保存结果
    if args.out:
        results = {
            "delta": float(delta),
            "diameter": int(diameter),
            "delta_over_D": float(ratio)
        }
        
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] 结果已保存至 {args.out}")

if __name__ == "__main__":
    main() 