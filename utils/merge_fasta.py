#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并Gram+-.fasta和Gram-.fasta文件到一个新的Gram.fasta文件
"""

from Bio import SeqIO
import os

def merge_fasta_files(input_files, output_file):
    """
    合并多个FASTA文件到一个文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
    """
    records = []
    
    # 读取所有输入文件中的序列记录
    for input_file in input_files:
        print(f"读取文件: {input_file}")
        file_records = list(SeqIO.parse(input_file, "fasta"))
        records.extend(file_records)
        print(f"从 {input_file} 读取了 {len(file_records)} 条序列")
    
    # 写入合并后的记录到输出文件
    SeqIO.write(records, output_file, "fasta")
    print(f"合并完成，总共写入 {len(records)} 条序列到 {output_file}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_files = [
        os.path.join(current_dir, "Gram+-.fasta"),
        os.path.join(current_dir, "Gram-.fasta")
    ]
    output_file = os.path.join(current_dir, "Gram.fasta")
    
    # 合并文件
    merge_fasta_files(input_files, output_file)
