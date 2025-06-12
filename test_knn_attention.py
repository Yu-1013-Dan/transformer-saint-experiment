#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-NN跨样本注意力模块测试脚本

演示增强版SAINT模型的k-NN跨样本注意力功能：
1. 创建增强版SAINT模型
2. 测试k-NN注意力机制
3. 分析注意力模式
4. 与原始模型进行比较
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.append('.')

from saint.models.enhanced_saint import EnhancedTabAttention, create_enhanced_saint_model
from saint.models.model import TabAttention
from saint.models.knn_attention import visualize_knn_graph


def create_test_data(batch_size=16, num_categorical=5, num_continuous=10):
    """
    创建测试数据
    
    Args:
        batch_size: 批次大小
        num_categorical: 分类特征数量
        num_continuous: 连续特征数量
    
    Returns:
        测试数据字典
    """
    # 分类特征的cardinality
    categories = [10, 5, 8, 15, 6][:num_categorical]
    
    # 生成分类特征数据
    x_categ = torch.randint(0, max(categories), (batch_size, num_categorical))
    
    # 生成连续特征数据
    x_cont = torch.randn(batch_size, num_continuous)
    
    # 生成编码后的特征（模拟嵌入后的特征）
    embed_dim = 32
    x_categ_enc = torch.randn(batch_size, num_categorical, embed_dim)
    x_cont_enc = torch.randn(batch_size, num_continuous, embed_dim)
    
    # 生成标签（分类任务）
    labels = torch.randint(0, 8, (batch_size,))  # 8个设备类别
    
    return {
        'categories': categories,
        'x_categ': x_categ,
        'x_cont': x_cont,
        'x_categ_enc': x_categ_enc,
        'x_cont_enc': x_cont_enc,
        'labels': labels,
        'batch_size': batch_size,
        'num_categorical': num_categorical,
        'num_continuous': num_continuous
    }


def test_knn_attention_module():
    """
    测试k-NN跨样本注意力模块
    """
    print("="*60)
    print("测试 k-NN跨样本注意力模块")
    print("="*60)
    
    # 创建测试数据
    test_data = create_test_data(batch_size=8, num_categorical=3, num_continuous=5)
    
    print(f"批次大小: {test_data['batch_size']}")
    print(f"分类特征数量: {test_data['num_categorical']}")
    print(f"连续特征数量: {test_data['num_continuous']}")
    print(f"分类特征cardinality: {test_data['categories']}")
    
    # 创建增强版SAINT模型
    model = create_enhanced_saint_model(
        categories=test_data['categories'],
        num_continuous=test_data['num_continuous'],
        dim=32,
        depth=4,
        heads=4,
        k_neighbors=3,
        use_knn_attention=True,
        knn_start_layer=1,
        dim_out=8  # 8个设备类别
    )
    
    print(f"\n模型信息:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 前向传播测试
    print(f"\n前向传播测试:")
    with torch.no_grad():
        output = model(
            test_data['x_categ'], 
            test_data['x_cont'], 
            test_data['x_categ_enc'], 
            test_data['x_cont_enc']
        )
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 测试返回注意力信息
    print(f"\n注意力信息测试:")
    with torch.no_grad():
        output, attention_info = model(
            test_data['x_categ'], 
            test_data['x_cont'], 
            test_data['x_categ_enc'], 
            test_data['x_cont_enc'],
            return_attention_info=True
        )
        
        if attention_info:
            print(f"注意力层数: {len(attention_info)}")
            for i, info in enumerate(attention_info):
                print(f"  层 {i}: {info['type']}, 输入形状: {info['input_shape']}, 输出形状: {info['output_shape']}")
        else:
            print("  无注意力信息返回")
    
    return model, test_data


def test_knn_analysis():
    """
    测试k-NN注意力分析功能
    """
    print("\n" + "="*60)
    print("测试 k-NN注意力分析功能")
    print("="*60)
    
    # 创建测试数据（更大的批次以便观察k-NN效果）
    test_data = create_test_data(batch_size=12, num_categorical=3, num_continuous=5)
    
    # 创建增强版SAINT模型
    model = create_enhanced_saint_model(
        categories=test_data['categories'],
        num_continuous=test_data['num_continuous'],
        dim=32,
        depth=3,
        heads=4,
        k_neighbors=4,
        use_knn_attention=True,
        knn_start_layer=1,
        dim_out=8
    )
    
    # 分析k-NN注意力
    print(f"分析k-NN注意力模式:")
    with torch.no_grad():
        analysis_results = model.analyze_knn_attention(
            test_data['x_categ'], 
            test_data['x_cont'], 
            test_data['x_categ_enc'], 
            test_data['x_cont_enc']
        )
        
        if analysis_results:
            for result in analysis_results:
                print(f"\n层 {result['layer_idx']}:")
                print(f"  样本表示形状: {result['sample_representations_shape']}")
                print(f"  平均相似度: {result['avg_similarity']:.4f}")
                print(f"  相似度标准差: {result['similarity_std']:.4f}")
                
                # 显示k-NN图结构
                knn_indices = result['knn_indices']
                knn_similarities = result['knn_similarities']
                
                print(f"  k-NN图结构（前5个样本）:")
                for i in range(min(5, knn_indices.shape[0])):
                    neighbors = knn_indices[i].numpy()
                    similarities = knn_similarities[i].numpy()
                    neighbor_str = ", ".join([f"{n}({s:.3f})" for n, s in zip(neighbors, similarities)])
                    print(f"    样本 {i}: {neighbor_str}")
        else:
            print("  k-NN注意力未启用或分析失败")
    
    return model, test_data, analysis_results


def compare_with_original_model():
    """
    与原始SAINT模型进行比较
    """
    print("\n" + "="*60)
    print("与原始SAINT模型进行比较")
    print("="*60)
    
    # 创建测试数据
    test_data = create_test_data(batch_size=8, num_categorical=3, num_continuous=5)
    
    # 创建原始SAINT模型
    original_model = TabAttention(
        categories=test_data['categories'],
        num_continuous=test_data['num_continuous'],
        dim=32,
        depth=3,
        heads=4,
        dim_out=8
    )
    
    # 创建增强版SAINT模型
    enhanced_model = create_enhanced_saint_model(
        categories=test_data['categories'],
        num_continuous=test_data['num_continuous'],
        dim=32,
        depth=3,
        heads=4,
        k_neighbors=3,
        use_knn_attention=True,
        knn_start_layer=1,
        dim_out=8
    )
    
    print(f"原始模型参数数量: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"增强模型参数数量: {sum(p.numel() for p in enhanced_model.parameters()):,}")
    
    # 比较输出
    with torch.no_grad():
        original_output = original_model(
            test_data['x_categ'], 
            test_data['x_cont'], 
            test_data['x_categ_enc'], 
            test_data['x_cont_enc']
        )
        
        enhanced_output = enhanced_model(
            test_data['x_categ'], 
            test_data['x_cont'], 
            test_data['x_categ_enc'], 
            test_data['x_cont_enc']
        )
        
        # 计算差异
        output_diff = torch.abs(original_output - enhanced_output).mean().item()
        
        print(f"\n输出比较:")
        print(f"  原始模型输出形状: {original_output.shape}")
        print(f"  增强模型输出形状: {enhanced_output.shape}")
        print(f"  平均绝对差异: {output_diff:.6f}")
        print(f"  原始模型输出范围: [{original_output.min().item():.4f}, {original_output.max().item():.4f}]")
        print(f"  增强模型输出范围: [{enhanced_output.min().item():.4f}, {enhanced_output.max().item():.4f}]")
    
    return original_model, enhanced_model, test_data


def visualize_attention_patterns(model, test_data, analysis_results):
    """
    可视化注意力模式
    """
    print("\n" + "="*60)
    print("可视化注意力模式")
    print("="*60)
    
    if not analysis_results:
        print("没有可视化的注意力数据")
        return
    
    # 选择第一层的k-NN结果进行可视化
    first_layer_result = analysis_results[0]
    knn_indices = first_layer_result['knn_indices']
    knn_similarities = first_layer_result['knn_similarities']
    
    print(f"可视化第 {first_layer_result['layer_idx']} 层的k-NN注意力模式:")
    
    # 创建相似度矩阵用于可视化
    batch_size = knn_indices.shape[0]
    similarity_matrix = torch.zeros(batch_size, batch_size)
    
    for i in range(batch_size):
        for j, neighbor_idx in enumerate(knn_indices[i]):
            similarity_matrix[i, neighbor_idx] = knn_similarities[i, j]
    
    # 使用文本方式显示相似度矩阵
    print("\n相似度矩阵 (只显示k-NN连接):")
    print("行=查询样本, 列=邻居样本")
    print("  ", end="")
    for j in range(batch_size):
        print(f"{j:6}", end="")
    print()
    
    for i in range(batch_size):
        print(f"{i:2}", end="")
        for j in range(batch_size):
            val = similarity_matrix[i, j].item()
            if val > 0:
                print(f"{val:6.3f}", end="")
            else:
                print("  .   ", end="")
        print()
    
    # 显示每个样本的邻居信息
    print(f"\n详细邻居信息:")
    for i in range(min(5, batch_size)):  # 只显示前5个样本
        neighbors = knn_indices[i].numpy()
        similarities = knn_similarities[i].numpy()
        print(f"样本 {i} 的邻居:")
        for j, (neighbor, sim) in enumerate(zip(neighbors, similarities)):
            print(f"  第{j+1}邻居: 样本{neighbor} (相似度: {sim:.4f})")


def main():
    """
    主测试函数
    """
    print("k-NN跨样本注意力模块测试")
    print("="*60)
    
    try:
        # 1. 测试基本功能
        model, test_data = test_knn_attention_module()
        
        # 2. 测试k-NN分析功能
        analysis_model, analysis_data, analysis_results = test_knn_analysis()
        
        # 3. 与原始模型比较
        original_model, enhanced_model, compare_data = compare_with_original_model()
        
        # 4. 可视化注意力模式
        visualize_attention_patterns(analysis_model, analysis_data, analysis_results)
        
        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)
        
        print("\nk-NN跨样本注意力模块主要功能:")
        print("1. ✅ 样本表示提取 - 使用平均池化获得样本级别表示")
        print("2. ✅ k-NN图计算 - 支持余弦相似度、欧几里得距离等")
        print("3. ✅ 注意力掩码生成 - 限制注意力到k个最相似邻居")
        print("4. ✅ 带掩码的跨样本注意力 - 实现精准的局部注意力")
        print("5. ✅ 注意力分析功能 - 可视化和分析k-NN图结构")
        print("6. ✅ 模型比较功能 - 与原始SAINT模型对比")
        
        print("\n使用建议:")
        print("- k_neighbors=3-7: 适中的邻居数量，平衡效果和计算复杂度")
        print("- knn_start_layer=1: 从第二层开始使用k-NN注意力，保留第一层的全局注意力")
        print("- similarity_metric='cosine': 推荐使用余弦相似度进行样本比较")
        print("- temperature=1.0: 注意力温度参数，可根据数据调整")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 