#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC IoT 2024 特征分类和索引表生成模块
根据用户具体要求进行精确特征分类
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CICIoTFeatureClassifier:
    """CIC IoT数据集特征分类器 - 用户定制版本"""
    
    def __init__(self):
        """初始化特征分类器 - 根据用户具体要求"""
        
        # 标签列
        self.label_columns = ['device_mac']
        
        # 需要排除的列（用户明确指定）
        self.excluded_columns = [
            'stream', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 
            'Label 1 for DI', 'Label 2 for AD'
        ]
        
        # 用户精确指定的类别特征列表
        self.user_defined_categorical_list = [
            'src_port', 'dst_port', 'port_class_dst', 'l4_tcp', 'l4_udp',
            'handshake_version', 'handshake_ciphersuites', 'tls_server',
            'http_request_method', 'http_host', 'http_response_code', 'user_agent',
            'dns_server', 'dns_query_type', 'eth_src_oui', 'eth_dst_oui',
            'highest_layer', 'http_uri', 'http_content_type', 'icmp_type', 
            'icmp_checksum_status'
        ]
        
        # 转换为字典格式以保持兼容性
        self.user_defined_categorical = {feature: '类别型' for feature in self.user_defined_categorical_list}
        
        # 用户精确指定的数值特征列表
        self.user_defined_numerical_list = [
            'inter_arrival_time', 'time_since_previously_displayed_frame', 'ttl', 'eth_size',
            'tcp_window_size', 'payload_entropy', 'handshake_cipher_suites_length',
            'handshake_extensions_length', 'handshake_sig_hash_alg_len', 'dns_len_qry',
            'dns_interval', 'dns_len_ans', 'payload_length', 'http_content_len',
            'icmp_data_size', 'jitter', 'stream_1_count', 'stream_1_mean', 'stream_1_var',
            'src_ip_1_count', 'src_ip_1_mean', 'src_ip_1_var', 'src_ip_mac_1_count',
            'src_ip_mac_1_mean', 'src_ip_mac_1_var', 'channel_1_count', 'channel_1_mean',
            'channel_1_var', 'stream_jitter_1_sum', 'stream_jitter_1_mean',
            'stream_jitter_1_var', 'stream_5_count', 'stream_5_mean', 'stream_5_var',
            'src_ip_5_count', 'src_ip_5_mean', 'src_ip_5_var', 'src_ip_mac_5_count',
            'src_ip_mac_5_mean', 'src_ip_mac_5_var', 'channel_5_count', 'channel_5_mean',
            'channel_5_var', 'stream_jitter_5_sum', 'stream_jitter_5_mean',
            'stream_jitter_5_var', 'stream_10_count', 'stream_10_mean', 'stream_10_var',
            'src_ip_10_count', 'src_ip_10_mean', 'src_ip_10_var', 'src_ip_mac_10_count',
            'src_ip_mac_10_mean', 'src_ip_mac_10_var', 'channel_10_count',
            'channel_10_mean', 'channel_10_var', 'stream_jitter_10_sum',
            'stream_jitter_10_mean', 'stream_jitter_10_var', 'stream_30_count',
            'stream_30_mean', 'stream_30_var', 'src_ip_30_count', 'src_ip_30_mean',
            'src_ip_30_var', 'src_ip_mac_30_count', 'src_ip_mac_30_mean',
            'src_ip_mac_30_var', 'channel_30_count', 'channel_30_mean',
            'channel_30_var', 'stream_jitter_30_sum', 'stream_jitter_30_mean',
            'stream_jitter_30_var', 'stream_60_count', 'stream_60_mean', 'stream_60_var',
            'src_ip_60_count', 'src_ip_60_mean', 'src_ip_60_var', 'src_ip_mac_60_count',
            'src_ip_mac_60_mean', 'src_ip_mac_60_var', 'channel_60_count',
            'channel_60_mean', 'channel_60_var', 'stream_jitter_60_sum',
            'stream_jitter_60_mean', 'stream_jitter_60_var', 'ntp_interval',
            'most_freq_spot', 'min_et', 'q1', 'min_e', 'var_e', 'q1_e', 'sum_p',
            'min_p', 'max_p', 'med_p', 'average_p', 'var_p', 'q3_p', 'q1_p', 'iqr_p',
            'l3_ip_dst_count'
        ]
        
        # 转换为字典格式以保持兼容性
        self.user_defined_numerical = {feature: '数值型' for feature in self.user_defined_numerical_list}
        

        
        # 高基数特征（需要特殊处理）
        self.high_cardinality_features = [
            'handshake_ciphersuites', 'tls_server', 'http_host', 
            'user_agent', 'dns_server', 'http_uri'
        ]
    
    def classify_features_automatically(self, df: pd.DataFrame) -> Dict[str, List]:
        """根据用户具体要求进行自动特征分类"""
        print("🏷️  根据用户要求进行精确特征分类...")
        
        all_columns = list(df.columns)
        classification = {
            'label_columns': [],
            'identifier_columns': [],
            'categorical_features': [],
            'numerical_features': [],
            'high_cardinality_features': [],
            'text_features': [],
            'excluded_features': []
        }
        
        for col in all_columns:
            col_lower = col.lower()
            
            # 1. 检查是否为标签列
            if col in self.label_columns:
                classification['label_columns'].append(col)
                continue
            
            # 2. 检查是否为排除列
            if col in self.excluded_columns:
                classification['excluded_features'].append(col)
                continue
            
            # 3. 检查用户精确定义的类别特征
            if col in self.user_defined_categorical_list:
                classification['categorical_features'].append(col)
                # 检查是否为高基数特征
                if col in self.high_cardinality_features:
                    classification['high_cardinality_features'].append(col)
                continue
            
            # 4. 检查用户精确定义的数值特征
            if col in self.user_defined_numerical_list:
                classification['numerical_features'].append(col)
                continue
            
            # 5. 对于用户未明确指定的特征，排除它们
            classification['excluded_features'].append(col)
        
        # 打印详细分类结果
        print(f"\n📊 用户要求的特征分类结果:")
        print(f"   标签列 ({len(classification['label_columns'])}): {classification['label_columns']}")
        print(f"   排除列 ({len(classification['excluded_features'])}): {classification['excluded_features'][:10]}{'...' if len(classification['excluded_features']) > 10 else ''}")
        print(f"   类别特征 ({len(classification['categorical_features'])}): {classification['categorical_features'][:10]}{'...' if len(classification['categorical_features']) > 10 else ''}")
        print(f"   数值特征 ({len(classification['numerical_features'])}): {classification['numerical_features'][:10]}{'...' if len(classification['numerical_features']) > 10 else ''}")
        print(f"   高基数特征 ({len(classification['high_cardinality_features'])}): {classification['high_cardinality_features']}")
        
        total_features = len(classification['categorical_features']) + len(classification['numerical_features'])
        print(f"   总有效特征: {total_features} 个")
        print(f"   期望特征总数: {len(self.user_defined_categorical_list) + len(self.user_defined_numerical_list)} 个")
        
        return classification
    
    def analyze_dataframe_structure(self, df: pd.DataFrame) -> Dict:
        """分析DataFrame的结构和特征类型"""
        print("🔍 开始分析DataFrame结构...")
        
        structure_info = {
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'column_names': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_counts': {},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # 计算唯一值数量（仅对前1000行，避免内存问题）
        sample_df = df.head(1000) if len(df) > 1000 else df
        for col in df.columns:
            try:
                structure_info['unique_counts'][col] = sample_df[col].nunique()
            except:
                structure_info['unique_counts'][col] = -1
        
        print(f"   总列数: {structure_info['total_columns']}")
        print(f"   总行数: {structure_info['total_rows']:,}")
        print(f"   内存使用: {structure_info['memory_usage_mb']:.1f} MB")
        
        return structure_info
    
    def create_feature_indices(self, df: pd.DataFrame, classification: Dict[str, List]) -> Dict:
        """创建特征索引映射"""
        print("📋 创建特征索引映射...")
        
        categorical_features = classification['categorical_features']
        numerical_features = classification['numerical_features']
        
        # 按原始列顺序排序特征
        all_columns = list(df.columns)
        categorical_features = [col for col in all_columns if col in categorical_features]
        numerical_features = [col for col in all_columns if col in numerical_features]
        
        # 创建索引映射
        cat_idxs = list(range(len(categorical_features)))
        con_idxs = list(range(len(categorical_features), len(categorical_features) + len(numerical_features)))
        
        # 计算类别特征的维度
        cat_dims = []
        for col in categorical_features:
            try:
                unique_count = df[col].nunique()
                cat_dims.append(unique_count)
            except:
                cat_dims.append(2)  # 默认二分类
        
        indices = {
            'categorical_indices': cat_idxs,
            'numerical_indices': con_idxs,
            'categorical_dimensions': cat_dims,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'total_features': len(categorical_features) + len(numerical_features)
        }
        
        print(f"   类别特征索引: {len(cat_idxs)} 个")
        print(f"   数值特征索引: {len(con_idxs)} 个")
        print(f"   总特征数: {indices['total_features']}")
        
        return indices
    
    def process_dataframe(self, df: pd.DataFrame, output_dir: str = "feature_analysis") -> Dict:
        """处理DataFrame并生成完整的特征分析报告"""
        print(f"🚀 开始处理DataFrame (输出目录: {output_dir})")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 结构分析
        structure = self.analyze_dataframe_structure(df)
        
        # 2. 特征分类
        classification = self.classify_features_automatically(df)
        
        # 3. 创建索引
        indices = self.create_feature_indices(df, classification)
        
        # 4. 生成报告
        results = {
            'structure': structure,
            'classification': classification,
            'indices': indices,
            'output_directory': output_dir
        }
        
        print(f"\n✅ DataFrame处理完成!")
        return results


def analyze_cic_iot_features(csv_file_path: str, output_dir: str = "feature_analysis") -> Dict:
    """分析CIC IoT数据集特征的主函数"""
    
    print(f"📊 开始分析CIC IoT特征文件: {csv_file_path}")
    
    # 创建分类器
    classifier = CICIoTFeatureClassifier()
    
    # 读取数据样本
    df_sample = pd.read_csv(csv_file_path, nrows=5000)  # 只读取前5000行用于分析
    
    # 处理数据
    results = classifier.process_dataframe(df_sample, output_dir)
    
    return results


if __name__ == "__main__":
    # 示例用法
    csv_path = "/mnt/d/数据集/CIC/BenignTraffic.csv"  # 替换为您的文件路径
    results = analyze_cic_iot_features(csv_path)
    
    print(f"\n🎯 分析完成! 结果保存在: {results['output_directory']}") 