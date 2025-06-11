#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC IoT 2024 特征分类和索引表生成模块
主要功能：
1. 自动分析CSV文件的特征结构
2. 根据特征类型进行分类
3. 生成特征索引表
4. 处理高基数特征
5. 与SAINT模型兼容的特征格式化
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CICIoTFeatureClassifier:
    """CIC IoT数据集特征分类器"""
    
    def __init__(self):
        """初始化特征分类器"""
        
        # 基于CIC IoT数据集的特征分类规则
        self.label_columns = ['device_mac']  # 标签列
        
        # 标识符和待排除的列
        self.identifier_columns = [
            'stream', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip',
            'Label 1 for DI', 'Label 2 for AD'
        ]
        
        # 类别特征规则（基于列名模式）
        self.categorical_patterns = {
            # 端口相关
            'port': ['src_port', 'dst_port', 'port_class_dst'],
            # 协议相关
            'protocol': ['l4_tcp', 'l4_udp', 'protocol', 'highest_layer'],
            # 加密/TLS相关
            'crypto': ['handshake_version', 'handshake_ciphersuites', 'tls_server'],
            # HTTP相关
            'http': ['http_request_method', 'http_host', 'http_response_code', 
                    'user_agent', 'http_content_type'],
            # DNS相关
            'dns': ['dns_server', 'dns_query_type'],
            # 以太网相关
            'ethernet': ['eth_src_oui', 'eth_dst_oui'],
            # ICMP相关
            'icmp': ['icmp_type', 'icmp_checksum_status']
        }
        
        # 数值特征规则（基于列名模式）
        self.numerical_patterns = {
            # 时间相关
            'time': ['inter_arrival_time', 'time_since_previously_displayed_frame', 
                    'dns_interval', 'ntp_interval', 'jitter'],
            # 网络基础
            'network': ['ttl', 'eth_size', 'tcp_window_size', 'payload_length'],
            # 加密长度
            'crypto_len': ['handshake_cipher_suites_length', 'handshake_extensions_length',
                          'handshake_sig_hash_alg_len'],
            # DNS长度
            'dns_len': ['dns_len_qry', 'dns_len_ans'],
            # HTTP长度
            'http_len': ['http_content_len'],
            # ICMP大小
            'icmp_size': ['icmp_data_size'],
            # 熵值
            'entropy': ['payload_entropy'],
            # 统计特征
            'stats': ['stream_', 'sum_', 'min_', 'max_', 'med_', 'average_', 
                     'var_', 'iqr_', 'most_freq_', 'l3_ip_dst_count']
        }
        
        # 高基数特征（需要特殊处理）
        self.high_cardinality_features = [
            'handshake_ciphersuites', 'tls_server', 'http_host', 
            'user_agent', 'dns_server', 'http_uri'
        ]
        
        # 需要特殊处理的文本特征
        self.text_features = ['http_uri', 'user_agent']
    
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
    
    def classify_features_automatically(self, df: pd.DataFrame) -> Dict[str, List]:
        """自动分类特征"""
        print("🏷️  开始自动特征分类...")
        
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
            if col in self.label_columns or 'device_mac' in col_lower:
                classification['label_columns'].append(col)
                continue
            
            # 2. 检查是否为标识符列
            if (col in self.identifier_columns or 
                any(pattern in col_lower for pattern in ['stream', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 'label'])):
                classification['identifier_columns'].append(col)
                continue
            
            # 3. 检查是否为高基数特征
            if col in self.high_cardinality_features:
                classification['high_cardinality_features'].append(col)
                classification['categorical_features'].append(col)
                continue
            
            # 4. 检查是否为文本特征
            if col in self.text_features or 'uri' in col_lower:
                classification['text_features'].append(col)
                classification['excluded_features'].append(col)
                continue
            
            # 5. 基于模式匹配分类
            is_categorical = False
            is_numerical = False
            
            # 检查类别特征模式
            for category, patterns in self.categorical_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    classification['categorical_features'].append(col)
                    is_categorical = True
                    break
            
            if is_categorical:
                continue
            
            # 检查数值特征模式
            for category, patterns in self.numerical_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    classification['numerical_features'].append(col)
                    is_numerical = True
                    break
            
            if is_numerical:
                continue
            
            # 6. 基于数据类型和唯一值数量判断
            try:
                dtype = df[col].dtype
                unique_count = df[col].nunique()
                total_count = len(df[col].dropna())
                
                if total_count == 0:
                    classification['excluded_features'].append(col)
                elif dtype in ['object', 'category'] or unique_count < total_count * 0.05:
                    classification['categorical_features'].append(col)
                elif dtype in ['int64', 'float64'] or unique_count > total_count * 0.05:
                    classification['numerical_features'].append(col)
                else:
                    classification['excluded_features'].append(col)
                    
            except Exception as e:
                print(f"   ⚠️  处理列 '{col}' 时出错: {e}")
                classification['excluded_features'].append(col)
        
        # 打印分类结果
        print(f"\n📊 特征分类结果:")
        for category, features in classification.items():
            if features:
                print(f"   {category}: {len(features)} 个特征")
                if len(features) <= 10:
                    print(f"      {features}")
                else:
                    print(f"      {features[:5]} ... {features[-2:]}")
        
        return classification
    
    def create_feature_indices(self, df: pd.DataFrame, classification: Dict[str, List]) -> Dict:
        """创建特征索引表"""
        print("\n📋 创建特征索引表...")
        
        # 获取所有列名和对应的索引
        all_columns = list(df.columns)
        column_to_index = {col: idx for idx, col in enumerate(all_columns)}
        
        # 创建索引映射
        indices = {
            'label_indices': [column_to_index[col] for col in classification['label_columns'] if col in column_to_index],
            'cat_indices': [column_to_index[col] for col in classification['categorical_features'] if col in column_to_index],
            'num_indices': [column_to_index[col] for col in classification['numerical_features'] if col in column_to_index],
            'excluded_indices': [column_to_index[col] for col in (
                classification['identifier_columns'] + 
                classification['excluded_features']
            ) if col in column_to_index],
            'high_cardinality_indices': [column_to_index[col] for col in classification['high_cardinality_features'] if col in column_to_index]
        }
        
        # 添加特征名称映射
        indices.update({
            'label_features': classification['label_columns'],
            'categorical_features': classification['categorical_features'],
            'numerical_features': classification['numerical_features'],
            'excluded_features': classification['identifier_columns'] + classification['excluded_features'],
            'high_cardinality_features': classification['high_cardinality_features']
        })
        
        # 添加SAINT模型需要的格式
        indices.update({
            'cat_idxs': indices['cat_indices'],
            'con_idxs': indices['num_indices'],
            'total_features': len(indices['cat_indices']) + len(indices['num_indices']),
            'input_columns': [all_columns[i] for i in indices['cat_indices']] + 
                           [all_columns[i] for i in indices['num_indices']]
        })
        
        print(f"   ✅ 类别特征索引: {len(indices['cat_indices'])} 个")
        print(f"      {indices['cat_indices']}")
        print(f"   ✅ 数值特征索引: {len(indices['num_indices'])} 个")
        print(f"      前10个: {indices['num_indices'][:10]}...")
        print(f"   ✅ 总输入特征: {indices['total_features']} 个")
        
        return indices
    
    def export_feature_mapping(self, df: pd.DataFrame, classification: Dict[str, List], 
                             indices: Dict, output_dir: str = "feature_analysis") -> str:
        """导出特征映射和配置"""
        print(f"\n💾 导出特征映射到 '{output_dir}'...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成SAINT配置文件
        saint_config = {
            'cat_dims': [],  # 需要在实际处理时计算
            'cat_idxs': indices['cat_indices'],
            'con_idxs': indices['num_indices'],
            'categorical_features': classification['categorical_features'],
            'numerical_features': classification['numerical_features'],
            'high_cardinality_features': classification['high_cardinality_features'],
            'target_column': classification['label_columns'][0] if classification['label_columns'] else 'device_mac'
        }
        
        saint_config_path = os.path.join(output_dir, "saint_feature_config.json")
        with open(saint_config_path, 'w', encoding='utf-8') as f:
            json.dump(saint_config, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ SAINT配置已保存: {saint_config_path}")
        
        return output_dir
    
    def process_dataframe(self, df: pd.DataFrame, output_dir: str = "feature_analysis") -> Dict:
        """完整处理DataFrame的特征分类流程"""
        print("🚀 开始完整的特征分类流程...")
        
        # 1. 分析结构
        structure_info = self.analyze_dataframe_structure(df)
        
        # 2. 自动分类
        classification = self.classify_features_automatically(df)
        
        # 3. 创建索引
        indices = self.create_feature_indices(df, classification)
        
        # 4. 导出结果
        export_path = self.export_feature_mapping(df, classification, indices, output_dir)
        
        # 整合结果
        results = {
            'structure_info': structure_info,
            'classification': classification,
            'indices': indices,
            'export_path': export_path
        }
        
        print("\n🎉 特征分类流程完成！")
        return results

def analyze_cic_iot_features(csv_file_path: str, output_dir: str = "feature_analysis") -> Dict:
    """分析CIC IoT CSV文件的特征"""
    print(f"📂 开始分析文件: {csv_file_path}")
    
    try:
        # 先读取前1000行了解结构
        df_sample = pd.read_csv(csv_file_path, nrows=1000)
        print(f"   样本数据形状: {df_sample.shape}")
        
        # 初始化分类器
        classifier = CICIoTFeatureClassifier()
        
        # 处理样本数据
        results = classifier.process_dataframe(df_sample, output_dir)
        
        print(f"\n📋 分析完成！结果保存在: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    print("🧪 特征分类模块测试") 