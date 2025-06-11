#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC IoT 2024 数据集分析和预筛选脚本
主要功能：
1. 统计所有设备和流量分布
2. 识别和筛选智能家居设备
3. 进行数据平衡处理
4. 保存预处理后的数据供后续处理
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import re
from config import CSV_FILES, DATA_ROOT, DEVELOPMENT_MODE, DEV_NROWS_PER_FILE

# 可选的可视化依赖
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("📊 注意: matplotlib未安装，跳过可视化功能")

class CICIoTDataAnalyzer:
    def __init__(self, csv_files, output_dir="iot_analysis_output"):
        """
        初始化CIC IoT数据分析器
        
        Args:
            csv_files: CSV文件路径列表
            output_dir: 输出目录
        """
        self.csv_files = csv_files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 智能家居设备关键词（用于自动识别）
        self.smart_home_keywords = {
            # 相机类
            'camera': ['camera', 'cam', 'arlo', 'nest', 'yi', 'wyze', 'netatmo', 'dlink', 'amcrest', 'tapo'],
            # 音箱类  
            'speaker': ['echo', 'speaker', 'alexa', 'google', 'nest', 'sonos', 'harman', 'kardon'],
            # 灯具类
            'lighting': ['bulb', 'light', 'lamp', 'lifx', 'hue', 'strip', 'lumiman'],
            # 插座类
            'plug': ['plug', 'socket', 'outlet', 'gosund', 'teckin', 'yutron', 'wemo'],
            # 传感器类
            'sensor': ['sensor', 'weather', 'motion', 'door', 'window', 'temperature', 'humidity'],
            # 网关/集线器
            'hub': ['hub', 'bridge', 'gateway', 'base', 'station', 'smartthings', 'homebase'],
            # 家电类
            'appliance': ['tv', 'coffee', 'maker', 'purifier', 'roomba', 'vacuum', 'hvac', 'fan', 'humidifier'],
            # 娱乐设备
            'entertainment': ['tv', 'smart tv', 'streaming', 'roku', 'chromecast', 'apple tv']
        }
        
        # 非智能家居设备关键词（用于排除）
        self.non_smart_home_keywords = [
            'computer', 'laptop', 'desktop', 'pc', 'server', 'router', 'switch', 
            'phone', 'iphone', 'android', 'mobile', 'tablet', 'ipad',
            'printer', 'scanner', 'fax', 'mouse', 'keyboard', 'monitor',
            'unknown', 'generic', 'test', 'demo'
        ]

    def load_and_combine_data(self, nrows_per_file=None, sample_analysis=True):
        """
        加载和合并数据，支持采样分析模式
        
        Args:
            nrows_per_file: 每个文件读取的行数限制
            sample_analysis: 是否进行采样分析（先读取少量数据了解结构）
        """
        print("🔄 开始加载CIC IoT数据集...")
        
        # 如果是采样分析，先读取少量数据了解结构
        if sample_analysis:
            print("📊 采样分析模式：先读取少量数据了解数据结构...")
            nrows_sample = min(1000, nrows_per_file) if nrows_per_file else 1000
        else:
            nrows_sample = nrows_per_file
            
        all_dfs = []
        data_info = {}
        
        for i, filepath in enumerate(self.csv_files):
            if not os.path.exists(filepath):
                print(f"⚠️  文件不存在，跳过: {filepath}")
                continue
                
            print(f"📁 正在读取文件 {i+1}/{len(self.csv_files)}: {os.path.basename(filepath)}")
            
            try:
                # 先读取少量数据检查结构
                df_sample = pd.read_csv(filepath, nrows=100)
                print(f"   列数: {len(df_sample.columns)}")
                print(f"   主要列: {list(df_sample.columns[:10])}")
                
                # 检查设备标识列
                device_columns = [col for col in df_sample.columns 
                                if any(keyword in col.lower() for keyword in ['device', 'mac', 'label'])]
                print(f"   设备相关列: {device_columns}")
                
                # 读取实际数据
                df = pd.read_csv(filepath, nrows=nrows_sample)
                
                # 处理可能的索引列
                if df.columns[0].startswith('Unnamed:'):
                    df = df.iloc[:, 1:]
                
                all_dfs.append(df)
                data_info[filepath] = {
                    'shape': df.shape,
                    'device_columns': device_columns,
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
                }
                
                print(f"   ✅ 加载完成: {df.shape} 行数据，内存使用: {data_info[filepath]['memory_usage']:.1f}MB")
                
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("❌ 没有成功加载任何数据文件！")
        
        # 合并数据
        print("🔗 正在合并数据...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        total_memory = combined_df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"✅ 数据合并完成:")
        print(f"   总形状: {combined_df.shape}")
        print(f"   总内存使用: {total_memory:.1f}MB")
        print(f"   列名示例: {list(combined_df.columns[:15])}")
        
        return combined_df, data_info

    def analyze_device_distribution(self, df, device_column_name=None):
        """
        分析设备分布情况
        
        Args:
            df: 数据框
            device_column_name: 设备列名，如果为None则自动检测
        """
        print("\n📊 开始分析设备分布...")
        
        # 自动检测设备列
        if device_column_name is None:
            possible_device_cols = [col for col in df.columns 
                                  if any(keyword in col.lower() for keyword in 
                                       ['device', 'mac', 'label', 'name'])]
            
            if not possible_device_cols:
                print("❌ 未找到设备标识列！请手动指定device_column_name")
                return None
            
            # 选择最可能的设备列（通常包含设备名称信息）
            device_column_name = possible_device_cols[0]
            print(f"🎯 自动检测设备列: {device_column_name}")
            
            # 如果有多个候选，显示给用户选择
            if len(possible_device_cols) > 1:
                print(f"   其他候选列: {possible_device_cols[1:]}")
        
        if device_column_name not in df.columns:
            print(f"❌ 指定的设备列 '{device_column_name}' 不存在！")
            return None
        
        # 统计设备分布
        device_counts = df[device_column_name].value_counts()
        total_devices = len(device_counts)
        total_records = len(df)
        
        print(f"\n📈 设备统计结果:")
        print(f"   总设备数: {total_devices}")
        print(f"   总记录数: {total_records:,}")
        print(f"   平均每设备记录数: {total_records/total_devices:.1f}")
        
        # 显示Top 20设备
        print(f"\n🏆 Top 20 设备 (按记录数排序):")
        for i, (device, count) in enumerate(device_counts.head(20).items(), 1):
            percentage = count / total_records * 100
            print(f"   {i:2d}. {device:<40} {count:>8,} 条 ({percentage:5.1f}%)")
        
        # 统计分布特征
        print(f"\n📊 分布特征:")
        print(f"   最大记录数: {device_counts.max():,}")
        print(f"   最小记录数: {device_counts.min():,}")
        print(f"   中位数记录数: {device_counts.median():.0f}")
        print(f"   平均记录数: {device_counts.mean():.1f}")
        print(f"   标准差: {device_counts.std():.1f}")
        
        # 分析记录数分布区间
        bins = [0, 100, 1000, 10000, 50000, 100000, float('inf')]
        labels = ['<100', '100-1K', '1K-10K', '10K-50K', '50K-100K', '>100K']
        device_counts_binned = pd.cut(device_counts, bins=bins, labels=labels, right=False)
        bin_distribution = device_counts_binned.value_counts().sort_index()
        
        print(f"\n📋 设备记录数分布区间:")
        for bin_label, count in bin_distribution.items():
            percentage = count / total_devices * 100
            print(f"   {bin_label:<10}: {count:>4} 个设备 ({percentage:5.1f}%)")
        
        # 保存设备统计
        output_file = os.path.join(self.output_dir, "device_distribution.csv")
        device_stats = pd.DataFrame({
            'Device': device_counts.index,
            'Record_Count': device_counts.values,
            'Percentage': (device_counts.values / total_records * 100).round(2)
        })
        device_stats.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 设备统计已保存到: {output_file}")
        
        return {
            'device_column': device_column_name,
            'device_counts': device_counts,
            'total_devices': total_devices,
            'total_records': total_records,
            'device_stats': device_stats
        }

    def identify_smart_home_devices(self, device_counts, min_records=1000):
        """
        识别智能家居设备
        
        Args:
            device_counts: 设备计数Series
            min_records: 最小记录数阈值
        """
        print(f"\n🏠 开始识别智能家居设备 (最小记录数: {min_records:,})...")
        
        smart_home_devices = {}
        non_smart_home_devices = []
        uncertain_devices = []
        
        for device, count in device_counts.items():
            if count < min_records:
                continue
                
            device_lower = str(device).lower()
            
            # 检查是否为非智能家居设备
            is_non_smart = any(keyword in device_lower for keyword in self.non_smart_home_keywords)
            if is_non_smart:
                non_smart_home_devices.append((device, count))
                continue
            
            # 检查是否为智能家居设备
            device_category = None
            for category, keywords in self.smart_home_keywords.items():
                if any(keyword in device_lower for keyword in keywords):
                    device_category = category
                    break
            
            if device_category:
                if device_category not in smart_home_devices:
                    smart_home_devices[device_category] = []
                smart_home_devices[device_category].append((device, count))
            else:
                uncertain_devices.append((device, count))
        
        # 打印识别结果
        print(f"\n✅ 智能家居设备识别结果:")
        total_smart_records = 0
        for category, devices in smart_home_devices.items():
            category_records = sum(count for _, count in devices)
            total_smart_records += category_records
            print(f"\n📱 {category.upper()} ({len(devices)} 设备, {category_records:,} 记录):")
            for device, count in sorted(devices, key=lambda x: x[1], reverse=True):
                print(f"   • {device:<50} {count:>8,} 条")
        
        print(f"\n❌ 非智能家居设备 ({len(non_smart_home_devices)} 设备):")
        for device, count in sorted(non_smart_home_devices, key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {device:<50} {count:>8,} 条")
        
        print(f"\n❓ 不确定设备 ({len(uncertain_devices)} 设备) - 需要手动检查:")
        for device, count in sorted(uncertain_devices, key=lambda x: x[1], reverse=True)[:15]:
            print(f"   • {device:<50} {count:>8,} 条")
        
        print(f"\n📊 总结:")
        print(f"   智能家居设备: {sum(len(devices) for devices in smart_home_devices.values())} 个")
        print(f"   智能家居记录: {total_smart_records:,} 条")
        print(f"   非智能家居设备: {len(non_smart_home_devices)} 个")
        print(f"   不确定设备: {len(uncertain_devices)} 个")
        
        return {
            'smart_home_devices': smart_home_devices,
            'non_smart_home_devices': non_smart_home_devices,
            'uncertain_devices': uncertain_devices,
            'total_smart_records': total_smart_records
        }

    def manual_device_selection(self, uncertain_devices, max_display=20):
        """
        手动选择不确定的设备
        
        Args:
            uncertain_devices: 不确定设备列表
            max_display: 最大显示设备数
        """
        print(f"\n🤔 手动设备选择 - 请判断以下设备是否为智能家居设备:")
        print("   输入格式: 设备编号,类别 (多个用空格分隔)")
        print("   类别选项: camera, speaker, lighting, plug, sensor, hub, appliance, entertainment")
        print("   跳过设备: 直接按回车")
        
        selected_devices = {}
        
        display_devices = uncertain_devices[:max_display]
        for i, (device, count) in enumerate(display_devices, 1):
            print(f"\n{i:2d}. {device} ({count:,} 条记录)")
            
        print(f"\n请选择智能家居设备 (例如: 1,camera 3,speaker 5,lighting):")
        user_input = input().strip()
        
        if user_input:
            try:
                selections = user_input.split()
                for selection in selections:
                    if ',' in selection:
                        idx_str, category = selection.split(',', 1)
                        idx = int(idx_str) - 1
                        if 0 <= idx < len(display_devices):
                            device, count = display_devices[idx]
                            if category in self.smart_home_keywords:
                                if category not in selected_devices:
                                    selected_devices[category] = []
                                selected_devices[category].append((device, count))
                                print(f"   ✅ {device} -> {category}")
                            else:
                                print(f"   ❌ 无效类别: {category}")
                        else:
                            print(f"   ❌ 无效设备编号: {idx_str}")
                    else:
                        print(f"   ❌ 格式错误: {selection}")
            except Exception as e:
                print(f"   ❌ 输入解析错误: {e}")
        
        return selected_devices

    def balance_device_data(self, df, smart_devices_dict, device_column, 
                          min_samples=1000, max_samples=10000, balance_strategy='undersample'):
        """
        平衡设备数据
        
        Args:
            df: 原始数据框
            smart_devices_dict: 智能家居设备字典
            device_column: 设备列名
            min_samples: 最小样本数
            max_samples: 最大样本数
            balance_strategy: 平衡策略 ('undersample', 'oversample', 'hybrid')
        """
        print(f"\n⚖️  开始进行数据平衡 (策略: {balance_strategy})...")
        print(f"   目标范围: {min_samples:,} - {max_samples:,} 样本/设备")
        
        # 收集所有智能家居设备名称
        all_smart_devices = []
        for category, devices in smart_devices_dict.items():
            all_smart_devices.extend([device for device, _ in devices])
        
        # 过滤数据只保留智能家居设备
        smart_home_df = df[df[device_column].isin(all_smart_devices)].copy()
        print(f"   智能家居数据: {smart_home_df.shape[0]:,} 条记录")
        
        balanced_dfs = []
        device_final_counts = {}
        
        for device in all_smart_devices:
            device_df = smart_home_df[smart_home_df[device_column] == device]
            original_count = len(device_df)
            
            if original_count < min_samples:
                print(f"   ⚠️  {device}: {original_count} < {min_samples} (跳过)")
                continue
            
            # 确定目标样本数
            if balance_strategy == 'undersample':
                target_count = min(original_count, max_samples)
            elif balance_strategy == 'oversample':
                target_count = max(original_count, min_samples)
                target_count = min(target_count, max_samples)  # 仍然限制最大值
            else:  # hybrid
                target_count = min(max(original_count, min_samples), max_samples)
            
            # 应用采样
            if target_count < original_count:
                # 下采样
                sampled_df = device_df.sample(n=target_count, random_state=42)
                action = "下采样"
            elif target_count > original_count:
                # 上采样 (重复采样)
                n_repeats = target_count // original_count
                n_remainder = target_count % original_count
                
                repeated_dfs = [device_df] * n_repeats
                if n_remainder > 0:
                    repeated_dfs.append(device_df.sample(n=n_remainder, random_state=42))
                
                sampled_df = pd.concat(repeated_dfs, ignore_index=True)
                action = "上采样"
            else:
                # 保持不变
                sampled_df = device_df
                action = "保持"
            
            balanced_dfs.append(sampled_df)
            device_final_counts[device] = len(sampled_df)
            
            print(f"   📊 {device:<40} {original_count:>6,} -> {len(sampled_df):>6,} ({action})")
        
        # 合并平衡后的数据
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print(f"\n✅ 数据平衡完成:")
        print(f"   原始数据: {smart_home_df.shape[0]:,} 条")
        print(f"   平衡后数据: {balanced_df.shape[0]:,} 条")
        print(f"   保留设备: {len(device_final_counts)} 个")
        print(f"   平均每设备: {balanced_df.shape[0] / len(device_final_counts):.0f} 条")
        
        return balanced_df, device_final_counts

    def save_processed_data(self, df, smart_devices_dict, device_final_counts, 
                          output_filename="cic_iot_smart_home_processed.csv"):
        """
        保存处理后的数据和元数据
        
        Args:
            df: 处理后的数据框
            smart_devices_dict: 智能家居设备字典
            device_final_counts: 最终设备计数
            output_filename: 输出文件名
        """
        print(f"\n💾 开始保存处理后的数据...")
        
        # 保存主数据文件
        output_path = os.path.join(self.output_dir, output_filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 主数据已保存: {output_path}")
        print(f"      形状: {df.shape}")
        print(f"      大小: {os.path.getsize(output_path) / 1024**2:.1f} MB")
        
        # 保存设备映射文件
        device_mapping = []
        for category, devices in smart_devices_dict.items():
            for device, original_count in devices:
                final_count = device_final_counts.get(device, 0)
                device_mapping.append({
                    'Device': device,
                    'Category': category,
                    'Original_Count': original_count,
                    'Final_Count': final_count,
                    'Included': final_count > 0
                })
        
        mapping_df = pd.DataFrame(device_mapping)
        mapping_path = os.path.join(self.output_dir, "smart_home_device_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 设备映射已保存: {mapping_path}")
        
        # 保存处理报告
        report = {
            'processing_summary': {
                'total_smart_devices': len(device_final_counts),
                'total_records': len(df),
                'average_records_per_device': len(df) / len(device_final_counts),
                'categories': list(smart_devices_dict.keys()),
                'category_counts': {cat: len(devices) for cat, devices in smart_devices_dict.items()}
            }
        }
        
        import json
        report_path = os.path.join(self.output_dir, "processing_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   ✅ 处理报告已保存: {report_path}")
        
        return output_path, mapping_path, report_path

def main():
    """主函数"""
    print("🚀 CIC IoT 2024 数据集分析和预筛选")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = CICIoTDataAnalyzer(CSV_FILES)
    
    # 确定数据读取量
    if DEVELOPMENT_MODE:
        print(f"🔧 开发模式: 每文件读取 {DEV_NROWS_PER_FILE:,} 行")
        nrows = DEV_NROWS_PER_FILE
    else:
        print("🔥 生产模式: 读取全部数据")
        nrows = None
    
    try:
        # 1. 加载数据
        df, data_info = analyzer.load_and_combine_data(nrows_per_file=nrows)
        
        # 2. 分析设备分布
        device_analysis = analyzer.analyze_device_distribution(df)
        if device_analysis is None:
            return
        
        device_column = device_analysis['device_column']
        device_counts = device_analysis['device_counts']
        
        # 3. 识别智能家居设备
        smart_home_analysis = analyzer.identify_smart_home_devices(device_counts, min_records=1000)
        
        # 4. 手动选择不确定设备
        if smart_home_analysis['uncertain_devices']:
            print(f"\n发现 {len(smart_home_analysis['uncertain_devices'])} 个不确定设备")
            user_choice = input("是否需要手动选择智能家居设备? (y/n): ").strip().lower()
            
            if user_choice == 'y':
                manual_selections = analyzer.manual_device_selection(
                    smart_home_analysis['uncertain_devices']
                )
                # 合并手动选择结果
                for category, devices in manual_selections.items():
                    if category not in smart_home_analysis['smart_home_devices']:
                        smart_home_analysis['smart_home_devices'][category] = []
                    smart_home_analysis['smart_home_devices'][category].extend(devices)
        
        # 5. 数据平衡
        print(f"\n请选择数据平衡策略:")
        print("1. undersample - 下采样 (推荐)")
        print("2. oversample - 上采样")
        print("3. hybrid - 混合策略")
        
        strategy_choice = input("请选择 (1-3, 默认为1): ").strip()
        strategy_map = {'1': 'undersample', '2': 'oversample', '3': 'hybrid'}
        balance_strategy = strategy_map.get(strategy_choice, 'undersample')
        
        balanced_df, device_final_counts = analyzer.balance_device_data(
            df, 
            smart_home_analysis['smart_home_devices'],
            device_column,
            min_samples=1000,
            max_samples=10000,
            balance_strategy=balance_strategy
        )
        
        # 6. 保存处理后的数据
        output_files = analyzer.save_processed_data(
            balanced_df,
            smart_home_analysis['smart_home_devices'],
            device_final_counts
        )
        
        print(f"\n🎉 数据预处理完成！")
        print(f"📁 输出目录: {analyzer.output_dir}")
        print(f"📋 下一步: 使用 data-processing.py 进行特征工程")
        print(f"   python data-processing.py --input {output_files[0]}")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 