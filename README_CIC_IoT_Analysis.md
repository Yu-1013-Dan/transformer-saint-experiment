# CIC IoT 2024 数据集分析和预筛选指南

## 📋 概述

这个脚本专门用于处理CIC IoT 2024数据集，帮助您：
1. **统计分析**所有设备和流量分布
2. **自动识别**智能家居设备
3. **手动筛选**不确定的设备
4. **数据平衡**处理（解决流量不均衡问题）
5. **保存**处理后的数据供SAINT模型使用

## 🚀 快速开始

### 1. 准备工作

确保您的数据文件路径在 `config.py` 中正确配置：

```python
# 修改为您的实际数据路径
DATA_ROOT = "/your/path/to/CIC/data"
CSV_FILES = [
    os.path.join(DATA_ROOT, "BenignTraffic.csv"),
    os.path.join(DATA_ROOT, "BenignTraffic1.csv"),
    # ... 更多文件
]
```

### 2. 运行分析脚本

```bash
# 开发模式（快速测试）
python cic_iot_data_analysis.py

# 或者先设置config.py中的DEVELOPMENT_MODE = False进行完整分析
```

## 📊 脚本功能详解

### 🔍 1. 数据加载与结构分析

脚本会自动：
- 加载所有CSV文件并合并
- 检测数据结构和列名
- 自动识别设备标识列（通常包含'device', 'mac', 'label'等关键词）
- 显示内存使用情况

### 📈 2. 设备分布统计

**输出内容：**
- 总设备数和总记录数
- Top 20设备列表（按流量记录数排序）
- 设备记录数的分布特征（最大、最小、中位数、平均值）
- 设备记录数区间分布（<100, 100-1K, 1K-10K, 10K-50K, 50K-100K, >100K）

**保存文件：**
- `iot_analysis_output/device_distribution.csv` - 完整设备统计

### 🏠 3. 智能家居设备识别

**自动识别类别：**
- **📹 Camera**: 相机类设备（Arlo, Nest, Yi, Wyze等）
- **🔊 Speaker**: 音箱类设备（Echo, Google, Sonos等）
- **💡 Lighting**: 照明设备（Bulb, Light, LIFX, Hue等）
- **🔌 Plug**: 插座设备（GoSund, Teckin, Wemo等）
- **📡 Sensor**: 传感器设备（Weather, Motion等）
- **🏠 Hub**: 网关集线器（SmartThings, Bridge等）
- **🏠 Appliance**: 家电设备（TV, Coffee Maker, Roomba等）
- **📺 Entertainment**: 娱乐设备（Smart TV, Streaming等）

**自动排除：**
- 计算机设备（PC, Laptop等）
- 手机平板（iPhone, Android等）
- 网络设备（Router, Switch等）
- 办公设备（Printer, Scanner等）

### 🤔 4. 手动设备筛选

对于脚本无法自动判断的设备，您可以：

**交互式选择：**
```
请选择智能家居设备 (例如: 1,camera 3,speaker 5,lighting):
1,camera 3,speaker 5,lighting 7,hub
```

**输入格式：**
- `设备编号,类别` 用空格分隔多个选择
- 类别选项：`camera`, `speaker`, `lighting`, `plug`, `sensor`, `hub`, `appliance`, `entertainment`

### ⚖️ 5. 数据平衡处理

**三种策略：**

1. **Undersample (下采样)** 🔽 - 推荐
   - 将高流量设备降采样到指定上限
   - 保持数据质量，减少计算量

2. **Oversample (上采样)** 🔼
   - 通过重复采样增加低流量设备数据
   - 可能导致过拟合

3. **Hybrid (混合策略)** ⚖️
   - 结合上下采样，平衡各设备数据量

**默认参数：**
- 最小样本数：1,000条/设备
- 最大样本数：10,000条/设备

### 💾 6. 输出文件

**主要输出：**
- `cic_iot_smart_home_processed.csv` - 平衡后的智能家居数据
- `smart_home_device_mapping.csv` - 设备类别映射表
- `device_distribution.csv` - 完整设备统计
- `processing_report.json` - 处理过程报告

## 📁 输出目录结构

```
iot_analysis_output/
├── cic_iot_smart_home_processed.csv      # 主数据文件
├── smart_home_device_mapping.csv         # 设备映射
├── device_distribution.csv               # 设备统计
└── processing_report.json                # 处理报告
```

## 🔄 后续处理流程

完成数据预筛选后，使用处理后的数据进行特征工程：

```bash
# 使用预处理后的数据
python data-processing.py --input iot_analysis_output/cic_iot_smart_home_processed.csv
```

## ⚙️ 配置选项

### 开发模式设置

```python
# config.py
DEVELOPMENT_MODE = True          # 快速测试模式
DEV_NROWS_PER_FILE = 50000      # 每文件读取行数限制
```

### 自定义设备关键词

如果需要添加新的设备类型或关键词，可以修改 `cic_iot_data_analysis.py` 中的：

```python
self.smart_home_keywords = {
    'camera': ['camera', 'cam', 'arlo', '...', 'your_camera_keyword'],
    # 添加新类别
    'your_category': ['keyword1', 'keyword2', '...']
}
```

## 🛠️ 故障排除

### 常见问题

1. **找不到设备列**
   - 检查数据文件是否包含设备标识列
   - 手动指定device_column_name参数

2. **内存不足**
   - 启用开发模式限制数据量
   - 减少DEV_NROWS_PER_FILE值

3. **没有识别到智能家居设备**
   - 检查设备名称格式
   - 考虑添加自定义关键词

4. **文件编码问题**
   - 输出文件使用UTF-8-BOM编码
   - Excel兼容性良好

## 📊 预期结果

运行成功后，您应该得到：
- ✅ 纯净的智能家居设备数据集
- ✅ 平衡的设备流量分布
- ✅ 详细的处理统计报告
- ✅ 可直接用于SAINT模型训练的数据

## 🎯 下一步

完成数据预筛选后，您可以：
1. 使用 `data-processing.py` 进行特征工程
2. 添加PLE bin计算功能（如前面讨论的）
3. 进行SAINT模型训练 