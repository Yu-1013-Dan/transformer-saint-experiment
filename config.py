"""
SAINT项目配置文件
统一管理数据路径、模型参数和训练配置
"""
import os

# === 数据路径配置 ===
DATA_ROOT = "/mnt/d/数据集/CIC"  # 用户需要根据实际情况修改
PROCESSED_DATA_DIR = "processed_data"

# CSV文件列表 (用户需要根据实际情况调整)
CSV_FILES = [
    os.path.join(DATA_ROOT, "BenignTraffic.csv"),
    os.path.join(DATA_ROOT, "BenignTraffic1.csv"),
    os.path.join(DATA_ROOT, "BenignTraffic2.csv"),
    os.path.join(DATA_ROOT, "BenignTraffic3.csv")
]

# === 数据预处理配置 ===
TARGET_COLUMN = 'target_device_type'
RAW_DEVICE_NAME_SOURCE_COLUMN = 'device_mac'

# 开发模式：使用少量数据快速测试
DEVELOPMENT_MODE = True
DEV_NROWS_PER_FILE = 50000  # 每个文件只读取5万行用于快速开发

# === 模型配置 ===
MODEL_CONFIG = {
    'embedding_size': 32,
    'transformer_depth': 6,
    'attention_heads': 8,
    'attention_dropout': 0.1,
    'ff_dropout': 0.1,
    'categorical_embedding_type': 'saint',  # 'saint' or 'ft'
    'numerical_embedding_type': 'mlp',      # 'mlp' or 'ple'
    'attentiontype': 'colrow',              # 'col', 'row', 'colrow'
    'final_mlp_style': 'common'             # 'common' or 'sep'
}

# === 训练配置 ===
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 256,
    'lr': 0.0001,
    'optimizer': 'AdamW',  # 'AdamW', 'Adam', 'SGD'
    'scheduler': 'cosine', # 'cosine', 'linear'
    'device': 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
}

# === 性能优化配置 ===
PERFORMANCE_CONFIG = {
    'num_workers': 4,           # DataLoader工作进程数
    'pin_memory': True,         # 加速GPU数据传输
    'mixed_precision': True,    # 混合精度训练
    'gradient_accumulation_steps': 1,  # 梯度累积
    'max_grad_norm': 1.0       # 梯度裁剪
}

# === 自动性能调整 ===
def auto_adjust_config(num_features, num_samples):
    """根据数据规模自动调整配置"""
    config = MODEL_CONFIG.copy()
    
    # 特征数量过多时的调整
    if num_features > 100:
        config['embedding_size'] = min(16, config['embedding_size'])
        TRAINING_CONFIG['batch_size'] = min(128, TRAINING_CONFIG['batch_size'])
        print(f"🔧 检测到高维特征({num_features})，自动调整嵌入维度为{config['embedding_size']}")
    
    # 样本数量较少时的调整
    if num_samples < 100000:
        config['transformer_depth'] = min(3, config['transformer_depth'])
        config['attention_heads'] = min(4, config['attention_heads'])
        print(f"🔧 检测到小数据集({num_samples})，自动调整模型深度为{config['transformer_depth']}")
    
    # 行注意力特殊调整
    if config['attentiontype'] in ['row', 'colrow']:
        config['transformer_depth'] = 1
        config['attention_dropout'] = 0.8
        config['ff_dropout'] = 0.8
        print(f"🔧 使用{config['attentiontype']}注意力，应用特殊配置")
    
    return config 