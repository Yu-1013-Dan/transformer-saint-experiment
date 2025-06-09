"""
SAINT优化训练脚本
集成性能优化、混合精度训练、配置管理等功能
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
import numpy as np
from tqdm import tqdm

# 导入项目模块
try:
    from config import MODEL_CONFIG, TRAINING_CONFIG, PERFORMANCE_CONFIG, auto_adjust_config
    from saint.models import SAINT
    from saint.data_utils import DataSetCatCon
    from saint.augmentations import embed_data_mask
    from saint.utils import classification_scores
    USE_CONFIG = True
except ImportError as e:
    print(f"⚠️ 配置导入失败: {e}")
    print("📌 使用默认配置")
    USE_CONFIG = False

def load_processed_data(data_dir="processed_data"):
    """加载预处理后的数据"""
    print("📂 加载预处理数据...")
    
    import pandas as pd
    import joblib
    
    # 加载特征数据
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_processed.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val_processed.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_processed.csv'))
    
    # 加载标签数据
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_processed.csv')).values.flatten()
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val_processed.csv')).values.flatten()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test_processed.csv')).values.flatten()
    
    # 加载预处理器
    label_encoders = joblib.load(os.path.join(data_dir, 'label_encoders.joblib'))
    
    print(f"✅ 数据加载完成: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoders

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, label_encoders):
    """创建优化的数据加载器"""
    print("🔄 创建数据加载器...")
    
    # 确定类别特征和连续特征的索引
    cat_cols = list(label_encoders.keys())
    all_cols = X_train.columns.tolist()
    cat_idxs = [all_cols.index(col) for col in cat_cols if col in all_cols]
    con_idxs = [i for i in range(len(all_cols)) if i not in cat_idxs]
    
    # 转换为数据集格式
    def prepare_data(X, y):
        return {
            'data': X.values,
            'mask': np.ones_like(X.values)  # 假设无缺失值
        }, {'data': y.reshape(-1, 1)}
    
    X_train_dict, y_train_dict = prepare_data(X_train, y_train)
    X_val_dict, y_val_dict = prepare_data(X_val, y_val)
    X_test_dict, y_test_dict = prepare_data(X_test, y_test)
    
    # 获取配置
    if USE_CONFIG:
        batch_size = TRAINING_CONFIG['batch_size']
        num_workers = PERFORMANCE_CONFIG['num_workers']
        pin_memory = PERFORMANCE_CONFIG['pin_memory']
    else:
        batch_size, num_workers, pin_memory = 256, 4, True
    
    # 创建数据集
    train_ds = DataSetCatCon(X_train_dict, y_train_dict, cat_idxs, 'clf')
    val_ds = DataSetCatCon(X_val_dict, y_val_dict, cat_idxs, 'clf')
    test_ds = DataSetCatCon(X_test_dict, y_test_dict, cat_idxs, 'clf')
    
    # 创建数据加载器 (优化配置)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"✅ 数据加载器创建完成")
    return train_loader, val_loader, test_loader, cat_idxs, con_idxs

def create_model(X_train, y_train, label_encoders):
    """创建并配置SAINT模型"""
    print("🤖 创建SAINT模型...")
    
    # 计算类别维度
    cat_dims = [len(enc.classes_) for enc in label_encoders.values()]
    cat_dims = [1] + cat_dims  # 添加[CLS] token
    
    num_continuous = len([col for col in X_train.columns if col not in label_encoders.keys()])
    num_classes = len(np.unique(y_train))
    
    # 自动调整配置
    if USE_CONFIG:
        config = auto_adjust_config(X_train.shape[1], X_train.shape[0])
    else:
        config = {
            'embedding_size': 32, 'transformer_depth': 6, 'attention_heads': 8,
            'attention_dropout': 0.1, 'ff_dropout': 0.1,
            'categorical_embedding_type': 'saint', 'numerical_embedding_type': 'mlp',
            'attentiontype': 'colrow', 'final_mlp_style': 'common'
        }
    
    # 创建模型
    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=num_continuous,
        dim=config['embedding_size'],
        depth=config['transformer_depth'],
        heads=config['attention_heads'],
        attn_dropout=config['attention_dropout'],
        ff_dropout=config['ff_dropout'],
        categorical_embedding_type=config['categorical_embedding_type'],
        numerical_embedding_type=config['numerical_embedding_type'],
        attentiontype=config['attentiontype'],
        final_mlp_style=config['final_mlp_style'],
        y_dim=num_classes
    )
    
    print(f"✅ 模型创建完成: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")
    return model, config

def train_model(model, train_loader, val_loader, config):
    """优化的训练循环"""
    print("🚀 开始训练...")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"📱 使用设备: {device}")
    
    # 获取训练配置
    if USE_CONFIG:
        epochs = TRAINING_CONFIG['epochs']
        lr = TRAINING_CONFIG['lr']
        optimizer_name = TRAINING_CONFIG['optimizer']
        use_amp = PERFORMANCE_CONFIG['mixed_precision']
        max_grad_norm = PERFORMANCE_CONFIG['max_grad_norm']
    else:
        epochs, lr, optimizer_name, use_amp, max_grad_norm = 100, 0.0001, 'AdamW', True, 1.0
    
    # 优化器和损失函数
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 训练记录
    best_val_acc = 0.0
    train_losses, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, data in enumerate(progress_bar):
            x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in data]
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            if use_amp:
                with autocast():
                    _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
                    reps = model.transformer(x_categ_enc, x_cont_enc)
                    y_reps = reps[:, 0, :]
                    y_outs = model.mlpfory(y_reps)
                    loss = criterion(y_outs, y_gts.squeeze())
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                loss = criterion(y_outs, y_gts.squeeze())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        if (epoch + 1) % 5 == 0:  # 每5个epoch验证一次
            val_acc, _ = classification_scores(model, val_loader, device, 'multiclass', False)
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_saint_model.pth')
                print(f"🎯 新的最佳验证准确率: {val_acc:.4f}")
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val_Acc={val_acc:.4f}, Best={best_val_acc:.4f}")
    
    print(f"✅ 训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    return model, train_losses, val_accs

def main():
    """主训练函数"""
    print("=" * 50)
    print("🤖 SAINT 智能家居设备分类训练")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. 加载数据
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = load_processed_data()
        
        # 2. 创建数据加载器
        train_loader, val_loader, test_loader, cat_idxs, con_idxs = create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test, label_encoders
        )
        
        # 3. 创建模型
        model, config = create_model(X_train, y_train, label_encoders)
        
        # 4. 训练模型
        trained_model, train_losses, val_accs = train_model(model, train_loader, val_loader, config)
        
        # 5. 最终测试
        print("🧪 最终测试评估...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_acc, test_auc = classification_scores(trained_model, test_loader, device, 'multiclass', False)
        
        print(f"\n🎉 训练完成!")
        print(f"   测试准确率: {test_acc:.4f}")
        print(f"   训练用时: {(time.time() - start_time)/60:.2f} 分钟")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 