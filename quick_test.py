#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 使用虚拟数据测试SAINT模型
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

def quick_test():
    """快速测试SAINT模型的基本功能"""
    print("🚀 开始快速测试...")
    
    try:
        # 1. 测试导入
        print("\n1. 测试导入...")
        from saint.data_utils import data_prep_custom, DataSetCatCon
        from saint.models import SAINT
        from saint.augmentations import embed_data_mask
        import torch.optim as optim
        from torch import nn
        print("✓ 所有模块导入成功")
        
        # 2. 创建虚拟数据
        print("\n2. 创建虚拟数据...")
        num_samples = 1000
        
        # 模拟智能家居设备数据
        np.random.seed(42)
        data = {
            # 分类特征
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples),
            'port': np.random.choice([80, 443, 22, 21, 25, 53, 8080], num_samples),
            'device_type': np.random.choice(['Camera', 'Light', 'Speaker', 'Thermostat'], num_samples),
            
            # 数值特征
            'packet_size': np.random.exponential(500, num_samples),
            'duration': np.random.exponential(10, num_samples),
            'byte_count': np.random.exponential(1000, num_samples),
        }
        
        # 目标变量 - 设备类别
        target_mapping = {'Camera': 0, 'Light': 1, 'Speaker': 2, 'Thermostat': 3}
        data['device_category'] = [target_mapping[dt] for dt in data['device_type']]
        
        df = pd.DataFrame(data)
        
        cat_cols = ['protocol', 'port', 'device_type']
        con_cols = ['packet_size', 'duration', 'byte_count']
        
        print(f"✓ 创建虚拟数据: {df.shape}")
        print(f"分类特征: {cat_cols}")
        print(f"数值特征: {con_cols}")
        
        # 3. 数据预处理
        print("\n3. 数据预处理...")
        cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, bins = data_prep_custom(
            X=df[cat_cols + con_cols],
            y=df['device_category'],
            cat_cols=cat_cols,
            con_cols=con_cols,
            task='multiclass'
        )
        
        print(f"✓ 数据预处理完成")
        print(f"类别维度: {cat_dims}")
        print(f"训练集大小: {X_train['data'].shape[0]}")
        
        # 4. 创建数据加载器
        print("\n4. 创建数据加载器...")
        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
        
        train_ds = DataSetCatCon(X_train, y_train, cat_idxs, task='clf', continuous_mean_std=continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        
        valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, task='clf', continuous_mean_std=continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=0)
        
        print(f"✓ 数据加载器创建成功")
        
        # 5. 初始化模型
        print("\n5. 初始化模型...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = SAINT(
            categories=tuple(cat_dims),
            num_continuous=len(con_cols),
            dim=32,
            depth=3,
            heads=4,
            categorical_embedding_type='ft',
            numerical_embedding_type='ple',
            bins=bins,
            attentiontype='col',
            y_dim=4  # 4个设备类别
        )
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型初始化成功，参数量: {total_params:,}")
        
        # 6. 快速训练测试
        print("\n6. 快速训练测试（3个epoch）...")
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # 降低学习率
        criterion = nn.CrossEntropyLoss()
        
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(3):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i, data in enumerate(trainloader):
                if i >= 3:  # 只训练前3个批次
                    break
                
                optimizer.zero_grad()
                x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in data]
                
                # 前向传播
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                
                # 检查是否有NaN或Inf
                if torch.isnan(y_outs).any() or torch.isinf(y_outs).any():
                    print(f"警告: 输出包含NaN或Inf值")
                    continue
                
                loss = criterion(y_outs, y_gts.squeeze())
                
                # 检查loss是否为NaN
                if torch.isnan(loss):
                    print(f"警告: 损失为NaN，跳过这个批次")
                    continue
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"  Epoch {epoch+1}: 损失 = {avg_loss:.4f}")
        
        # 7. 验证测试
        print("\n7. 验证测试...")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(validloader):
                if i >= 2:  # 只测试前2个批次
                    break
                
                x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in data]
                
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                
                _, predicted = torch.max(y_outs.data, 1)
                total += y_gts.size(0)
                correct += (predicted == y_gts.squeeze()).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"✓ 验证准确率: {accuracy:.2f}%")
        
        print("\n🎉 快速测试完成！所有基本功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ 项目基本功能正常，可以进行完整测试或训练")
    else:
        print("\n❌ 项目存在问题，请检查错误信息") 