#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目完整性测试脚本
测试整个 SAINT 智能家居设备分类项目的各个组件
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import traceback
from config import Config

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def test_imports():
    """测试所有必要的导入"""
    print_separator("1. 测试导入")
    
    try:
        from saint.data_utils import data_prep_custom, DataSetCatCon
        print("✓ saint.data_utils 导入成功")
        
        from saint.models import SAINT
        print("✓ saint.models 导入成功")
        
        from saint.augmentations import embed_data_mask
        print("✓ saint.augmentations 导入成功")
        
        import config
        print("✓ config 导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_data_files():
    """测试数据文件是否存在"""
    print_separator("2. 测试数据文件")
    
    config = Config()
    
    # 检查处理后的数据文件
    data_files = {
        '训练数据': config.TRAIN_DATA_PATH,
        '验证数据': config.VAL_DATA_PATH,
        '测试数据': config.TEST_DATA_PATH,
        '特征信息': config.FEATURE_INFO_PATH
    }
    
    all_exist = True
    for name, path in data_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"✓ {name}: {path} (大小: {size:.1f}MB)")
        else:
            print(f"✗ {name}: {path} 不存在")
            all_exist = False
    
    return all_exist

def test_data_loading():
    """测试数据加载"""
    print_separator("3. 测试数据加载")
    
    try:
        from saint.data_utils import data_prep_custom, DataSetCatCon
        
        config = Config()
        
        # 加载特征信息
        print("加载特征信息...")
        feature_info = pd.read_csv(config.FEATURE_INFO_PATH)
        print(f"特征信息形状: {feature_info.shape}")
        print(f"分类特征: {feature_info['feature_name'][feature_info['is_categorical']].tolist()}")
        print(f"数值特征: {feature_info['feature_name'][~feature_info['is_categorical']].tolist()}")
        
        # 加载小量数据进行测试
        print("\n加载训练数据样本...")
        train_data = pd.read_csv(config.TRAIN_DATA_PATH, nrows=1000)
        print(f"训练数据形状: {train_data.shape}")
        print(f"设备类别分布:\n{train_data['device_category'].value_counts().head()}")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        traceback.print_exc()
        return False

def test_model_initialization():
    """测试模型初始化"""
    print_separator("4. 测试模型初始化")
    
    try:
        from saint.models import SAINT
        from saint.data_utils import data_prep_custom, DataSetCatCon
        
        config = Config()
        
        # 加载小量数据进行测试
        print("准备测试数据...")
        feature_info = pd.read_csv(config.FEATURE_INFO_PATH)
        train_data = pd.read_csv(config.TRAIN_DATA_PATH, nrows=1000)
        
        cat_cols = feature_info['feature_name'][feature_info['is_categorical']].tolist()
        con_cols = feature_info['feature_name'][~feature_info['is_categorical']].tolist()
        
        X = train_data[cat_cols + con_cols]
        y = train_data['device_category']
        
        print(f"特征数量: 分类={len(cat_cols)}, 数值={len(con_cols)}")
        
        # 数据预处理
        print("数据预处理...")
        cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, bins = data_prep_custom(
            X=X, y=y, cat_cols=cat_cols, con_cols=con_cols, task='multiclass'
        )
        
        print(f"类别维度: {cat_dims}")
        print(f"类别数量: {len(set(y))}")
        
        # 初始化模型
        print("初始化模型...")
        model = SAINT(
            categories=tuple(cat_dims),
            num_continuous=len(con_cols),
            dim=config.model_params['dim'],
            depth=config.model_params['depth'],
            heads=config.model_params['heads'],
            categorical_embedding_type='ft',
            numerical_embedding_type='ple',
            bins=bins,
            attentiontype=config.model_params['attentiontype'],
            y_dim=len(set(y))
        )
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 模型初始化成功")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        return True, (model, cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, train_mean, train_std, bins)
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        traceback.print_exc()
        return False, None

def test_training_loop(model_data):
    """测试训练循环"""
    print_separator("5. 测试训练循环")
    
    try:
        from saint.augmentations import embed_data_mask
        import torch.optim as optim
        from torch import nn
        
        model, cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, train_mean, train_std, bins = model_data
        config = Config()
        
        # 创建数据加载器
        print("创建数据加载器...")
        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
        
        train_ds = DataSetCatCon(X_train, y_train, cat_idxs, task='clf', continuous_mean_std=continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        
        valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, task='clf', continuous_mean_std=continuous_mean_std)
        validloader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=0)
        
        print(f"训练批次数: {len(trainloader)}")
        print(f"验证批次数: {len(validloader)}")
        
        # 设置设备和优化器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config.training_params['lr'])
        criterion = nn.CrossEntropyLoss()
        
        # 快速训练测试（只训练2个epoch）
        print("\n开始快速训练测试...")
        for epoch in range(2):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i, data in enumerate(trainloader):
                if i >= 5:  # 只测试前5个批次
                    break
                    
                optimizer.zero_grad()
                x_categ, x_cont, y_gts, cat_mask, con_mask = [d.to(device) for d in data]
                
                # 前向传播
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
                reps = model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                
                loss = criterion(y_outs, y_gts.squeeze())
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
        
        # 验证测试
        print("\n验证测试...")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(validloader):
                if i >= 3:  # 只测试前3个批次
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
        print(f"测试准确率: {accuracy:.2f}%")
        
        print("✓ 训练循环测试成功")
        return True
    except Exception as e:
        print(f"✗ 训练循环测试失败: {e}")
        traceback.print_exc()
        return False

def test_optimized_training():
    """测试优化的训练脚本"""
    print_separator("6. 测试优化训练脚本")
    
    try:
        import train_optimized
        print("✓ train_optimized.py 可以导入")
        
        # 检查关键函数是否存在
        functions_to_check = ['create_model', 'create_data_loaders', 'train_epoch', 'validate']
        for func_name in functions_to_check:
            if hasattr(train_optimized, func_name):
                print(f"✓ {func_name} 函数存在")
            else:
                print(f"✗ {func_name} 函数缺失")
        
        return True
    except Exception as e:
        print(f"✗ 优化训练脚本测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("SAINT 智能家居设备分类项目 - 完整性测试")
    print("="*60)
    
    all_passed = True
    
    # 1. 测试导入
    if not test_imports():
        all_passed = False
        return False
    
    # 2. 测试数据文件
    if not test_data_files():
        all_passed = False
        print("\n⚠️  数据文件缺失，请先运行 data-processing.py 生成数据")
        return False
    
    # 3. 测试数据加载
    if not test_data_loading():
        all_passed = False
        return False
    
    # 4. 测试模型初始化
    model_success, model_data = test_model_initialization()
    if not model_success:
        all_passed = False
        return False
    
    # 5. 测试训练循环
    if not test_training_loop(model_data):
        all_passed = False
        return False
    
    # 6. 测试优化训练脚本
    if not test_optimized_training():
        all_passed = False
    
    # 总结
    print_separator("测试总结")
    if all_passed:
        print("🎉 所有测试通过！项目准备就绪")
        print("\n下一步建议:")
        print("1. 运行完整训练: python train_optimized.py")
        print("2. 或使用开发模式: python train_optimized.py --dev")
    else:
        print("❌ 部分测试失败，请检查上述错误信息")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 