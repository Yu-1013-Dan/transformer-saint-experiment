import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
import numpy as np

class KNNCrossSampleAttention(nn.Module):
    """
    k-NN跨样本注意力模块 - SAINT模型的核心创新
    
    实现步骤：
    A. 提取样本表示: 使用[CLS] Token向量为每个样本生成代表性向量
    B. 计算k-NN图: 基于代表性向量计算批次内样本间相似度，找出k个邻居
    C. 生成掩码: 创建注意力掩码，限制样本只能关注自身和k个邻居
    D. 执行带掩码的注意力: 应用掩码进行精准的局部注意力计算
    """
    
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=16,
        dropout=0.,
        k_neighbors=5,
        temperature=1.0,
        similarity_metric='cosine'
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.similarity_metric = similarity_metric
        
        # 用于生成Q、K、V的线性层
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        # 用于提取样本表示的投影层
        self.sample_repr_proj = nn.Linear(dim, dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def extract_sample_representations(self, x):
        """
        A. 提取样本表示: 使用[CLS] Token向量为每个样本生成代表性向量
        
        Args:
            x: [batch_size, n_features, dim] - 输入特征表示
            
        Returns:
            sample_reprs: [batch_size, dim] - 每个样本的代表性向量
        """
        # 假设第一个token是[CLS] token，或者我们可以使用全局池化
        # 这里我们使用平均池化来获得样本级别的表示
        sample_reprs = x.mean(dim=1)  # [batch_size, dim]
        
        # 通过投影层进一步处理
        sample_reprs = self.sample_repr_proj(sample_reprs)
        
        return sample_reprs
    
    def compute_knn_graph(self, sample_reprs):
        """
        B. 计算k-NN图: 基于代表性向量计算批次内样本间相似度
        
        Args:
            sample_reprs: [batch_size, dim] - 样本代表性向量
            
        Returns:
            knn_indices: [batch_size, k+1] - 每个样本的k个邻居索引（包含自身）
            knn_similarities: [batch_size, k+1] - 对应的相似度分数
        """
        batch_size = sample_reprs.shape[0]
        
        if self.similarity_metric == 'cosine':
            # 计算余弦相似度
            normalized_reprs = F.normalize(sample_reprs, p=2, dim=-1)
            similarity_matrix = torch.matmul(normalized_reprs, normalized_reprs.T)
        elif self.similarity_metric == 'euclidean':
            # 计算欧几里得距离（转换为相似度）
            distances = torch.cdist(sample_reprs, sample_reprs, p=2)
            similarity_matrix = -distances  # 距离越小，相似度越高
        else:
            # 默认使用点积相似度
            similarity_matrix = torch.matmul(sample_reprs, sample_reprs.T)
        
        # 温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        # 找出每个样本的top-k邻居（包含自身）
        k_actual = min(self.k_neighbors + 1, batch_size)  # +1是为了包含自身
        knn_similarities, knn_indices = torch.topk(
            similarity_matrix, k=k_actual, dim=-1, largest=True
        )
        
        return knn_indices, knn_similarities
    
    def generate_attention_mask(self, knn_indices, batch_size):
        """
        C. 生成掩码: 创建注意力掩码，只允许样本关注自身和k个邻居
        
        Args:
            knn_indices: [batch_size, k+1] - k-NN邻居索引
            batch_size: int - 批次大小
            
        Returns:
            attention_mask: [batch_size, batch_size] - 注意力掩码
        """
        # 初始化掩码矩阵，所有位置都被掩码（设为-inf）
        attention_mask = torch.full(
            (batch_size, batch_size), 
            float('-inf'), 
            device=knn_indices.device,
            dtype=torch.float
        )
        
        # 为每个样本的邻居位置设置为0（允许注意）
        for i in range(batch_size):
            neighbor_indices = knn_indices[i]  # [k+1]
            attention_mask[i, neighbor_indices] = 0.0
        
        return attention_mask
    
    def forward(self, x, return_attention_weights=False):
        """
        D. 执行带掩码的注意力: 应用k-NN掩码进行局部注意力计算
        
        Args:
            x: [batch_size, n_features, dim] - 输入特征
            return_attention_weights: bool - 是否返回注意力权重
            
        Returns:
            out: [batch_size, n_features, dim] - 输出特征
            attention_weights: (可选) 注意力权重
        """
        batch_size, n_features, dim = x.shape
        h = self.heads
        
        # Step A: 提取样本表示
        sample_reprs = self.extract_sample_representations(x)  # [batch_size, dim]
        
        # Step B: 计算k-NN图
        knn_indices, knn_similarities = self.compute_knn_graph(sample_reprs)
        
        # Step C: 生成注意力掩码
        cross_sample_mask = self.generate_attention_mask(knn_indices, batch_size)
        
        # Step D: 执行带掩码的跨样本注意力
        # 1. 生成Q、K、V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        # 2. 重塑为跨样本计算的形式
        # 将所有样本的特征concatenate起来进行跨样本注意力
        q_cross = rearrange(q, 'b h n d -> (b n) h d')  # [batch_size*n_features, heads, dim_head]
        k_cross = rearrange(k, 'b h n d -> (b n) h d')
        v_cross = rearrange(v, 'b h n d -> (b n) h d')
        
        # 3. 计算跨样本注意力分数
        # 这里我们需要重新组织计算方式来应用样本级别的掩码
        attention_outputs = []
        
        for feature_idx in range(n_features):
            # 为每个特征位置计算跨样本注意力
            q_feat = q[:, :, feature_idx, :]  # [batch_size, heads, dim_head]
            k_feat = k[:, :, feature_idx, :]  # [batch_size, heads, dim_head]
            v_feat = v[:, :, feature_idx, :]  # [batch_size, heads, dim_head]
            
            # 计算注意力分数
            sim = einsum('b h d, B h d -> b h B', q_feat, k_feat) * self.scale
            
            # 应用k-NN掩码
            sim = sim + cross_sample_mask.unsqueeze(1)  # [batch_size, heads, batch_size]
            
            # Softmax
            attn = sim.softmax(dim=-1)
            attn = self.dropout(attn)
            
            # 应用注意力到值
            out_feat = einsum('b h B, B h d -> b h d', attn, v_feat)
            attention_outputs.append(out_feat)
        
        # 4. 重组输出
        out = torch.stack(attention_outputs, dim=2)  # [batch_size, heads, n_features, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 5. 最终输出投影
        out = self.to_out(out)
        
        if return_attention_weights:
            return out, (knn_indices, knn_similarities, cross_sample_mask)
        else:
            return out


class EnhancedSAINTBlock(nn.Module):
    """
    增强的SAINT块，集成k-NN跨样本注意力
    """
    
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=16,
        dropout=0.,
        ff_dropout=0.,
        k_neighbors=5,
        use_knn_attention=True
    ):
        super().__init__()
        self.use_knn_attention = use_knn_attention
        
        # 原始的自注意力（特征内注意力）
        from .model import Attention, FeedForward, PreNorm, Residual
        
        self.self_attention = PreNorm(dim, Residual(
            Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        ))
        
        # k-NN跨样本注意力
        if use_knn_attention:
            self.knn_attention = PreNorm(dim, Residual(
                KNNCrossSampleAttention(
                    dim, heads=heads, dim_head=dim_head, 
                    dropout=dropout, k_neighbors=k_neighbors
                )
            ))
        
        # 前馈网络
        self.feed_forward = PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout)))
        
    def forward(self, x):
        # 1. 特征内自注意力
        x = self.self_attention(x)
        
        # 2. k-NN跨样本注意力（如果启用）
        if self.use_knn_attention:
            x = self.knn_attention(x)
        
        # 3. 前馈网络
        x = self.feed_forward(x)
        
        return x


# 辅助函数
def visualize_knn_graph(knn_indices, knn_similarities, sample_labels=None):
    """
    可视化k-NN图结构（用于调试和分析）
    
    Args:
        knn_indices: [batch_size, k+1] - 邻居索引
        knn_similarities: [batch_size, k+1] - 相似度分数
        sample_labels: 可选的样本标签
    """
    print("k-NN Graph Structure:")
    print("=" * 50)
    
    batch_size = knn_indices.shape[0]
    for i in range(batch_size):
        neighbors = knn_indices[i].cpu().numpy()
        similarities = knn_similarities[i].cpu().numpy()
        
        print(f"Sample {i}:")
        for j, (neighbor_idx, sim) in enumerate(zip(neighbors, similarities)):
            label_info = f" (label: {sample_labels[neighbor_idx]})" if sample_labels is not None else ""
            print(f"  Neighbor {j}: Sample {neighbor_idx}, Similarity: {sim:.4f}{label_info}")
        print() 