import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# 导入原有的组件
from .model import (
    Residual, PreNorm, GEGLU, FeedForward, Attention, 
    MLP, simple_MLP, ff_encodings
)
from .knn_attention import KNNCrossSampleAttention, EnhancedSAINTBlock


class EnhancedTransformer(nn.Module):
    """
    增强的Transformer，支持k-NN跨样本注意力
    """
    
    def __init__(
        self, 
        num_tokens, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        attn_dropout, 
        ff_dropout,
        k_neighbors=5,
        use_knn_attention=True,
        knn_start_layer=1  # 从第几层开始使用k-NN注意力
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_knn_attention = use_knn_attention
        self.knn_start_layer = knn_start_layer

        for layer_idx in range(depth):
            # 决定这一层是否使用k-NN注意力
            use_knn_this_layer = use_knn_attention and (layer_idx >= knn_start_layer)
            
            if use_knn_this_layer:
                # 使用增强的SAINT块
                layer = EnhancedSAINTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    k_neighbors=k_neighbors,
                    use_knn_attention=True
                )
                self.layers.append(layer)
            else:
                # 使用标准的注意力层
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, return_attention_info=False):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        
        attention_info = []
        
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, EnhancedSAINTBlock):
                # 增强的SAINT块
                x_before = x
                x = layer(x)
                if return_attention_info:
                    attention_info.append({
                        'layer': layer_idx,
                        'type': 'enhanced_saint',
                        'input_shape': x_before.shape,
                        'output_shape': x.shape
                    })
            else:
                # 标准的注意力层
                attn, ff = layer
                x = attn(x)
                x = ff(x)
                if return_attention_info:
                    attention_info.append({
                        'layer': layer_idx,
                        'type': 'standard_attention',
                        'input_shape': x.shape,
                        'output_shape': x.shape
                    })
        
        if return_attention_info:
            return x, attention_info
        return x


class EnhancedTabAttention(nn.Module):
    """
    增强版SAINT模型 - 集成k-NN跨样本注意力模块
    
    新增功能：
    1. k-NN跨样本注意力模块
    2. 可配置的注意力层级使用策略
    3. 更详细的注意力分析功能
    """
    
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=1,
        continuous_mean_std=None,
        attn_dropout=0.,
        ff_dropout=0.,
        lastmlp_dropout=0.,
        cont_embeddings='MLP',
        scalingfactor=10,
        attentiontype='col',
        # k-NN相关参数
        use_knn_attention=True,
        k_neighbors=5,
        knn_start_layer=1,
        knn_temperature=1.0,
        similarity_metric='cosine'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # 保存k-NN相关参数
        self.use_knn_attention = use_knn_attention
        self.k_neighbors = k_neighbors
        self.knn_start_layer = knn_start_layer
        
        # categories相关计算
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # 创建category embeddings表
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # 用于自动将unique category ids偏移到categories embedding表中的正确位置
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continuous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # 增强的transformer
        if attentiontype == 'col':
            self.transformer = EnhancedTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                k_neighbors=k_neighbors,
                use_knn_attention=use_knn_attention,
                knn_start_layer=knn_start_layer
            )
        elif attentiontype in ['row', 'colrow']:
            # 对于row/colrow attention，暂时使用原有实现
            # 可以后续扩展支持k-NN
            from .model import RowColTransformer
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )
            print("Warning: k-NN attention not yet supported for row/colrow attention types")

        # MLP layers
        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        # Masking相关
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0) 
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)

    def forward(self, x_categ, x_cont, x_categ_enc, x_cont_enc, return_attention_info=False):
        """
        增强的前向传播，支持返回注意力分析信息
        
        Args:
            x_categ: 分类特征
            x_cont: 连续特征
            x_categ_enc: 编码后的分类特征
            x_cont_enc: 编码后的连续特征
            return_attention_info: 是否返回注意力信息
            
        Returns:
            如果return_attention_info=False: 预测结果
            如果return_attention_info=True: (预测结果, 注意力信息)
        """
        device = x_categ.device
        
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim=-1)
            else:
                x = x_cont.clone()
            attention_info = None
        else:
            if self.cont_embeddings == 'MLP':
                if return_attention_info and hasattr(self.transformer, 'forward') and 'return_attention_info' in self.transformer.forward.__code__.co_varnames:
                    x, attention_info = self.transformer(x_categ_enc, x_cont_enc.to(device), return_attention_info=True)
                else:
                    x = self.transformer(x_categ_enc, x_cont_enc.to(device))
                    attention_info = None
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else:
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim=-1)
                attention_info = None
                    
        flat_x = x.flatten(1)
        output = self.mlp(flat_x)
        
        if return_attention_info:
            return output, attention_info
        else:
            return output

    def analyze_knn_attention(self, x_categ, x_cont, x_categ_enc, x_cont_enc, layer_idx=None):
        """
        分析k-NN注意力机制的效果
        
        Args:
            x_categ, x_cont, x_categ_enc, x_cont_enc: 输入数据
            layer_idx: 要分析的层索引，None表示分析所有k-NN层
            
        Returns:
            注意力分析结果
        """
        if not self.use_knn_attention:
            return {"message": "k-NN attention is not enabled"}
        
        device = x_categ.device
        
        # 获取transformer输入
        if self.cont_embeddings == 'MLP':
            x = torch.cat((x_categ_enc, x_cont_enc.to(device)), dim=1)
        else:
            x = x_categ_enc
        
        analysis_results = []
        
        # 逐层分析
        current_x = x
        for i, layer in enumerate(self.transformer.layers):
            if isinstance(layer, EnhancedSAINTBlock) and layer.use_knn_attention:
                if layer_idx is None or i == layer_idx:
                    # 获取k-NN注意力信息
                    knn_module = layer.knn_attention.fn.fn  # PreNorm -> Residual -> KNNCrossSampleAttention
                    
                    # 提取样本表示
                    sample_reprs = knn_module.extract_sample_representations(current_x)
                    
                    # 计算k-NN图
                    knn_indices, knn_similarities = knn_module.compute_knn_graph(sample_reprs)
                    
                    analysis_results.append({
                        'layer_idx': i,
                        'sample_representations_shape': sample_reprs.shape,
                        'knn_indices': knn_indices.cpu(),
                        'knn_similarities': knn_similarities.cpu(),
                        'avg_similarity': knn_similarities.mean().item(),
                        'similarity_std': knn_similarities.std().item()
                    })
            
            # 前向传播到下一层
            if isinstance(layer, EnhancedSAINTBlock):
                current_x = layer(current_x)
            else:
                attn, ff = layer
                current_x = attn(current_x)
                current_x = ff(current_x)
        
        return analysis_results

    def get_model_info(self):
        """
        获取模型架构信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Enhanced SAINT with k-NN Cross-Sample Attention',
            'use_knn_attention': self.use_knn_attention,
            'k_neighbors': self.k_neighbors,
            'knn_start_layer': self.knn_start_layer,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_categories': self.num_categories,
            'num_continuous': self.num_continuous,
            'embedding_dim': self.dim,
            'attention_type': self.attentiontype
        }


# 辅助函数
def create_enhanced_saint_model(
    categories,
    num_continuous,
    dim=32,
    depth=6,
    heads=8,
    k_neighbors=5,
    use_knn_attention=True,
    **kwargs
):
    """
    便捷函数：创建增强版SAINT模型
    
    Args:
        categories: 分类特征的cardinality列表
        num_continuous: 连续特征数量
        dim: 嵌入维度
        depth: 网络深度
        heads: 注意力头数
        k_neighbors: k-NN中的邻居数量
        use_knn_attention: 是否使用k-NN注意力
        **kwargs: 其他参数
    
    Returns:
        EnhancedTabAttention模型实例
    """
    return EnhancedTabAttention(
        categories=categories,
        num_continuous=num_continuous,
        dim=dim,
        depth=depth,
        heads=heads,
        k_neighbors=k_neighbors,
        use_knn_attention=use_knn_attention,
        **kwargs
    )


def compare_models(original_model, enhanced_model, x_categ, x_cont, x_categ_enc, x_cont_enc):
    """
    比较原始SAINT模型和增强版模型的输出
    
    Args:
        original_model: 原始SAINT模型
        enhanced_model: 增强版SAINT模型
        x_categ, x_cont, x_categ_enc, x_cont_enc: 输入数据
    
    Returns:
        比较结果字典
    """
    with torch.no_grad():
        # 原始模型输出
        original_output = original_model(x_categ, x_cont, x_categ_enc, x_cont_enc)
        
        # 增强模型输出
        enhanced_output, attention_info = enhanced_model(
            x_categ, x_cont, x_categ_enc, x_cont_enc, return_attention_info=True
        )
        
        # 计算差异
        output_diff = torch.abs(original_output - enhanced_output).mean().item()
        
        return {
            'original_output_shape': original_output.shape,
            'enhanced_output_shape': enhanced_output.shape,
            'output_difference': output_diff,
            'attention_info': attention_info,
            'original_output_stats': {
                'mean': original_output.mean().item(),
                'std': original_output.std().item(),
                'min': original_output.min().item(),
                'max': original_output.max().item()
            },
            'enhanced_output_stats': {
                'mean': enhanced_output.mean().item(),
                'std': enhanced_output.std().item(),
                'min': enhanced_output.min().item(),
                'max': enhanced_output.max().item()
            }
        } 