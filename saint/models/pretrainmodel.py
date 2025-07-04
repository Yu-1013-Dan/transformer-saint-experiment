from saint.models.model import *
from saint.models.layers import CategoricalEmbedder, PiecewiseLinearEmbeddings


class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 0,
        attn_dropout = 0.,
        ff_dropout = 0.,
        numerical_embedding_type = 'mlp',
        categorical_embedding_type = 'saint',
        bins = None,
        scalingfactor = 10,
        attentiontype = 'col',
        final_mlp_style = 'common',
        y_dim = 2
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # embedding types
        self.numerical_embedding_type = numerical_embedding_type
        self.categorical_embedding_type = categorical_embedding_type

        # categories related calculations
        self.num_categories = len(categories)
        if categorical_embedding_type == 'saint':
            self.num_unique_categories = sum(categories)
            # create category embeddings table
            self.num_special_tokens = num_special_tokens
            self.total_tokens = self.num_unique_categories + num_special_tokens

            # for automatically offsetting unique category ids to the correct position in the categories embedding table
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.embeds = nn.Embedding(self.total_tokens, self.dim)
        elif categorical_embedding_type == 'ft':
            self.categorical_embedder = CategoricalEmbedder(categories, dim)
            self.num_special_tokens = 0 # FT-Transformer does not use special tokens like [CLS] in the same way
            self.total_tokens = 0 # Not applicable for FT-Transformer style embeddings
        else:
            raise ValueError("Invalid categorical_embedding_type. Choose 'saint' or 'ft'.")


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.numerical_embedding_type == 'mlp':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
        elif self.numerical_embedding_type == 'ple':
            assert bins is not None, "bins must be provided for 'ple' embedding type"
            self.numerical_embedder = PiecewiseLinearEmbeddings(bins, self.dim, activation=True)
        else:
            raise ValueError("Invalid numerical_embedding_type. Choose 'mlp' or 'ple'.")
        
        nfeats = self.num_categories + num_continuous
        input_size = (dim * nfeats)
        
        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens if categorical_embedding_type == 'saint' else 0,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens if categorical_embedding_type == 'saint' else 0,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        
        if self.final_mlp_style == 'common':
            if categorical_embedding_type == 'saint':
                self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            else: # ft
                self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

        self.final_norm = nn.LayerNorm(dim)

        
    def forward(self, x_categ, x_cont):
        # Generate embeddings
        if self.numerical_embedding_type == 'ple':
            x_cont_embed = self.numerical_embedder(x_cont)
        elif self.numerical_embedding_type == 'mlp':
            x_cont_embed = torch.empty(x_cont.shape[0], self.num_continuous, self.dim, device=x_cont.device)
            for i in range(self.num_continuous):
                x_cont_embed[:,i,:] = self.simple_MLP[i](x_cont[:,i].unsqueeze(1))
        
        if self.categorical_embedding_type == 'ft':
            x_categ_embed = self.categorical_embedder(x_categ)
        elif self.categorical_embedding_type == 'saint':
            x_categ_embed = self.embeds(x_categ + self.categories_offset.type_as(x_categ))

        # Concatenate embeddings
        x = torch.cat((x_categ_embed, x_cont_embed), dim=1)
        
        # Transformer
        x = self.transformer(x)
        x = self.final_norm(x)
        
        # Output MLPs
        cat_outs = self.mlp1(x[:,:self.num_categories,:])
        con_outs = self.mlp2(x[:,self.num_categories:,:])
        
        return cat_outs, con_outs 