import math
import warnings
from typing import Any, Literal, Optional, Union

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# FT-Transformer style categorical embedding
class CategoricalEmbedder(nn.Module):
    def __init__(self, categories, d_embedding):
        super().__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(cardinality, d_embedding) for cardinality in categories
        ])
        self.n_categories = len(categories)
        self.biases = nn.Parameter(torch.zeros(self.n_categories, d_embedding))

    def forward(self, x_categ):
        """
        x_categ: LongTensor of shape (batch_size, n_categoricals)
        """
        all_embeddings = []
        for i, embedding_layer in enumerate(self.embedding_layers):
            all_embeddings.append(embedding_layer(x_categ[:, i]))
        
        # (batch_size, n_categoricals, d_embedding)
        x = torch.stack(all_embeddings, dim=1)
        x += self.biases.unsqueeze(0)
        return x


def _check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(
            f'The input must have at least one dimension, however: {x.ndim=}'
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            'The last dimension of the input was expected to be'
            f' {expected_n_features}, however, {x.shape[-1]=}'
        )


def _check_bins(bins: list[Tensor]) -> None:
    for i, b in enumerate(bins):
        if not isinstance(b, Tensor):
            raise TypeError(f'bins[{i}] must be a torch.Tensor')
        if b.ndim != 1:
            raise ValueError(f'bins[{i}] must be a one-dimensional tensor')
        if (b.diff() <= 0.0).any():
            raise ValueError(f'The values of bins[{i}] must be strictly increasing')


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[dict[str, Any]] = None,
    y: Optional[Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> list[Tensor]:
    _check_input_shape(X, X.shape[-1])
    if n_bins <= 1:
        raise ValueError(f'n_bins must be greater than 1, however: {n_bins=}')
    if tree_kwargs is None:
        tree_kwargs = {}
    if sklearn_tree is None:
        raise ImportError('sklearn is not installed')
    if y is None:
        if regression is not None:
            warnings.warn(
                'If y is None, then the value of the "regression" argument is ignored.'
            )
        regression = False
    elif regression is None:
        raise ValueError('If y is not None, then "regression" must be bool.')

    bins = []
    iterator = range(X.shape[-1])
    if verbose and tqdm is not None:
        iterator = tqdm(iterator, desc='Computing bins')
    for i in iterator:
        x_i = X[:, i]
        y_i = None if y is None else y

        if y_i is not None:
            not_nan_mask = ~torch.isnan(x_i)
            if y_i.ndim == 2:
                # In this case, y is represented by one-hot encoded vectors,
                # so there is no need to select not-nan values.
                pass
            else:
                y_i = y_i[not_nan_mask]
            x_i = x_i[not_nan_mask]

        if regression:
            tree = sklearn_tree.DecisionTreeRegressor(
                max_leaf_nodes=n_bins, **tree_kwargs
            )
        else:
            tree = sklearn_tree.DecisionTreeClassifier(
                max_leaf_nodes=n_bins, **tree_kwargs
            )
        tree.fit(x_i.cpu().numpy()[:, None], y_i.cpu().numpy() if y_i is not None else None)
        thresholds = torch.from_numpy(tree.tree_.threshold)
        b = thresholds[thresholds != sklearn_tree._tree.TREE_UNDEFINED]
        b = torch.unique(b)
        if len(b) == 0:
            b = torch.tensor([-torch.inf, torch.inf])
        else:
            b = torch.cat(
                [torch.tensor([-torch.inf]), b, torch.tensor([torch.inf])]
            )
        bins.append(b)
    return bins


class _PiecewiseLinearEncodingImpl(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        _check_bins(bins)
        super().__init__()

        n_features = len(bins)
        max_n_bins = max(map(len, bins)) - 1
        if max_n_bins == 0:
            raise ValueError('Each feature must have at least two bin edges.')

        bins_tensor = torch.empty(n_features, max_n_bins, 2)
        for i_feature, feature_bins in enumerate(bins):
            n_bins = len(feature_bins) - 1
            if n_bins < max_n_bins:
                bins_tensor[i_feature, n_bins:] = torch.nan
            bins_tensor[i_feature, :n_bins, 0] = feature_bins[:-1]
            bins_tensor[i_feature, :n_bins, 1] = feature_bins[1:]
        self.register_buffer('bins', bins_tensor)

        weight = 1.0 / (self.bins[..., 1] - self.bins[..., 0])
        self.register_buffer('weight', weight)
        bias = -self.bins[..., 0] * self.weight
        self.register_buffer('bias', bias)

        single_bin_mask = torch.tensor(
            [len(b) == 2 for b in bins], dtype=torch.bool
        )
        if single_bin_mask.any():
            self.register_buffer('single_bin_mask', single_bin_mask)
        else:
            self.single_bin_mask = None

        mask = torch.isnan(self.bins[:, :, 0])
        if mask.any():
            self.register_buffer('mask', mask)
        else:
            self.mask = None

    def get_max_n_bins(self) -> int:
        return self.bins.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        _check_input_shape(x, self.bins.shape[0])
        x = torch.addcmul(
            self.bias, self.weight, x.unsqueeze(-1)
        )
        x = torch.clamp(x, 0.0, 1.0)
        if self.mask is not None:
            x = x.masked_fill(self.mask, 0.0)
        if self.single_bin_mask is not None:
            # Clone to avoid inplace operation
            x_clone = x.clone()
            x_clone[self.single_bin_mask] = 0.5
            x = x_clone
        return x


class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.n_features, self.n_bins, self.d_encoding = (
            self.impl.bins.shape[0],
            self.impl.get_max_n_bins(),
            self.impl.get_max_n_bins(),
        )

    def get_output_shape(self) -> torch.Size:
        return torch.Size([self.n_features, self.d_encoding])

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)


class PiecewiseLinearEmbeddings(nn.Module):
    def __init__(
        self,
        bins: list[Tensor],
        d_embedding: int,
        *,
        activation: bool,
        version: Literal[None, 'A', 'B'] = None,
    ) -> None:
        if version is not None:
            warnings.warn(
                'The "version" argument is deprecated and will be removed in the future,'
                ' it has no effect.',
                DeprecationWarning,
            )
        super().__init__()
        self.encoding = _PiecewiseLinearEncodingImpl(bins)
        self.n_features, self.n_bins = self.encoding.bins.shape[:2]
        self.linear = nn.Linear(self.n_bins, d_embedding)
        self.activation = nn.ReLU(inplace=False) if activation else None

    @property
    def bins(self) -> Tensor:
        return self.encoding.bins

    def get_output_shape(self) -> torch.Size:
        return torch.Size([self.n_features, self.linear.out_features])

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoding(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x.clone())
        return x 