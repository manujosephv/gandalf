# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
import random
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor

from pytorch_tabular.models.common.layers import Embedding1dLayer, GatedFeatureLearningUnit
from pytorch_tabular.models.common.layers.activations import t_softmax
from pytorch_tabular.utils import get_logger

logger = get_logger(__name__)


def t_softmax(input: Tensor, t: Tensor = None, dim: int = -1) -> Tensor:
    if t is None:
        t = torch.tensor(0.5, device=input.device)
    assert (t >= 0.0).all()
    maxes = torch.max(input, dim=dim, keepdim=True).values
    input_minus_maxes = input - maxes

    w = torch.relu(input_minus_maxes + t) + 1e-8
    return torch.softmax(input_minus_maxes + torch.log(w), dim=dim)


class TSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, t: Tensor) -> Tensor:
        return t_softmax(input, t, self.dim)


class RSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.tsoftmax = TSoftmax(dim=dim)

    @classmethod
    def calculate_t(cls, input: Tensor, r: Tensor, dim: int = -1, eps: float = 1e-8):
        # r represents what is the fraction of zero values that we want to have
        assert ((0.0 <= r) & (r <= 1.0)).all()

        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes

        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]

        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        if q.ndim > 1:
            t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
            t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps
        else:
            t = -torch.quantile(x_minus_maxes, q, dim=dim).detach() + eps
        return t

    def forward(self, input: Tensor, r: Tensor):
        t = RSoftmax.calculate_t(input, r, self.dim, self.eps)
        return self.tsoftmax(input, t)


class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = t_softmax,
        feature_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self.feature_sparsity = feature_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()

    def _create_feature_mask(self):
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        return nn.Parameter(
            feature_masks,
            requires_grad=True,
        )

    def _build_network(self):
        self.W_in = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, 2 * self.n_features_in) for _ in range(self.n_stages)]
        )
        self.W_out = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, self.n_features_in) for _ in range(self.n_stages)]
        )

        self.feature_masks = self._create_feature_mask()
        if self.feature_mask_function.__name__ == "t_softmax":
            t = RSoftmax.calculate_t(self.feature_masks, r=torch.tensor([self.feature_sparsity]), dim=-1)
            self.t = nn.Parameter(t, requires_grad=self.learnable_sparsity)
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        t = torch.relu(self.t) if self.feature_mask_function.__name__ == "t_softmax" else None
        for d in range(self.n_stages):
            if self.feature_mask_function.__name__ == "t_softmax":
                feature = self.feature_mask_function(self.feature_masks[d], t[d]) * x
            else:
                feature = self.feature_mask_function(self.feature_masks[d]) * x
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])  # noqa: E203
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h


class GANDALFBackbone(nn.Module):
    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = True,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        assert True, "This is an illustrative model. Please use the model in PyTorch Tabular to run the experiments"
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = self.n_continuous_features + self._embedded_cat_features
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()

    def _build_network(self):
        self.gflus = GatedFeatureLearningUnit(
            n_features_in=self.n_features,
            n_stages=self.gflu_stages,
            feature_mask_function=t_softmax,
            dropout=self.gflu_dropout,
            feature_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
        )
        self.embedding = Embedding1dLayer(
            continuous_dim=self.n_continuous_features,
            categorical_embedding_dims=self.cat_embedding_dims,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
        )
        self.mlps = nn.Sequential(
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.gflus(x)
        return self.mlps(x)
