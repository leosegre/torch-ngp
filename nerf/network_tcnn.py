import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 semantic_feat_dim=15,
                 num_layers_semantic=3,
                 hidden_dim_semantic=64,
                 bound=1,
                 num_classes=92,
                 no_seg=False,
                 only_seg=False,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # Options
        self.no_seg = no_seg
        self.only_seg = only_seg

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.semantic_feat_dim = semantic_feat_dim
        self.num_classes = num_classes

        self.base_resolution = 32
        per_level_scale = np.exp2(np.log2(2048 * bound / self.base_resolution) / (self.base_resolution - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.base_resolution,
                "n_features_per_level": 2,
                "log2_hashmap_size": 20,
                "base_resolution": self.base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=2 * self.base_resolution,
            n_output_dims=1 + self.geo_feat_dim + self.semantic_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        self.semantic_net = tcnn.Network(
            n_input_dims=self.semantic_feat_dim,
            n_output_dims=self.num_classes,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_semantic,
                "n_hidden_layers": num_layers_semantic - 1,
            },
        )


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:self.geo_feat_dim+1]
        semantic_feat = h[..., self.geo_feat_dim+1:self.geo_feat_dim+1+self.semantic_feat_dim].detach()

        # semantics
        semantic = self.semantic_net(semantic_feat)
        # Softmax for semantic segmentation
        # semantic = torch.log_softmax(semantic, dim=1)
        # semantic = torch.nn.functional.log_softmax(semantic, dim=1)
        semantic = torch.softmax(semantic, dim=1)

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        # print(semantic.sum(axis=1))



        return sigma, color, semantic

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def semantic(self, x, mask=None):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if mask is not None:
            semantic = torch.zeros(mask.shape[0], self.num_classes, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return semantic
            x = x[mask]

        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # semantics
        s = self.semantic_net(geo_feat)
        # print(torch.topk(s, 2))

        # Softmax for semantic segmentation
        semantic = torch.log_softmax(s, dim=1)
        # semantic = torch.softmax(semantic, dim=1)
        # if(semantic.max())
        # print(semantic.max())

        if mask is not None:
            semantic[mask] = s.to(semantic.dtype)  # fp16 --> fp32
        else:
            semantic = s

        return semantic

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

        # optimizer utils

    def get_params(self, lr):
        if self.only_seg:
            params = [{'params': self.semantic_net.parameters(), 'lr': lr}]
        elif self.no_seg:
            params = [
                {'params': self.encoder.parameters(), 'lr': lr},
                {'params': self.sigma_net.parameters(), 'lr': lr},
                {'params': self.encoder_dir.parameters(), 'lr': lr},
                {'params': self.color_net.parameters(), 'lr': lr}
            ]
        else:
            params = [
                {'params': self.encoder.parameters(), 'lr': lr},
                {'params': self.sigma_net.parameters(), 'lr': lr},
                {'params': self.encoder_dir.parameters(), 'lr': lr},
                {'params': self.color_net.parameters(), 'lr': lr},
                {'params': self.semantic_net.parameters(), 'lr': lr},
            ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params