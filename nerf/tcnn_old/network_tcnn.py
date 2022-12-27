import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 nerf_model=None,
                 semantics_model=None,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.nerf_model = nerf_model
        self.semantics_model = semantics_model
        self.num_classes = self.semantics_model.num_classes


    def forward(self, x, d):

        sigma, color, geo_feat = self.nerf_model(x, d)
        semantic = self.semantics_model(geo_feat)

        return sigma, color, semantic

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.nerf_model.bound) / (2 * self.nerf_model.bound)  # to [0, 1]
        x = self.nerf_model.encoder(x)
        h = self.nerf_model.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.nerf_model.bound) / (2 * self.nerf_model.bound)  # to [0, 1]

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
        d = self.nerf_model.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.nerf_model.color_net(h)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

        # optimizer utils

    def semantic(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.nerf_model.encoder(x)
        h = self.nerf_model.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # semantics
        semantic = self.semantics_model.semantic_net(geo_feat)
        # print(torch.topk(s, 2))

        # Softmax for semantic segmentation
        # semantic = torch.softmax(s, dim=1)
        # if(semantic.max())
        # print(semantic.max())

        return semantic

    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.nerf_model.encoder.parameters(), 'lr': lr},
            {'params': self.nerf_model.sigma_net.parameters(), 'lr': lr},
            {'params': self.nerf_model.encoder_dir.parameters(), 'lr': lr},
            {'params': self.nerf_model.color_net.parameters(), 'lr': lr},
            {'params': self.semantics_model.semantic_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params


class NeRFNetwork_semantics(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 geo_feat_dim=15,
                 num_layers_semantic=2,
                 hidden_dim_semantic=64,
                 bound=1,
                 num_classes=92,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.geo_feat_dim = geo_feat_dim
        self.num_classes = num_classes

        self.semantic_net = tcnn.Network(
            n_input_dims=self.geo_feat_dim,
            n_output_dims=self.num_classes,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_semantic,
                "n_hidden_layers": num_layers_semantic - 1,
            },
        )

    def forward(self, geo_feat):
        # semantics
        semantic = self.semantic_net(geo_feat)
        # Softmax for semantic segmentation
        # semantic = torch.softmax(s, dim=1)

        return semantic

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.semantic_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params


class NeRFNetwork_nerf(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 num_classes=92,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_classes = num_classes

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
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

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2  # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color, geo_feat


    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params