"""
This is a modified version Graphormer from Graphormer project by Mircosoft
https://github.com/microsoft/Graphormer/blob/main/graphormer/models/graphormer_3d.py
Copyright (c) 2023 Mesbah Lab. All Rights Reserved.
Contributor(s): Ketong Shao

Copyright (c) Microsoft Corporation.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graphormer_utils import (
    GaussianLayer,
    Graphormer3DEncoderLayer,
    NodeTaskHead,
    NonLinear,
)


class Graphormer3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.atom_types = 64
        self.edge_types = 64 * 64
        self.atom_encoder = nn.Embedding(
            self.atom_types, self.args.embed_dim, padding_idx=510
        )
        self.tag_encoder = nn.Embedding(3, self.args.embed_dim)
        self.input_dropout = self.args.input_dropout
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    num_attention_heads=self.args.attention_heads,
                    dropout=self.args.dropout,
                    attention_dropout=self.args.attention_dropout,
                    activation_dropout=self.args.activation_dropout,
                )
                for _ in range(self.args.layers)
            ]
        )

        self.final_ln: Callable[[Tensor], Tensor] = nn.LayerNorm(
            self.args.embed_dim
        )

        self.engergy_proj: Callable[[Tensor], Tensor] = NonLinear(
            self.args.embed_dim, 1
        )
        self.energe_agg_factor: Callable[[Tensor], Tensor] = nn.Embedding(3, 1)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        K = self.args.num_kernel

        self.gbf: Callable[[Tensor, Tensor], Tensor] = GaussianLayer(
            K, self.edge_types
        )
        self.bias_proj: Callable[[Tensor], Tensor] = NonLinear(
            K, self.args.attention_heads
        )
        self.edge_proj: Callable[[Tensor], Tensor] = nn.Linear(
            K, self.args.embed_dim
        )
        self.node_proc: Callable[
            [Tensor, Tensor, Tensor], Tensor
        ] = NodeTaskHead(self.args.embed_dim, self.args.attention_heads)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        return super().set_num_updates(num_updates)

    def forward(
        self, atoms: Tensor, tags: Tensor, pos: Tensor, real_mask: Tensor
    ):
        padding_mask = atoms.eq(0)

        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: Tensor = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_type = atoms.view(
            n_graph, n_node, 1
        ) * self.atom_types + atoms.view(n_graph, 1, n_node)

        gbf_feature = self.gbf(dist, edge_type)
        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(edge_features.sum(dim=-2))
        )

        # ===== MAIN MODEL =====
        output = F.dropout(
            graph_node_feature, p=self.input_dropout, training=self.training
        )
        output = output.transpose(0, 1).contiguous()

        graph_attn_bias = (
            self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        )
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        for _ in range(self.args.blocks):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)

        output = self.final_ln(output)
        output = output.transpose(0, 1)

        eng_output = F.dropout(output, p=0.1, training=self.training)
        eng_output = (
            self.engergy_proj(eng_output) * self.energe_agg_factor(tags)
        ).flatten(-2)
        output_mask = (
            tags > 0
        ) & real_mask  # no need to consider padding, since padding has tag 0, real_mask False

        eng_output *= output_mask
        eng_output = eng_output.sum(dim=-1)

        node_output = self.node_proc(output, graph_attn_bias, delta_pos)

        node_target_mask = output_mask.unsqueeze(-1)
        return eng_output, node_output, node_target_mask
