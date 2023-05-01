"""
This is a modified version of Gemnet-TAAG from the OCP project (transfer learning) for plasma-catalysis
This model uses a Graphormer to compensate for charge and reuse the embeddings from GemNet
Copyright (c) 2023 Mesbah Lab. All Rights Reserved.
Contributor(s): Ketong Shao

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnFunctional
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.scaling.compat import load_scales_compat

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.graphormer_utils import (
    GaussianLayer,
    Graphormer3DEncoderLayer,
    NodeTaskHead,
    NonLinear,
    SelfMultiheadAttention,
)
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer
from .utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)


@registry.register_model("gemnet_graphormer")
class GemNetGraphormer(BaseModel):
    """
    GemNet-Graphormer, triplets-only variant of GemNet, with a Graphormer conpensating for charge

    Parameters
    ----------
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        regress_forces: bool
            Whether to predict forces. Default: True
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
        freeze: bool
            Whether we should be doing the transfer learning experiments.
        after_freeze_numblocks: int
            Number of randomly initialized interaction blocks added towards the end for TL
        attn_type: str
            either have the "base" taag approach or the "multi"
        num_heads: int
            Number of heads of multi-head-attention for the "multi" attn_type
        add_positional_embedding: bool
            Whether to add positional embedding
    """

    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        regress_forces: bool = True,
        direct_forces: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        extensive: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        num_elements: int = 83,
        scale_file: Optional[str] = None,
        freeze: bool = True,
        after_freeze_numblocks: int = 1,
        attn_type: str = "base",
        num_heads: int = 4,
        add_positional_embedding: bool = True,
    ):
        super().__init__()
        self.num_targets = num_targets
        # assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.cutoff = cutoff
        assert self.cutoff <= 6 or otf_graph

        self.max_neighbors = max_neighbors
        assert self.max_neighbors == 50 or otf_graph

        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc

        self.attn_type = attn_type
        self.add_positional_embedding = add_positional_embedding

        # GemNet variants
        self.direct_forces = direct_forces
        self.freeze = freeze
        self.after_freeze_numblocks = after_freeze_numblocks

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        out_blocks = []
        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly  # GemNet-(d)T
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    name=f"IntBlock_{i+1}",
                )
            )

        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=direct_forces,
                    attn_type=attn_type,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3.linear.weight, self.num_blocks),
            (self.mlp_cbf3.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.num_blocks + 1),
        ]

        if self.attn_type == "base":
            emb_size_taag = 1
        else:
            emb_size_taag = emb_size_atom

        self.lin_query_MHA = nn.Linear(emb_size_taag, emb_size_taag)
        self.lin_key_MHA = nn.Linear(emb_size_taag, emb_size_taag)
        self.lin_value_MHA = nn.Linear(emb_size_taag, emb_size_taag)

        self.softmax = nn.Softmax(dim=1)
        self.MHA = nn.MultiheadAttention(
            embed_dim=emb_size_taag,
            num_heads=num_heads,
            bias=True,
            dropout=0.0,
        )

        if self.add_positional_embedding:
            self.MHA_positional_embedding = PositionalEncoding(
                emb_size_atom, dropout=0.0, max_len=len(self.out_blocks)
            )
        # defined graphormer after this, so the graphormer will not be frozen
        if self.freeze:
            if self.attn_type == "multi":
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        if "MHA" not in name and "block" not in name:
                            param.requires_grad = False
            else:
                pass

            self.after_freeze_IB = torch.nn.ModuleList(
                [
                    interaction_block(
                        emb_size_atom=emb_size_atom,
                        emb_size_edge=emb_size_edge,
                        emb_size_trip=emb_size_trip,
                        emb_size_rbf=emb_size_rbf,
                        emb_size_cbf=emb_size_cbf,
                        emb_size_bil_trip=emb_size_bil_trip,
                        num_before_skip=num_before_skip,
                        num_after_skip=num_after_skip,
                        num_concat=num_concat,
                        num_atom=num_atom,
                        activation=activation,
                        # scale_file=scale_file,
                        name=f"AfterFreezeIntBlock_{num_blocks+i+1}",
                    )
                    for i in range(self.after_freeze_numblocks)
                ]
            )
            self.after_freeze_OB = torch.nn.ModuleList(
                [
                    OutputBlock(
                        emb_size_atom=emb_size_atom,
                        emb_size_edge=emb_size_edge,
                        emb_size_rbf=emb_size_rbf,
                        nHidden=num_atom,
                        num_targets=num_targets,
                        activation=activation,
                        output_init=output_init,
                        direct_forces=direct_forces,
                        # scale_file=scale_file,
                        attn_type="base",
                        name=f"AfterFreezeOutBlock_{i}",
                    )
                    for i in range(self.after_freeze_numblocks)
                ]
            )

            self.out_energy = Dense(emb_size_atom, num_targets)
        # this is different from the one used on TAAG, it is a universal scaling file loading function
        load_scales_compat(self, scale_file)
        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Graphormer blocks ----------------------------------- ###
        # special token for padding used in Graphormer, the index is 510, which should be modified
        self.padding_emb = nn.Embedding(1, emb_size_atom)
        # special token for proton used in Graphormer, the index is 0
        self.proton_emb = nn.Embedding(1, emb_size_atom)
        self.tag_encoder = nn.Embedding(3, emb_size_atom)
        # consider for padding and proton so +2
        self.gbf = GaussianLayer(
            num_radial, (num_elements + 2) * (num_elements + 2)
        )
        # self.input_dropout = self.args.input_dropout
        # TODO: add model attribute to control the graphormer encoder layers
        self.layers = nn.ModuleList(
            [
                Graphormer3DEncoderLayer(
                    emb_size_atom,
                    ffn_embedding_dim=emb_size_atom * 2,
                    num_attention_heads=8,
                    dropout=0.1,
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                )
                for _ in range(4)
            ]
        )

        self.edge_proj = nn.Linear(num_radial, emb_size_atom)
        # TODO: add model attribute to control the graphormer heads
        self.bias_proj = NonLinear(num_radial, 8)
        self.node_proc = NodeTaskHead(emb_size_atom, 8)
        self.final_ln = nn.LayerNorm(emb_size_atom)
        self.energe_agg_factor = nn.Embedding(3, 1)
        nn.init.normal_(self.energe_agg_factor.weight, 0, 0.01)

        self.engergy_proj = NonLinear(emb_size_atom, 1)

    def update_proton_embedding(self) -> None:
        """
        Update the proton embedding after loading the pretrained model
        The proton embedding is trainable and starts from the H embedding
        :return: None
        """
        pass

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(self, data):
        num_atoms = data.atomic_numbers.size(0)

        (
            edge_index,
            D_st,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        V_st = -distance_vec / D_st[:, None]

        # Mask interaction edges if required
        if self.otf_graph or np.isclose(self.cutoff, 6):
            select_cutoff = None
        else:
            select_cutoff = self.cutoff
        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=select_cutoff,
        )

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(
            edge_index, cell_offsets, neighbors, D_st, V_st
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        E_all, F_all = [], []
        E_all.append(E_t)
        F_all.append(F_st)
        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            E_all.append(E)
            F_all.append(F)

        # Implementing attention across pretrained blocks
        E_all = torch.stack(E_all, dim=0)

        if self.add_positional_embedding:
            E_all = self.MHA_positional_embedding(E_all)

        if self.attn_type == "base":

            alpha = torch.bmm(E_all, torch.transpose(E_all, 1, 2))
            alpha = alpha / math.sqrt(E_all.shape[-1])
            alpha = self.softmax(alpha)

            E_t = torch.bmm(alpha, E_all)
            E_t = torch.sum(E_t, dim=0)

        elif self.attn_type == "multi":

            q = self.lin_query_MHA(E_all)
            k = self.lin_key_MHA(E_all)
            v = self.lin_value_MHA(E_all)

            E_t, w = self.MHA(q, k, v)
            E_t = torch.sum(E_t, dim=0)

        if self.attn_type != "base":
            E_t = self.out_energy(E_t)

        if self.freeze:
            for i in range(self.after_freeze_numblocks):
                h, m = self.after_freeze_IB[i](
                    h=h,
                    m=m,
                    rbf3=rbf3,
                    cbf3=cbf3,
                    id3_ragged_idx=id3_ragged_idx,
                    id_swap=id_swap,
                    id3_ba=id3_ba,
                    id3_ca=id3_ca,
                    rbf_h=rbf_h,
                    idx_s=idx_s,
                    idx_t=idx_t,
                )
                E, F = self.after_freeze_OB[i](h, m, rbf_out, idx_t)
                F_st += F
                E_t += E

        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                F_st_vec = F_st[:, :, None] * V_st[:, None, :]
                # (nEdges, num_targets, 3)
                F_t = scatter(
                    F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=data.atomic_numbers.size(0),
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
                F_t = F_t.squeeze(1)  # (nAtoms, 3)
            else:
                if self.num_targets > 1:
                    forces = []
                    for i in range(self.num_targets):
                        # maybe this can be solved differently
                        forces += [
                            -torch.autograd.grad(
                                E_t[:, i].sum(), pos, create_graph=True
                            )[0]
                        ]
                    F_t = torch.stack(forces, dim=1)
                    # (nAtoms, num_targets, 3)
                else:
                    F_t = -torch.autograd.grad(
                        E_t.sum(), pos, create_graph=True
                    )[0]
                    # (nAtoms, 3)

            # return E_t, F_t  # (nMolecules, num_targets), (nAtoms, 3)
        else:
            pass
            # return E_t

        ### ------------------------------------------------------------------------------------- ###

        ### ------------------------------- Graphormer forward ---------------------------------- ###
        # This block will reuse the embedding, rbf, etc. from the GemNet block
        new_atom_embedding = torch.cat(
            [
                self.proton_emb.weight,
                self.atom_emb.embeddings.weight,
                self.padding_emb.weight,
            ]
        )
        # natoms_gh = data.natoms_gh
        atoms_gh = data.atoms_gh.long()
        # pos_gh = data.pos_gh
        # TODO: add a new tag type for proton and padding, which will be used in self.tag_encoder and self.energe_agg_factor
        tags_gh = data.tags_gh.long()
        tags_gh_padding_to_0 = tags_gh.clone()
        tags_gh_padding_to_0[tags_gh_padding_to_0 == 510] = 0
        # fixed_gh = data.fixed_gh
        # force_gh = data.force_gh  # (batch, natoms+padding, 3)
        delta_pos_gh_9cell = (
            data.delta_pos_gh_9cell
        )  # (batch, natoms+padding, 9*(natoms+padding), 3)
        mask_gh = data.mask_gh.int()  # (batch, natoms+padding)
        delta_pos_gh_mask = (
            data.delta_pos_gh_mask.int()
        )  # (batch, natoms+padding, 9*(natoms+padding))
        delta_pos_gh_mask = delta_pos_gh_mask.view(
            delta_pos_gh_mask.shape[0],
            delta_pos_gh_mask.shape[1],
            -1,
            delta_pos_gh_mask.shape[1],
        )
        # (batch, natoms+padding, 9, natoms+padding)
        delta_pos_gh_mask = delta_pos_gh_mask.permute(0, 2, 1, 3)
        # (batch, 9, natoms+padding, natoms+padding)

        delta_pos_gh_padding_mask = (
            data.delta_pos_gh_padding_mask.int()
        )  # (batch, natoms+padding, 9*(natoms+padding))
        delta_pos_gh_padding_mask = delta_pos_gh_padding_mask.view(
            delta_pos_gh_padding_mask.shape[0],
            delta_pos_gh_padding_mask.shape[1],
            -1,
            delta_pos_gh_padding_mask.shape[1],
        )
        # (batch, natoms+padding, 9, natoms+padding)
        delta_pos_gh_padding_mask = delta_pos_gh_padding_mask.permute(
            0, 2, 1, 3
        )
        # (batch, 9, natoms+padding, natoms+padding)

        # modify the padding index as num_elements + 1
        num_of_atom_gh = self.atom_emb.embeddings.weight.shape[0] + 2
        atoms_gh[atoms_gh == 510] = num_of_atom_gh - 1
        atom_emb_gh = nnFunctional.embedding(atoms_gh, new_atom_embedding)
        # (batch, natoms+padding, emb_dim)

        # build the unique edge index
        edge_type = atoms_gh.unsqueeze(
            -1
        ) * num_of_atom_gh + atoms_gh.unsqueeze(1)
        # (batch, natoms+padding, natoms+padding)

        # build the edge features using the distance matrix for Graphormer
        dist = delta_pos_gh_9cell.norm(dim=-1)
        # (batch, natoms+padding, 9*(natoms+padding))
        dist = dist.view(dist.shape[0], dist.shape[1], -1, dist.shape[1])
        # (batch, natoms+padding, 9, natoms+padding)
        dist = dist.permute(0, 2, 1, 3)
        # (batch, 9, natoms+padding, natoms+padding)
        gbf_feature = self.gbf(dist[:, 0, :, :], edge_type)
        graph_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2)
        edge_features = gbf_feature.masked_fill(
            delta_pos_gh_mask[:, 0, :, :].unsqueeze(-1), 0.0
        )
        for i in range(1, dist.shape[1]):
            gbf_feature = self.gbf(dist[:, i, :, :], edge_type)
            graph_attn_bias += self.bias_proj(gbf_feature).permute(0, 3, 1, 2)
            edge_features += gbf_feature.masked_fill(
                delta_pos_gh_mask[:, i, :, :].unsqueeze(-1), 0.0
            )
        graph_attn_bias = graph_attn_bias / dist.shape[1]
        # (batch, heads, natoms+padding, natoms+padding)
        edge_features = edge_features / dist.shape[1]
        # (batch, natoms+padding, natoms+padding, num_radial)

        ######################################################
        # build the edge features using the Gemnet trained rbf
        rbf_gh = self.radial_basis(dist.reshape(-1))
        rbf_gh = rbf_gh.view(
            dist.shape[0], -1, dist.shape[2], dist.shape[2], rbf_gh.shape[-1]
        )
        # (batch, 9, natoms+padding, natoms+padding, num_radial)
        rbf_gh = rbf_gh.masked_fill(delta_pos_gh_mask.unsqueeze(-1), 0.0)

        # complex transformation here for atom_emb_gh, the edge length is (natoms+padding)*(natoms+padding)*9
        # (batch, natoms+padding, emb_dim) ->
        # (batch, 9, (natoms+padding)*(natoms+padding), emb_dim) ->
        # (batch, 9, natoms+padding, natoms+padding, emb_dim)
        edge_emb_gh = torch.cat(
            (
                atom_emb_gh.repeat(1, atom_emb_gh.shape[1], 1)
                .unsqueeze(1)
                .repeat(1, dist.shape[1], 1, 1)
                .view(
                    rbf_gh.shape[0],
                    rbf_gh.shape[1],
                    rbf_gh.shape[2],
                    rbf_gh.shape[3],
                    -1,
                ),
                atom_emb_gh.repeat_interleave(atom_emb_gh.shape[1], dim=1)
                .unsqueeze(1)
                .repeat(1, dist.shape[1], 1, 1)
                .view(
                    rbf_gh.shape[0],
                    rbf_gh.shape[1],
                    rbf_gh.shape[2],
                    rbf_gh.shape[3],
                    -1,
                ),
                rbf_gh,
            ),
            dim=-1,
        )
        # (batch, 9, natoms+padding, natoms+padding, emb_dim+emb_dim+num_radial)
        edge_emb_gh = self.edge_emb.dense(edge_emb_gh)
        # (batch, 9, natoms+padding, natoms+padding, emb_dim)
        # this is only the first level of Gemnet edge embedding
        edge_emb_gh = torch.mean(edge_emb_gh, dim=1)
        # (batch, natoms+padding, natoms+padding, emb_dim)

        # Probably TODO: add other embedding layers for atom embedding
        # Current sitation: (one level) Atom embedding from Gemnet, (one level) edge embedding from Gemnet, edge embedding from Graphormer

        graph_node_feature = (
            self.tag_encoder(tags_gh_padding_to_0)
            + atom_emb_gh
            + self.edge_proj(edge_features.sum(dim=-2))
            + edge_emb_gh.sum(dim=-2)
        )
        # (batch, natoms+padding, emb_dim)

        # ===== MAIN MODEL =====
        # TODO: add a new model argument for dropout
        self.input_dropout = 0.1
        output = nnFunctional.dropout(graph_node_feature, p=self.input_dropout)
        output = output.transpose(0, 1)
        # (natoms+padding, batch, emb_dim)

        graph_attn_bias.masked_fill_(
            delta_pos_gh_padding_mask[:, 0, :, :].unsqueeze(1), float("-inf")
        )
        # (batch, heads, natoms+padding, natoms+padding)

        graph_attn_bias = graph_attn_bias.reshape(
            -1, graph_attn_bias.shape[2], graph_attn_bias.shape[3]
        )
        # (batch*heads, natoms+padding, natoms+padding)
        # TODO: add a new model argument for number of blocks
        for _ in range(4):
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias)
        # (natoms+padding, batch, emb_dim)

        output = self.final_ln(output)
        output = output.transpose(0, 1)
        # (batch, natoms+padding, emb_dim)

        eng_output = nnFunctional.dropout(output, p=0.1)
        eng_output = (
            self.engergy_proj(eng_output)
            * self.energe_agg_factor(tags_gh_padding_to_0)
        ).flatten(-2)
        # why flatten at -2???
        # (batch, natoms+padding, emb_dim) -> (batch, natoms+padding, 1) -> (batch, natoms+padding)
        # TODO: add model toggle for whether to predict surface atom types
        # output_mask = (
        #                      tags_gh > 0
        #              ) & mask_gh  # no need to consider padding, since padding has tag 0, real_mask False

        output_mask = mask_gh
        eng_output *= output_mask
        eng_output = eng_output.sum(dim=-1)
        # (batch)

        delta_pos = delta_pos_gh_9cell.view(
            delta_pos_gh_9cell.shape[0],
            delta_pos_gh_9cell.shape[1],
            -1,
            delta_pos_gh_9cell.shape[1],
            3,
        )
        delta_pos = delta_pos.permute(0, 2, 1, 3, 4)
        # (batch, 9, natoms+padding, natoms+padding, 3)
        dist_n_cell = dist.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        delta_pos[dist_n_cell > 0.0] /= dist_n_cell[dist_n_cell > 0.0]
        delta_pos = delta_pos.mean(dim=1)
        # (batch, natoms+padding, natoms+padding, 3)

        node_output = self.node_proc(output, graph_attn_bias, delta_pos)

        node_target_mask = output_mask.unsqueeze(-1)

        return E_t, F_t, eng_output, node_output, node_target_mask

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:, : x.size(1)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x
