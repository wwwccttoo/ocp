"""
This is a model merging gemnet-TAAG and equiformer, for plasma catalysis.
Copyright (c) 2023 Mesbah Lab. All Rights Reserved.
Contributor(s): Ketong Shao

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

from .equiformer_v2_plasma import EquiformerV2_plasma
from .gaussian_rbf import GaussianRadialBasisLayer
from .gemnet_trans import GemNetTrans
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
)
from .so3 import SO3_LinearV2


@registry.register_model("gemnet_equiformer_v2_newdist")
class GemnetEquiformer_V2(BaseModel):
    """
    Shared arguments
    ---------

    use_pbc (bool):         Use periodic boundary conditions
    regress_forces (bool):  Compute forces
    otf_graph (bool):       Compute graph On The Fly (OTF)

    ---------
    GemNet-TAAG part, triplets-only variant of GemNet, compensating for system/method mismatches

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

    ---------
    Equiformer part with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:

        max_neighbors_gh (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads_gh (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        avg_num_nodes (float):      Average number of nodes per graph
        avg_degree (float):         Average degree of nodes in the graph

        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.regress_forces = kwargs.get("regress_forces", False)
        self.direct_forces = kwargs.get("direct_forces", False)
        self.gemnet_part = GemNetTrans(**kwargs)
        kwargs["max_neighbors"] = kwargs["max_neighbors_gh"]
        kwargs["num_heads"] = kwargs["num_heads_gh"]
        self.equiformer_part = EquiformerV2_plasma(**kwargs)
        if kwargs.get("pretrained", None):
            try:
                self.load_state_dict(
                    torch.load(kwargs["pretrained"]), strict=False
                )
                print("Successfully load the pretrained model!!!")
            except Exception:
                print("The pretrained model path has problems, please check.")
        if kwargs.get("freeze_equiformer_embedding", False):
            print("The atom embedding of the equiformer is fixed!")
            self.freeze_equiformer_embedding()
        if kwargs.get("start_proton_from_H", False):
            print("Initialize the proton embedding from H!")
            self.assign_proton_embedding()

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        from_gemnet = self.gemnet_part(data)
        from_equiformer = self.equiformer_part(data)
        return from_gemnet, from_equiformer

    def freeze_equiformer_embedding(self):
        for name, p in self.equiformer_part.named_parameters():
            if p.requires_grad:
                if (
                    "source_embed" in name
                    or "target_embed" in name
                    or "sphere_embed" in name
                ):
                    p.requires_grad = False

    def assign_proton_embedding(self):
        para_dict = {}
        proton_dict = {}
        for name, p in self.equiformer_part.named_parameters():
            if (
                "source_embed" in name
                or "target_embed" in name
                or "sphere_embed" in name
            ):
                para_dict[name] = p
            if "proton" in name:
                proton_dict[name] = p
        for key, val in proton_dict.items():
            parent_key = key.replace("proton_", "")
            with torch.no_grad():
                # proton embedding start with the hydrogen embedding
                val[0, :] = para_dict[parent_key][1, :].detach().clone()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if "equiformer_part" in module_name and (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(
                    module, EquivariantLayerNormArraySphericalHarmonics
                )
                or isinstance(
                    module, EquivariantRMSNormArraySphericalHarmonics
                )
                or isinstance(
                    module, EquivariantRMSNormArraySphericalHarmonicsV2
                )
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(
                        module, SO3_LinearV2
                    ):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
