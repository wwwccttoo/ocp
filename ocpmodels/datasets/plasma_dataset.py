"""
This is a modified version of LMDBdataset from the OCP project for plasma-catalysis
Copyright (c) 2023 Mesbah Lab. All Rights Reserved.
Contributor(s): Ketong Shao

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path
from typing import Sequence, Union

import lmdb
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform


@registry.register_dataset("plasma")
@registry.register_dataset("single_point_plasma")
@registry.register_dataset("trajectory_plasma")
class PlasmaDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(PlasmaDataset, self).__init__()
        self.config = config

        assert not self.config.get(
            "train_on_plasma", False
        ), "For normal uncharged system, use original dataset"

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii")
                for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        # If specified, limit dataset to only a portion of the entire dataset
        # total_shards: defines total chunks to partition dataset
        # shard: defines dataset shard to make visible
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.transform is not None:
            data_object = self.transform(data_object)

        # TODO: add the forces information
        # assign data used for Graphormer before modifying the information
        data_object.natoms_gh = data_object.natoms
        data_object.atoms_gh = data_object.atomic_numbers
        data_object.pos_gh = data_object.pos
        data_object.tags_gh = data_object.tags
        data_object.fixed_gh = data_object.fixed
        data_object.mask_gh = torch.ones_like(data_object.atoms_gh)
        data_object.delta_pos_gh_9cell = extend_dist_calc(
            data_object.pos, data_object.cell
        )
        # mask out too large distances while keep the proton-atom distances
        data_object.delta_pos_ph_mask = (
            (data_object.delta_pos_gh_9cell.norm(dim=-1) < 8)
            | (data_object.atomic_numbers.eq(0).repeat(9))
            | (data_object.atomic_numbers.eq(0).unsqueeze(-1))
        )

        # delete the proton for simple gnn
        data_object.pos = data_object.pos[data_object.atomic_numbers.ne(0)]
        data_object.tags = data_object.tags[data_object.atomic_numbers.ne(0)]
        data_object.fixed = data_object.fixed[data_object.atomic_numbers.ne(0)]
        data_object.natoms = data_object.natoms - (
            1 - data_object.atomic_numbers.ne(0).all().int()
        )
        data_object.atomic_numbers = data_object.atomic_numbers[
            data_object.atomic_numbers.ne(0)
        ]

        # delete the proton from edges
        edge_not_with_proton = data_object.edge_index.ne(0).all(0)
        data_object.edge_index = data_object.edge_index[
            :, edge_not_with_proton
        ]
        data_object.cell_offsets = data_object.cell_offsets[
            edge_not_with_proton, :
        ]
        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


class SinglePointPlasmaDataset(PlasmaDataset):
    def __init__(self, config, transform=None):
        super(SinglePointPlasmaDataset, self).__init__(config, transform)
        warnings.warn(
            "SinglePointPlasmaDataset is deprecated and will be removed in the future."
            "Please use 'PlasmaDataset' instead.",
            stacklevel=3,
        )


class TrajectoryPlasmaDataset(PlasmaDataset):
    def __init__(self, config, transform=None):
        super(TrajectoryPlasmaDataset, self).__init__(config, transform)
        warnings.warn(
            "TrajectoryPlasmaDataset is deprecated and will be removed in the future."
            "Please use 'PlasmaDataset' instead.",
            stacklevel=3,
        )


def Plasmadata_list_collater(data_list, otf_graph=False):
    # exclude the dist, which will only be used in the graphormer
    batch = Batch.from_data_list(
        data_list,
        exclude_keys=[
            "fixed_gh",
            "atoms_gh",
            "pos_gh",
            "tags_gh",
            "mask_gh",
            "delta_pos_gh_9cell",
            "delta_pos_ph_mask",
        ],
    )
    # TODO: add the forces information
    # should also process for Graphormer for padding
    atoms_gh = pad_1d([_.atoms_gh for _ in data_list], fill=510)
    # (batch, natom)
    pos_gh = pad_1d([_.pos_gh for _ in data_list], fill=510)
    # (batch, natom, 3)
    tags_gh = pad_1d([_.tags_gh for _ in data_list], fill=510)
    # (batch, natom)
    fixed_gh = pad_1d([_.fixed_gh for _ in data_list], fill=510)
    # (batch, natom)

    delta_pos_gh_9cell = torch.cat(
        [
            extend_dist_calc(pos_gh[_id, :, :], _.cell).unsqueeze(0)
            for _id, _ in enumerate(data_list)
        ]
    )
    # (batch, natom, n_cell*natom, 3)

    mask_gh = torch.zeros_like(atoms_gh)
    for _ in range(len(data_list)):
        mask_gh[_, : len(data_list[_].atoms_gh)] = torch.ones_like(
            data_list[_].atoms_gh
        )

    delta_pos_gh_mask = []
    for _ in range(delta_pos_gh_9cell.shape[0]):
        delta_pos_gh_mask.append(
            (
                # exclude the padding ghost atom (510), while maintain the proton-atom distances
                (
                    (delta_pos_gh_9cell[_].norm(dim=-1) < 8)
                    & (atoms_gh[_].ne(510).repeat(9))
                    & (atoms_gh[_].ne(510).unsqueeze(-1))
                )
                | (atoms_gh[_].eq(0).repeat(9))
                | (atoms_gh[_].eq(0).unsqueeze(-1))
            ).unsqueeze(0)
        )
    delta_pos_gh_mask = torch.cat(delta_pos_gh_mask)
    # (batch, natom, n_cell*natom)

    batch.atoms_gh = atoms_gh
    batch.pos_gh = pos_gh
    batch.tags_gh = tags_gh
    batch.fixed_gh = fixed_gh
    batch.delta_pos_gh_9cell = delta_pos_gh_9cell
    batch.mask_gh = mask_gh
    batch.delta_pos_gh_mask = delta_pos_gh_mask

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch


def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]),
        fill,
        dtype=samples[0].dtype,
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


def extend_dist_calc(pos: Tensor, cell: Tensor):
    cell_offsets = torch.tensor(
        [
            [0, 0, 0],
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
    ).float()
    n_cells = cell_offsets.size(0)
    offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
    expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(
        -1, 3
    )
    dist_n_cell = pos.unsqueeze(1) - expand_pos.unsqueeze(0)
    return dist_n_cell
