"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
import torch_geometric
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers import ForcesTrainer

from .lr_scheduler import LRScheduler


def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    name_no_wd = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            name.endswith(".bias")
            or name.endswith(".affine_weight")
            or name.endswith(".affine_bias")
            or name.endswith(".mean_shift")
            or "bias." in name
            or any(name.endswith(skip_name) for skip_name in skip_list)
        ):
            no_decay.append(param)
            name_no_wd.append(name)
        else:
            decay.append(param)
    name_no_wd.sort()
    params = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    return params, name_no_wd


@registry.register_trainer("gemnet_equiformerv2_plasma_forces_newdist")
class GemnetEquiformerV2ForcesTrainer(ForcesTrainer):
    # This trainer does a few things differently from the parent forces trainer:
    # - Different way of setting up model parameters with no weight decay.
    # - Support for cosine LR scheduler.
    # - When using the LR scheduler, it first converts the epochs into number of
    #   steps and then passes it to the scheduler. That way in the config
    #   everything can be specified in terms of epochs.
    def load_model(self):
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        loader = self.train_loader or self.val_loader or self.test_loader
        if (
            loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
        ):
            self.config["model_attributes"]["num_atoms"] = loader.dataset[
                0
            ].x.shape[-1]
        else:
            self.config["model_attributes"]["num_atoms"] = None
        self.config["model_attributes"]["bond_feat_dim"] = bond_feat_dim
        self.config["model_attributes"]["num_targets"] = self.num_targets
        self.model = registry.get_model_class(self.config["model"])(
            **self.config["model_attributes"],
        ).to(self.device)

        # for no weight decay
        self.model_params_no_wd = {}
        if hasattr(self.model, "no_weight_decay"):
            self.model_params_no_wd = self.model.no_weight_decay()

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    def load_optimizer(self):
        if self.config["optim"].get("quantization", False):
            import bitsandbytes as bnb

            optimizer = self.config["optim"].get("optimizer", "AdamW")
            optimizer = getattr(bnb.optim, optimizer)
            optimizer_params = self.config["optim"]["optimizer_params"]
            weight_decay = optimizer_params["weight_decay"]

            parameters, name_no_wd = add_weight_decay(
                self.model, weight_decay, self.model_params_no_wd
            )
            logging.info("Parameters without weight decay:")
            logging.info(name_no_wd)

            self.optimizer = optimizer(
                parameters,
                lr=self.config["optim"]["lr_initial"],
                **optimizer_params,
            )
        else:
            optimizer = self.config["optim"].get("optimizer", "AdamW")
            optimizer = getattr(optim, optimizer)
            optimizer_params = self.config["optim"]["optimizer_params"]
            weight_decay = optimizer_params["weight_decay"]

            parameters, name_no_wd = add_weight_decay(
                self.model, weight_decay, self.model_params_no_wd
            )
            logging.info("Parameters without weight decay:")
            logging.info(name_no_wd)

            self.optimizer = optimizer(
                parameters,
                lr=self.config["optim"]["lr_initial"],
                **optimizer_params,
            )

    def load_extras(self):
        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]["scheduler_params"]["epochs"] = self.config[
            "optim"
        ]["max_epochs"]
        self.config["optim"]["scheduler_params"]["lr"] = self.config["optim"][
            "lr_initial"
        ]

        # convert epochs into number of steps
        if self.train_loader is None:
            logging.warning("Skipping scheduler setup. No training set found.")
            self.scheduler = None
        else:
            n_iter_per_epoch = len(self.train_loader)
            scheduler_params = self.config["optim"]["scheduler_params"]
            for k in scheduler_params.keys():
                if "epochs" in k:
                    if isinstance(scheduler_params[k], (int, float)):
                        scheduler_params[k] = multiply(
                            scheduler_params[k], n_iter_per_epoch
                        )
                    elif isinstance(scheduler_params[k], list):
                        scheduler_params[k] = [
                            x
                            for x in multiply(
                                scheduler_params[k], n_iter_per_epoch
                            )
                        ]
            self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file=None,
        disable_tqdm: bool = False,
    ) -> Dict[str, npt.NDArray[np.float_]]:
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": []}

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)
                # create energy and forces attributes
                out = self.create_predicted_energy_forces(out, batch_list)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )
                out["forces"] = self.normalizers["grad_target"].denorm(
                    out["forces"]
                )
            if per_image:
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                # use the original natoms here for consistency
                batch_natoms = torch.cat(
                    [batch.natoms_gh for batch in batch_list]
                )
                # use the original fixed here for consistency
                batch_fixed = torch.cat(
                    [batch.fixed_gh for batch in batch_list]
                )
                # total energy target requires predictions to be saved in float32
                # default is float16
                if (
                    self.config["task"].get("prediction_dtype", "float16")
                    == "float32"
                    or self.config["task"]["dataset"] == "oc22_lmdb"
                ):
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float32).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float32)
                else:
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float16).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [
                    force.numpy() for force in per_image_forces
                ]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_forces = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(
                            per_image_forces, _per_image_fixed
                        )
                    ]
                    _chunk_idx = np.array(
                        [
                            free_force.shape[0]
                            for free_force in _per_image_free_forces
                        ]
                    )
                    per_image_forces = _per_image_free_forces
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["forces"].extend(per_image_forces)
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                if self.ema:
                    self.ema.restore()
                return predictions

        predictions["forces"] = np.array(predictions["forces"], dtype=object)
        predictions["chunk_idx"] = np.array(
            predictions["chunk_idx"],
        )
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(
            predictions["id"],
        )
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx"]
        )

        if self.ema:
            self.ema.restore()

        return predictions

    def _forward(self, batch_list):
        # forward pass.
        if self.config["optim"].get("gradient_checkpoint", False):
            gemnet_out, equiformer_out = checkpoint(self.model, batch_list)
        else:
            gemnet_out, equiformer_out = self.model(batch_list)
        if self.config["model_attributes"].get("regress_forces", True):
            gemnet_out_energy, gemnet_out_forces = gemnet_out
            equiformer_energy, equiformer_forces = equiformer_out
        else:
            gemnet_out_energy = gemnet_out
            equiformer_energy = equiformer_out

        if gemnet_out_energy.shape[-1] == 1:
            gemnet_out_energy = gemnet_out_energy.view(-1)

        if equiformer_energy.shape[-1] == 1:
            equiformer_energy = equiformer_energy.view(-1)

        out = {
            "gemnet_energy": gemnet_out_energy,
            "equiformer_energy": equiformer_energy,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["gemnet_forces"] = gemnet_out_forces
            out["equiformer_forces"] = equiformer_forces

        return out

    def _compute_loss(self, out, batch_list) -> int:
        loss = []

        # Energy loss, for configurations with proton(s).
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        # natoms after deleting proton(s)
        natoms_modified = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        # natoms before deleting proton(s)
        natoms_original = torch.cat(
            [batch.natoms_gh.to(self.device) for batch in batch_list], dim=0
        )
        energies_need_to_change = (natoms_modified != natoms_original).int()
        energies_need_to_change = energies_need_to_change.reshape(
            out["gemnet_energy"].shape
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(
            energy_mult
            * self.loss_fn["energy"](
                out["gemnet_energy"]
                + out["equiformer_energy"] * energies_need_to_change,
                energy_target,
            )
        )

        # Extra energy loss, for configurations without proton(s).
        energies_need_not_to_change = (
            natoms_modified == natoms_original
        ).int()
        energies_need_not_to_change = energies_need_not_to_change.reshape(
            out["gemnet_energy"].shape
        )
        loss.append(
            energy_mult
            * self.loss_fn["energy"](
                out["equiformer_energy"] * energies_need_not_to_change,
                torch.zeros_like(out["equiformer_energy"]).to(
                    device=self.device, dtype=out["equiformer_energy"].dtype
                ),
            )
        )

        # Build mask for force
        force_need_to_change = []
        force_need_not_to_change = []
        for origin, modify in zip(
            natoms_original.cpu().tolist(), natoms_modified.cpu().tolist()
        ):
            if origin == modify:
                force_need_to_change = force_need_to_change + [0.0] * origin
                force_need_not_to_change = (
                    force_need_not_to_change + [1.0] * origin
                )
            else:
                force_need_to_change = force_need_to_change + [1.0] * origin
                force_need_not_to_change = (
                    force_need_not_to_change + [0.0] * origin
                )

        force_need_to_change = torch.tensor(force_need_to_change).to(
            self.device
        )
        force_need_not_to_change = torch.tensor(force_need_not_to_change).to(
            self.device
        )

        if len(force_need_to_change.shape) == 1:
            force_need_to_change = force_need_to_change.unsqueeze(-1)
            force_need_not_to_change = force_need_not_to_change.unsqueeze(-1)

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            # need special care here
            # under the existence of proton
            # the output shapes of gemnet and equiformer are different

            # use the force_gh here, which is the original shape
            force_target = torch.cat(
                [batch.force_gh.to(self.device) for batch in batch_list], dim=0
            )
            atomic_numbers = torch.cat(
                [batch.atomic_numbers_gh for batch in batch_list], dim=0
            ).tolist()

            assert len(atomic_numbers) == force_target.shape[0]

            gemnet_forces = out["gemnet_forces"]
            equiformer_forces = out["equiformer_forces"]

            check_signify_atomic_numbers = torch.cat(
                [batch.atomic_numbers_gh for batch in batch_list], dim=0
            )

            if self.config["optim"].get("signify_minor_atoms", False):
                # we are excluding the Al2O3 support
                atoms_to_signify = (check_signify_atomic_numbers != 13) & (
                    check_signify_atomic_numbers != 8
                )
                signify_factor = self.config["optim"].get(
                    "signify_factor", 100
                )
            else:
                atoms_to_signify = check_signify_atomic_numbers < 0
                signify_factor = self.config["optim"].get("signify_factor", 1)

            # create the signify_factor_mask for minor atoms!
            signify_factor_mask = torch.ones_like(force_target).to(
                force_target
            )
            signify_mask = atoms_to_signify.unsqueeze(-1).repeat(
                1, force_target.shape[-1]
            )
            signify_factor_mask[signify_mask] = signify_factor

            # we need to manually make up [0., 0., 0.] vector here for gemnet forces
            for i in range(len(atomic_numbers)):
                if int(atomic_numbers[i]) == 0:
                    left_part = gemnet_forces[:i, :]
                    right_part = gemnet_forces[i:, :]
                    append_part = (
                        torch.zeros(3)
                        .view(1, -1)
                        .to(device=self.device, dtype=gemnet_forces.dtype)
                    )
                    gemnet_forces = torch.cat(
                        [left_part, append_part, right_part], dim=0
                    )

            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                # use the original tags here for consistency
                batch_tags = torch.cat(
                    [
                        batch.tags_gh.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                if self.config["optim"].get("loss_force", "l2mae") == "l2mae":
                    # zero out nans, if any
                    found_nans_or_infs = not torch.all(
                        gemnet_forces.isfinite()
                    )
                    if found_nans_or_infs is True:
                        logging.warning(
                            "Found nans while computing loss for gemnet"
                        )
                        gemnet_forces = torch.nan_to_num(
                            gemnet_forces, nan=0.0
                        )

                    found_nans_or_infs = not torch.all(
                        equiformer_forces.isfinite()
                    )
                    if found_nans_or_infs is True:
                        logging.warning(
                            "Found nans while computing loss for equiformer"
                        )
                        equiformer_forces = torch.nan_to_num(
                            equiformer_forces, nan=0.0
                        )

                    # consider configurations with proton(s)
                    dists_change = torch.norm(
                        (
                            gemnet_forces
                            + equiformer_forces * force_need_to_change
                            - force_target
                        )
                        * signify_factor_mask,
                        p=2,
                        dim=-1,
                    )

                    # consider configurations without proton(s)
                    dists_not_change = torch.norm(
                        (equiformer_forces * force_need_not_to_change)
                        * signify_factor_mask,
                        p=2,
                        dim=-1,
                    )
                    weighted_dists_sum = (
                        (dists_change + dists_not_change) * weight
                    ).sum()

                    num_samples = gemnet_forces.shape[0]
                    num_samples = distutils.all_reduce(
                        num_samples, device=self.device
                    )
                    weighted_dists_sum = (
                        weighted_dists_sum
                        * distutils.get_world_size()
                        / num_samples
                    )

                    force_mult = self.config["optim"].get(
                        "force_coefficient", 30
                    )
                    loss.append(force_mult * weighted_dists_sum)
                else:
                    raise NotImplementedError
            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                if self.config["task"].get("train_on_free_atoms", False):
                    # use the original fixed here for consistency
                    fixed = torch.cat(
                        [
                            batch.fixed_gh.to(self.device)
                            for batch in batch_list
                        ]
                    )
                    mask = fixed == 0
                    if (
                        self.config["optim"]
                        .get("loss_force", "mae")
                        .startswith("atomwise")
                    ):
                        force_mult = self.config["optim"].get(
                            "force_coefficient", 1
                        )
                        # use the original natoms here for consistency
                        natoms = torch.cat(
                            [
                                batch.natoms_gh.to(self.device)
                                for batch in batch_list
                            ]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        # consider configurations with proton(s)
                        force_loss = force_mult * self.loss_fn["force"](
                            (
                                (
                                    gemnet_forces
                                    + equiformer_forces * force_need_to_change
                                )
                                * signify_factor_mask
                            )[mask],
                            (force_target * signify_factor_mask)[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms_gh.shape[0],
                        )
                        # consider configurations without proton(s)
                        force_loss += force_mult * self.loss_fn["force"](
                            (
                                equiformer_forces
                                * force_need_not_to_change
                                * signify_factor_mask
                            )[mask],
                            torch.zeros_like(force_target).to(
                                device=self.device, dtype=force_target.dtype
                            )[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms_gh.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        # consider configurations with proton(s)
                        force_loss = force_mult * self.loss_fn["force"](
                            (
                                (
                                    gemnet_forces
                                    + equiformer_forces * force_need_to_change
                                )
                                * signify_factor_mask
                            )[mask],
                            (force_target * signify_factor_mask)[mask],
                        )
                        # consider configurations without proton(s)
                        force_loss += force_mult * self.loss_fn["force"](
                            (
                                equiformer_forces
                                * force_need_not_to_change
                                * signify_factor_mask
                            )[mask],
                            torch.zeros_like(force_target).to(
                                device=self.device, dtype=force_target.dtype
                            )[mask],
                        )
                        loss.append(force_loss)
                else:
                    # consider configurations with proton(s)
                    force_loss = force_mult * self.loss_fn["force"](
                        (
                            (
                                gemnet_forces
                                + equiformer_forces * force_need_to_change
                            )
                            * signify_factor_mask
                        ),
                        force_target * signify_factor_mask,
                    )
                    # consider configurations without proton(s)
                    force_loss += force_mult * self.loss_fn["force"](
                        (
                            equiformer_forces
                            * force_need_not_to_change
                            * signify_factor_mask
                        ),
                        torch.zeros_like(force_target).to(
                            device=self.device, dtype=force_target.dtype
                        ),
                    )
                    loss.append(force_loss)

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        out = self.create_predicted_energy_forces(out, batch_list)
        # use the original natoms here for consistency
        natoms = torch.cat(
            [batch.natoms_gh.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            # use the force_gh here, which is the original shape
            "forces": torch.cat(
                [batch.force_gh.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
        }

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            # use the original fixed here for consistency
            fixed = torch.cat(
                [batch.fixed_gh.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(
                    torch.sum(mask[s_idx : s_idx + natoms]).item()
                )
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics

    def create_predicted_energy_forces(self, out, batch_list):
        target_forces = torch.cat(
            [batch.force_gh.to(self.device) for batch in batch_list], dim=0
        )

        # create energy attribute for out!!!
        # natoms after deleting proton(s)
        natoms_modified = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        # natoms before deleting proton(s)
        natoms_original = torch.cat(
            [batch.natoms_gh.to(self.device) for batch in batch_list], dim=0
        )
        energies_need_to_change = (natoms_modified != natoms_original).int()
        energies_need_to_change = energies_need_to_change.reshape(
            out["gemnet_energy"].shape
        )
        out["energy"] = (
            out["gemnet_energy"]
            + out["equiformer_energy"] * energies_need_to_change
        )

        # create forces attribute for out!!!
        atomic_numbers = torch.cat(
            [batch.atomic_numbers_gh for batch in batch_list], dim=0
        ).tolist()

        assert len(atomic_numbers) == target_forces.shape[0]

        gemnet_forces = out["gemnet_forces"]
        equiformer_forces = out["equiformer_forces"]

        # Build mask for force
        force_need_to_change = []
        for origin, modify in zip(
            natoms_original.cpu().tolist(), natoms_modified.cpu().tolist()
        ):
            if origin == modify:
                force_need_to_change = force_need_to_change + [0.0] * origin
            else:
                force_need_to_change = force_need_to_change + [1.0] * origin

        force_need_to_change = torch.tensor(force_need_to_change).to(
            self.device
        )

        if len(force_need_to_change.shape) == 1:
            force_need_to_change = force_need_to_change.unsqueeze(-1)

        # we need to manually make up [0., 0., 0.] vector here for gemnet forces
        for i in range(len(atomic_numbers)):
            if int(atomic_numbers[i]) == 0:
                left_part = gemnet_forces[:i, :]
                right_part = gemnet_forces[i:, :]
                append_part = (
                    torch.zeros(3)
                    .view(1, -1)
                    .to(device=self.device, dtype=gemnet_forces.dtype)
                )
                gemnet_forces = torch.cat(
                    [left_part, append_part, right_part], dim=0
                )

        out["forces"] = (
            gemnet_forces + equiformer_forces * force_need_to_change
        )

        return out
