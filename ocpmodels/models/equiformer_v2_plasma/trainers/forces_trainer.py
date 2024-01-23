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


@registry.register_trainer("equiformerv2_plasma_forces_newdist_direct")
class EquiformerV2ForcesTrainer(ForcesTrainer):
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
                batch_natoms = torch.cat(
                    [batch.natoms_gh for batch in batch_list]
                )
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
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_forces = self.model(batch_list)
        else:
            out_energy = self.model(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {
            "energy": out_energy,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        return out

    def _compute_loss(self, out, batch_list) -> int:
        loss = []

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(
            energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
        )

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force_gh.to(self.device) for batch in batch_list], dim=0
            )

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
                        out["forces"].isfinite()
                    )
                    if found_nans_or_infs is True:
                        logging.warning("Found nans while computing loss")
                        out["forces"] = torch.nan_to_num(
                            out["forces"], nan=0.0
                        )

                    dists = torch.norm(
                        (out["forces"] - force_target) * signify_factor_mask,
                        p=2,
                        dim=-1,
                    )
                    weighted_dists_sum = (dists * weight).sum()

                    num_samples = out["forces"].shape[0]
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
                        natoms = torch.cat(
                            [
                                batch.natoms_gh.to(self.device)
                                for batch in batch_list
                            ]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        force_loss = force_mult * self.loss_fn["force"](
                            (out["forces"] * signify_factor_mask)[mask],
                            (force_target * signify_factor_mask)[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms_gh.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        loss.append(
                            force_mult
                            * self.loss_fn["force"](
                                (out["forces"] * signify_factor_mask)[mask],
                                (force_target * signify_factor_mask)[mask],
                            )
                        )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](
                            out["forces"] * signify_factor_mask,
                            force_target * signify_factor_mask,
                        )
                    )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms_gh.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "forces": torch.cat(
                [batch.force_gh.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
        }

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
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

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        backward_every = self.config["optim"].get("backward_every", 1)
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        count = 0

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)
                # Forward, loss, backward every N batch (important!!!).
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                count += 1
                # this will not be an accurate estimation of the accumulated loss
                # but should be very close.
                self._backward(
                    loss / backward_every,
                    (count == backward_every)
                    or (
                        i == len(self.train_loader) - 1
                        and epoch_int == self.config["optim"]["max_epochs"] - 1
                    ),
                )
                if count == backward_every:
                    count = 0

                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )
                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _backward(self, loss, if_update) -> None:
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model.module, "shared_parameters"):
            for p, factor in self.model.module.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log(
                    {"grad_norm": grad_norm}, step=self.step, split="train"
                )
        if if_update:
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.ema:
                self.ema.update()
            self.optimizer.zero_grad()
