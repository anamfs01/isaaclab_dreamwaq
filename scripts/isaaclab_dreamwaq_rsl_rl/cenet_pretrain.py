# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamWaQ CENet pretraining.

This script trains the Context-Aided Estimator Network (CENet)
using a prepared dataset containing:

    history          : [N, 225]
    next_observation : [N, 45]
    velocity         : [N, 3]

The trained CENet is saved as a checkpoint and can later be loaded
by the DreamWaQ FTNet actor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from isaaclab_dreamwaq_rl.rsl_rl.models.cenet import CENet

from isaaclab_dreamwaq_rl.rsl_rl.cenet_pretrain_cfg import (
    CENetPretrainCfg,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Pretrain DreamWaQ CENet."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the CENet training dataset (.pt file).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="logs/cenet_pretrained.pt",
        help="Path where the trained CENet checkpoint will be saved.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the learning rate.",
    )

    return parser.parse_args()


def load_dataset(
    dataset_path: str,
) -> TensorDataset:
    """
    Load a prepared CENet dataset.

    Expected format:

        {
            "history": Tensor[N, 225],
            "next_observation": Tensor[N, 45],
            "velocity": Tensor[N, 3],
        }
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CENet dataset was not found: {path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
    )

    required_keys = (
        "history",
        "next_observation",
        "velocity",
    )

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Dataset is missing required key: '{key}'"
            )

    history = data["history"].float()
    next_observation = data["next_observation"].float()
    velocity = data["velocity"].float()

    if history.ndim != 2:
        raise ValueError(
            f"history must have shape [N, 225], "
            f"but got {tuple(history.shape)}"
        )

    if next_observation.ndim != 2:
        raise ValueError(
            "next_observation must have shape [N, 45], "
            f"but got {tuple(next_observation.shape)}"
        )

    if velocity.ndim != 2:
        raise ValueError(
            f"velocity must have shape [N, 3], "
            f"but got {tuple(velocity.shape)}"
        )

    if history.shape[1] != 225:
        raise ValueError(
            f"Expected history dimension 225, "
            f"got {history.shape[1]}"
        )

    if next_observation.shape[1] != 45:
        raise ValueError(
            f"Expected next observation dimension 45, "
            f"got {next_observation.shape[1]}"
        )

    if velocity.shape[1] != 3:
        raise ValueError(
            f"Expected velocity dimension 3, "
            f"got {velocity.shape[1]}"
        )

    if not (
        len(history)
        == len(next_observation)
        == len(velocity)
    ):
        raise ValueError(
            "history, next_observation and velocity "
            "must contain the same number of samples."
        )

    return TensorDataset(
        history,
        next_observation,
        velocity,
    )


def train(
    model: CENet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    """Run one complete CENet training epoch."""

    model.train()

    total_loss = 0.0
    total_velocity_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_vae_loss = 0.0

    num_batches = 0

    for history, next_observation, velocity_target in dataloader:

        history = history.to(device)
        next_observation = next_observation.to(device)
        velocity_target = velocity_target.to(device)

        optimizer.zero_grad()

        outputs = model(history)

        losses = model.compute_loss(
            next_observation=next_observation,
            velocity_target=velocity_target,
            outputs=outputs,
            beta=beta,
        )

        loss = losses["total_loss"]

        loss.backward()

        optimizer.step()

        total_loss += loss.detach().item()
        total_velocity_loss += (
            losses["velocity_loss"].detach().item()
        )
        total_reconstruction_loss += (
            losses["reconstruction_loss"].detach().item()
        )
        total_kl_loss += (
            losses["kl_loss"].detach().item()
        )
        total_vae_loss += (
            losses["vae_loss"].detach().item()
        )

        num_batches += 1

    if num_batches == 0:
        raise RuntimeError(
            "The CENet dataloader contains no batches."
        )

    return {
        "total_loss": total_loss / num_batches,
        "velocity_loss": total_velocity_loss / num_batches,
        "reconstruction_loss": (
            total_reconstruction_loss / num_batches
        ),
        "kl_loss": total_kl_loss / num_batches,
        "vae_loss": total_vae_loss / num_batches,
    }


def save_checkpoint(
    model: CENet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: dict[str, float],
    output_path: str,
) -> None:
    """Save a CENet training checkpoint."""

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }

    torch.save(
        checkpoint,
        output,
    )


def main() -> None:
    """Main CENet pretraining entry point."""

    args = parse_args()

    cfg = CENetPretrainCfg()

    epochs = (
    args.epochs
    if args.epochs is not None
    else cfg.num_epochs
)

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else cfg.batch_size
    )

    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else cfg.learning_rate
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("========================================")
    print("DreamWaQ CENet Pretraining")
    print("========================================")
    print(f"Device:         {device}")
    print(f"Dataset:        {args.dataset}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Beta:           {cfg.beta}")
    print("========================================")

    dataset = load_dataset(
        args.dataset
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = CENet(
        input_dim=cfg.history_dim,
        latent_dim=cfg.latent_dim,
        velocity_dim=cfg.velocity_dim,
        observation_dim=cfg.reconstruction_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):

        losses = train(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            beta=cfg.beta,
        )

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss: {losses['total_loss']:.6f} | "
            f"Vel: {losses['velocity_loss']:.6f} | "
            f"Recon: {losses['reconstruction_loss']:.6f} | "
            f"KL: {losses['kl_loss']:.6f}"
        )

        if losses["total_loss"] < best_loss:
            best_loss = losses["total_loss"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                losses=losses,
                output_path=args.output,
            )

            print(
                f"  → Best checkpoint saved: {args.output}"
            )

    print("========================================")
    print("CENet pretraining finished.")
    print(f"Checkpoint: {args.output}")
    print("========================================")


if __name__ == "__main__":
    main()