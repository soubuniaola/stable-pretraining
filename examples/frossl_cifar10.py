"""Train FroSSL on CIFAR-10 with an online linear probe.

This is an integration example, not a reproduction of the paper's full recipe.
It uses a held-out split of the CIFAR-10 training set for probe validation.

Examples::

    python examples/frossl_cifar10.py --smoke
    python examples/frossl_cifar10.py --epochs 100 --num-views 4
"""

import argparse
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, FakeData

import stable_pretraining as spt


def build_data(args: argparse.Namespace) -> spt.data.DataModule:
    """Build aligned training views and a separate single-view probe split.

    Args:
        args: Parsed example settings.

    Returns:
        DataModule with training and validation loaders.
    """
    transforms = spt.data.transforms
    mean = spt.data.dataset_stats.CIFAR10["mean"]
    std = spt.data.dataset_stats.CIFAR10["std"]
    augment = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.ToImage(mean=mean, std=std),
    )
    train_transform = transforms.MultiViewTransform([augment] * args.num_views)
    val_transform = transforms.Compose(
        transforms.RGB(), transforms.ToImage(mean=mean, std=std)
    )

    if args.smoke:
        train_raw = FakeData(size=16, image_size=(3, 32, 32), num_classes=10)
        val_raw = FakeData(
            size=8, image_size=(3, 32, 32), num_classes=10, random_offset=16
        )
    else:
        cifar = CIFAR10(root=args.data_root, train=True, download=True)
        train_raw, val_raw = random_split(
            cifar, [45_000, 5_000], generator=torch.Generator().manual_seed(args.seed)
        )

    train_ds = spt.data.FromTorchDataset(
        train_raw, names=("image", "label"), transform=train_transform
    )
    val_ds = spt.data.FromTorchDataset(
        val_raw, names=("image", "label"), transform=val_transform
    )
    batch_size = 8 if args.smoke else args.batch_size
    return spt.data.DataModule(
        train=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        ),
        val=DataLoader(val_ds, batch_size=batch_size, num_workers=args.workers),
    )


def main() -> None:
    """Parse settings and run FroSSL through the public Manager API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/frossl"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--invariance-weight", type=float)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.num_views < 2:
        parser.error("--num-views must be at least 2")

    weight = args.invariance_weight
    if weight is None:
        weight = 1.4 if args.num_views == 2 else 2.0
    spt.set(cache_dir=str(args.cache_dir.resolve()))
    hidden_dim, projection_dim = (128, 64) if args.smoke else (2048, 1024)
    projector = torch.nn.Sequential(
        torch.nn.Linear(512, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, projection_dim),
    )
    module = spt.FroSSL(
        backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
        projector=projector,
        optim={"optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9}},
        invariance_weight=weight,
    )
    probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(512, 10),
        loss=torch.nn.CrossEntropyLoss(),
        metrics=torchmetrics.classification.MulticlassAccuracy(10, average="micro"),
    )
    trainer = pl.Trainer(
        max_epochs=1 if args.smoke else args.epochs,
        accelerator="cpu" if args.smoke else "auto",
        devices=1,
        logger=False,
        enable_checkpointing=not args.smoke,
        enable_progress_bar=not args.smoke,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[probe],
    )
    manager = spt.Manager(
        trainer=trainer, module=module, data=build_data(args), seed=args.seed
    )
    manager()
    metrics = trainer.callback_metrics
    print(
        {
            key: float(value)
            for key, value in metrics.items()
            if key in {"fit/loss_epoch", "eval/linear_probe_MulticlassAccuracy"}
        }
    )


if __name__ == "__main__":
    main()
