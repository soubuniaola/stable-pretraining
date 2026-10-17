"""Two-process CPU DDP training for the public FroSSL method."""

import os
import sys
from pathlib import Path

import lightning as pl
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader

import stable_pretraining as spt


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "linux" or not torch.distributed.is_gloo_available(),
        reason="requires Linux and the Gloo distributed backend",
    ),
]


class _RankAudit(pl.Callback):
    """Save each rank's final weights and aggregated FroSSL loss."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        loss = trainer.callback_metrics["fit/loss_epoch"]
        torch.save(
            {
                "backbone_weight": pl_module.backbone.weight.detach().cpu(),
                "projector_weight": pl_module.projector.weight.detach().cpu(),
                "loss": loss.detach().cpu(),
                "global_step": trainer.global_step,
            },
            Path(trainer.default_root_dir) / f"rank-{trainer.global_rank}.pt",
        )


def test_frossl_two_process_ddp_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train both ranks and verify finite loss and synchronized parameter updates."""
    for name in tuple(os.environ):
        if name.startswith("SLURM_"):
            monkeypatch.delenv(name)

    generator = torch.Generator().manual_seed(31)
    dataset = []
    for _ in range(16):
        image = torch.randn(3, generator=generator)
        dataset.append(
            {
                "views": [
                    {"image": image + 0.1 * torch.randn(3, generator=generator)},
                    {"image": image + 0.1 * torch.randn(3, generator=generator)},
                ]
            }
        )
    loader = DataLoader(dataset, batch_size=4, drop_last=True)

    torch.manual_seed(17)
    model = spt.FroSSL(
        backbone=nn.Linear(3, 4),
        projector=nn.Linear(4, 3),
        optim={"optimizer": {"type": "SGD", "lr": 0.1}},
        invariance_weight=1.4,
    )
    initial_backbone = model.backbone.weight.detach().clone()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(
            start_method="fork",
            process_group_backend="gloo",
            find_unused_parameters=False,
        ),
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        callbacks=[_RankAudit()],
    )

    trainer.fit(model, train_dataloaders=loader)

    left = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert left["global_step"] == right["global_step"] == 2
    assert torch.isfinite(left["loss"])
    assert torch.isfinite(right["loss"])
    for key in ("backbone_weight", "projector_weight"):
        torch.testing.assert_close(left[key], right[key])
    assert not torch.equal(left["backbone_weight"], initial_backbone)
