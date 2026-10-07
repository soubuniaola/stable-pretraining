"""Convenience module for FroSSL."""

from typing import Any

import torch

from stable_pretraining import Module
from stable_pretraining.forward import frossl
from stable_pretraining.losses import FroSSLLoss


class FroSSL(Module):
    """Wire FroSSL components into the dictionary-based training interface.

    Args:
        backbone: Encoder producing one embedding per input image.
        projector: Head mapping embeddings to FroSSL projections.
        optim: Optimizer configuration accepted by :class:`Module`.
        invariance_weight: Weight of the view-invariance term.

    Note:
        The projector and optimizer are explicit so experiments can match
        the authors' settings without hidden architecture defaults.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        projector: torch.nn.Module,
        optim: dict[str, Any],
        invariance_weight: float = 1.0,
    ) -> None:
        super().__init__(
            forward=frossl,
            backbone=backbone,
            projector=projector,
            frossl_loss=FroSSLLoss(invariance_weight=invariance_weight),
            optim=optim,
        )
