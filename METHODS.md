# Methods

Full catalog of all SSL methods in `stable-pretraining`.

Methods come in two forms:

- **Forward functions** (`stable_pretraining/forward.py`) — stateless functions used with `spt.Module`. You compose them with a backbone, loss, and callbacks yourself. Best for custom experiments.
- **Method classes** (`stable_pretraining/methods/`) — full `LightningModule` implementations with the backbone, loss, and optimizer pre-wired. Best for reproducing published results quickly.

---

## Complete Method Table

`TeacherStudent†` in the callbacks column means the method requires `TeacherStudentCallback` when using the forward-function approach (EMA teacher updates). Method classes handle this internally.

| Method | Forward fn | Method class | Loss class(es) | Key callbacks | Paper |
|--------|-----------|--------------|----------------|---------------|-------|
| Supervised | `supervised` | — | any | — | — |
| SimCLR | `simclr` | `SimCLR` | `NTXEntLoss` | — | [Chen et al., 2020](https://arxiv.org/abs/2002.05709) |
| BYOL | `byol` | `BYOL` | `BYOLLoss` | `TeacherStudent†` | [Grill et al., 2020](https://arxiv.org/abs/2006.07733) |
| VICReg | `vicreg` | `VICReg` | `VICRegLoss` | — | [Bardes et al., 2022](https://arxiv.org/abs/2105.04906) |
| FroSSL | `frossl` | `FroSSL` | `FroSSLLoss` | — | [Skean et al., 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12410.pdf) |
| Barlow Twins | `barlow_twins` | `BarlowTwins` | `BarlowTwinsLoss` | — | [Zbontar et al., 2021](https://arxiv.org/abs/2103.03230) |
| SwAV | `swav` | `SwAV` | `SwAVLoss` | `OnlineQueue` | [Caron et al., 2020](https://arxiv.org/abs/2006.09882) |
| NNCLR | `nnclr` | `NNCLR` | `NTXEntLoss` | `OnlineQueue` | [Dwibedi et al., 2021](https://arxiv.org/abs/2104.14548) |
| DINO | `dino` | `DINO` | `DINOv1Loss` | `TeacherStudent†` | [Caron et al., 2021](https://arxiv.org/abs/2104.14294) |
| DINOv2 | `dinov2` | `DINOv2` | `DINOv2Loss`, `iBOTPatchLoss` | `TeacherStudent†` | [Oquab et al., 2024](https://arxiv.org/abs/2304.07193) |
| BEiT | — | `BEiT` | — | — | [Bao et al., 2022](https://arxiv.org/abs/2106.08254) |
| CMAE | — | `CMAE` | — | — | [Huang et al., 2023](https://arxiv.org/abs/2207.13532) |
| CrossMAE | — | `CrossMAE` | — | — | [Fu et al., 2025](https://arxiv.org/abs/2401.14391) |
| Data2Vec | — | `Data2Vec` | — | `TeacherStudent†` | [Baevski et al., 2022](https://arxiv.org/abs/2202.03555) |
| DINOv3 | — | `DINOv3` | `DINOv2Loss` | `TeacherStudent†` | [Siméoni et al., 2025](https://arxiv.org/abs/2309.16588) |
| iBOT | — | `iBOT` | `DINOv1Loss`, `iBOTPatchLoss` | `TeacherStudent†` | [Zhou et al., 2022](https://arxiv.org/abs/2111.07832) |
| iGPT | — | `iGPT` | — | — | [El-Nouby et al., 2024](https://arxiv.org/abs/2401.08541) |
| IJEPA | — | `IJEPA` | — | — | [Assran et al., 2023](https://arxiv.org/abs/2301.08243) |
| LeJEPA | — | `LeJEPA` | — | — | [Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) |
| MAE | — | `MAE` | `MAELoss` | — | [He et al., 2022](https://arxiv.org/abs/2111.06377) |
| MaskFeat | — | `MaskFeat` | — | — | [Wei et al., 2022](https://arxiv.org/abs/2112.09133) |
| MIMRefiner | — | `MIMRefiner` | `DINOv1Loss`, `iBOTPatchLoss` | — | [Lehner et al., 2024](https://arxiv.org/abs/2402.10093) |
| MoCov2 | — | `MoCov2` | `NTXEntLoss` | — | [Chen et al., 2020](https://arxiv.org/abs/2003.04297) |
| MoCov3 | — | `MoCov3` | `NTXEntLoss` | — | [Chen et al., 2021](https://arxiv.org/abs/2104.02057) |
| MSN | — | `MSN` | — | — | [Assran et al., 2022](https://arxiv.org/abs/2204.07141) |
| NEPA | — | `NEPA` | — | — | — |
| PIRL | — | `PIRL` | — | — | [Misra & van der Maaten, 2020](https://arxiv.org/abs/1912.01991) |
| PMSN | — | `PMSN` | — | — | [Assran et al., 2023](https://arxiv.org/abs/2210.07277) |
| SALT | — | `SALT` | — | — | [Li et al., 2025](https://arxiv.org/pdf/2509.24317) |
| SimMIM | — | `SimMIM` | — | — | [Xie et al., 2022](https://arxiv.org/abs/2111.09886) |
| SimSiam | — | `SimSiam` | — | — | [Chen & He, 2021](https://arxiv.org/abs/2011.10566) |
| TiCO | — | `TiCO` | — | — | [Zhu et al., 2022](https://arxiv.org/abs/2206.10698) |
| VICRegL | — | `VICRegL` | `VICRegLoss` | — | [Bardes et al., 2022](https://arxiv.org/abs/2210.01571) |
| VISReg | — | `VISReg` | — | — | [Wu et al., 2026](https://arxiv.org/abs/2606.02572) |
| WMSE | — | `WMSE` | — | — | [Ermolov et al., 2021](https://arxiv.org/abs/2007.06346) |

---

## Using Forward Functions

Forward functions are stateless and work with `spt.Module`. You own the backbone, loss, and callbacks.

```python
from stable_pretraining import Module
from stable_pretraining.forward import simclr
from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.backbone import from_torchvision

module = Module(
    forward=simclr,
    backbone=from_torchvision("resnet50"),
    projector=...,
    simclr_loss=NTXEntLoss(temperature=0.1),
)
```

All forward functions are importable from `stable_pretraining.forward`. They can also be
referenced by dotted path in YAML configs:

```yaml
module:
  _target_: stable_pretraining.Module
  forward: stable_pretraining.forward.simclr
```

---

## Using Method Classes

Method classes are full `LightningModule` subclasses. All are importable from
`stable_pretraining.methods`:

```python
from stable_pretraining.methods import SimCLR, BYOL, DINO, MAE

model = SimCLR(backbone=backbone, projector=projector, temperature=0.1)
```

Each method file (`stable_pretraining/methods/{method}.py`) contains a runnable example
config in its module docstring.

---

## FroSSL

FroSSL accepts two or more aligned views of each image. The backbone provides
features for evaluation callbacks; the projector maps those features to the
space regularized by `FroSSLLoss`. For two views, the paper's reported
invariance weight is `1.4`; for four or eight views it is `2.0`. The loss
class defaults to `1.0`, matching the authors' multiview implementation.

```python
import stable_pretraining as spt

model = spt.FroSSL(
    backbone=backbone,
    projector=projector,
    optim={"optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9}},
    invariance_weight=1.4,
)
```

Training batches come from `MultiViewTransform` as `{"views": [view1, view2, ...]}`
or a dictionary of named views. Each view contains an `image` tensor, and
optional `label` and `sample_idx` tensors. Evaluation uses one `{"image": ...}`
batch. The forward function returns a scalar `loss` and backbone `embedding`
for training, and `embedding` without `loss` for single-view evaluation.
Callbacks such as `OnlineProbe` can read the returned `embedding` and `label`.

The loss follows the authors' released multiview code: feature columns are
L2-normalized across the local batch without mean-centering, and the
trace-normalized Gram matrix is regularized. The paper presents a centered
covariance formulation. Distributed training uses each rank's local batch
statistics. See [`examples/frossl_cifar10.py`](examples/frossl_cifar10.py) for
the complete `Manager` and online-probe setup; `--smoke` runs without downloads
and does not measure representation quality.

For one-dataset validation, the example splits CIFAR-10's 50,000 training
images into 45,000 training and 5,000 validation images with a saved seed and
indices. It evaluates the final online linear probe once on the untouched
10,000-image official test set. This online-probe result is not the paper's
post-training linear-evaluation protocol. The run directory contains
`run_config.json`, `data_split.json`, and `evaluation.json` alongside the
`Manager`-generated `sidecar.json`, `metrics.csv`, and `checkpoints/last.ckpt`.
Keep run outputs and downloaded data outside the Git repository.

```bash
# Local data staging and training (one GPU if available).
python examples/frossl_cifar10.py --download --data-root /path/to/data \
    --cache-dir /path/to/runs --epochs 100 --batch-size 128

# Brown CS Hydra: stage CIFAR-10 from the login host before submission.
export REPO_ROOT=/path/to/stable-pretraining
export DATA_ROOT=/path/to/shared/data RUN_ROOT=/path/to/shared/runs
export PYTHON_BIN=/path/to/venv/bin/python
"$PYTHON_BIN" -c 'from torchvision.datasets import CIFAR10; import os; CIFAR10(root=os.environ["DATA_ROOT"], train=True, download=True); CIFAR10(root=os.environ["DATA_ROOT"], train=False, download=True)'
mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"
sbatch "$REPO_ROOT/examples/frossl_cifar10.slurm"
```

The batch script uses Hydra's `gpus` partition and one GPU. The compute job
does not download data. The `frossl-cifar10-<jobid>.out` and `.err` files are
written to the submission directory and capture the process log. Use
`spt web "$RUN_ROOT/runs"` to inspect run metrics.

---

## Loss Classes

All loss classes are importable from `stable_pretraining.losses`:

| Loss | Methods that use it |
|------|-------------------|
| `NTXEntLoss` | SimCLR, NNCLR, MoCov2, MoCov3 |
| `BYOLLoss` | BYOL |
| `VICRegLoss` | VICReg, VICRegL |
| `FroSSLLoss` | FroSSL |
| `BarlowTwinsLoss` | Barlow Twins |
| `SwAVLoss` | SwAV |
| `DINOv1Loss` | DINO, iBOT, MIMRefiner |
| `DINOv2Loss` | DINOv2, DINOv3 |
| `iBOTPatchLoss` | DINOv2, iBOT, MIMRefiner |
| `MAELoss` | MAE |
| `CLIPLoss` | Multimodal CLIP-style objectives |
| `NegativeCosineSimilarity` | SimSiam, general use |
