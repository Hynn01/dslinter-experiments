#!/usr/bin/env python
# coding: utf-8

# # GI-Seg PyTorch ⚡ Train & Infer
# - Downloaded weight and requirements come from [GI-Seg Downloads](https://www.kaggle.com/clemchris/gi-seg-download)
# - Dataset: [UW-Madison GI Tract Image Segmentation Masks](https://www.kaggle.com/datasets/clemchris/uw-madison-gi-tract-image-segmentation-masks)
# 
# 
# ## Sources --> please upvote them if you find this notebook useful
# - Awsaf's [UWMGI: Unet [Train] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch)
# - Awsaf's [UWMGI: 2.5D stride=2 Data](https://www.kaggle.com/code/awsaf49/uwmgi-2-5d-stride-2-data)
# - Awsaf's [UWMGI: Unet [Infer] [PyTorch]](https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch)
# 
# ## Scores
# - V13: 0.775 (`arch="Unet"`, `encoder_name="efficientnet-b2"`, `batch_size=64"`, `img_size=256`, `max_epochs=3`)
# - V14: 0.748 (`arch="Unet"`, `encoder_name="efficientnet-b0"`, `batch_size=128"`, `img_size=256`, `max_epochs=3`)
# - V15 - V19: Memory errors 
# - V20: 0.778 (`arch="Unet"`, `encoder_name="efficientnet-b4"`, `batch_size=64"`, `img_size=256`, `max_epochs=5`)
# - V21 - V24: Inference bug fixes
# - V25: 0.812 (`arch="Unet"`, `encoder_name="efficientnet-b1"`, `batch_size=128"`, `img_size=224`, `max_epochs=15`)
# - V26: 0.827 (2.5D Data, `arch="Unet"`, `encoder_name="efficientnet-b1"`, `batch_size=128"`, `img_size=224`, `max_epochs=15`)
# - V27 - V29: Memory errors
# - V30: 0.841 (2.5D Data, use scheduler every step, `arch="Unet"`, `encoder_name="efficientnet-b1"`, `batch_size=128"`, `img_size=224`, `max_epochs=15`)
# - V31: 0.XXX (2.5D Data, use scheduler every step, `arch="Unet"`, `encoder_name="efficientnet-b1"`, `batch_size=96"`, `img_size=256`, `max_epochs=15`)

# # Setup Pretrained Model Checkpoint

# In[ ]:


get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints')
get_ipython().system('cp ../input/gi-seg-downloads/efficientnet-b0-355c32eb.pth /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth')
get_ipython().system('cp ../input/gi-seg-downloads/efficientnet-b1-f1951068.pth /root/.cache/torch/hub/checkpoints/efficientnet-b1-f1951068.pth')
get_ipython().system('cp ../input/gi-seg-downloads/efficientnet-b2-8bb594d6.pth /root/.cache/torch/hub/checkpoints/efficientnet-b2-8bb594d6.pth')
get_ipython().system('cp ../input/gi-seg-downloads/efficientnet-b4-6ed6700e.pth /root/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth')
get_ipython().system('cp ../input/gi-seg-downloads/efficientnet-b6-c76e70fd.pth /root/.cache/torch/hub/checkpoints/efficientnet-b6-c76e70fd.pth')


# # Installs

# In[ ]:


get_ipython().system('cd ../input/gi-seg-downloads && pip install -q efficientnet_pytorch-0.6.3.tar.gz pretrainedmodels-0.7.4.tar.gz timm-0.4.12-py3-none-any.whl  segmentation_models_pytorch-0.2.1-py3-none-any.whl && pip install -q monai-0.8.1-202202162213-py3-none-any.whl && pip install -q torchmetrics-0.8.0-py3-none-any.whl')


# # Imports

# In[ ]:


from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import albumentations as A
import cupy as cp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from monai.metrics.utils import get_mask_edges
from monai.metrics.utils import get_surface_distance
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


# # Paths & Settings

# In[ ]:


KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
INPUT_DATA_NPY_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation-masks"

N_SPLITS = 5
RANDOM_SEED = 2022
IMG_SIZE = 256
VAL_FOLD = 0
LOAD_IMAGES = True # True for 2.5D data
USE_AUGS = True
BATCH_SIZE = 96
NUM_WORKERS = 2
ARCH = "Unet"
ENCODER_NAME = "efficientnet-b1"
ENCODER_WEIGHTS = "imagenet"
LOSS = "bce_tversky"
OPTIMIZER = "Adam"
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-6
SCHEDULER = "CosineAnnealingLR"
MIN_LR = 1e-6

FAST_DEV_RUN = False # Debug training
GPUS = 1
MAX_EPOCHS = 15
PRECISION = 16

CHANNELS = 3
STRIDE = 2
DEVICE = "cuda"
THR = 0.45

DEBUG = False # Debug complete pipeline


# # Dataset

# In[ ]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, load_images: bool, load_mask: bool, transforms: Optional[Callable] = None):
        self.df = df
        self.load_images = load_images
        self.load_mask = load_mask
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.load_images:
            image = self._load_images(eval(row["image_paths"]))
        else:
            image = self._load_image(row["image_path"])

        if self.load_mask:
            mask = self._load_mask(row["mask_path"])

            if self.transforms:
                data = self.transforms(image=image, mask=mask)
                image, mask = data["image"], data["mask"]

            return image, mask
        else:
            id_ = row["id"]
            h, w = image.shape[:2]

            if self.transforms:
                data = self.transforms(image=image)
                image = data["image"]

            return image, id_, h, w
        
    def _load_images(self, paths):
        images = [self._load_image(path, tile=False) for path in paths]
        image = np.stack(images, axis=-1)
        return image

    @staticmethod
    def _load_image(path, tile: bool = True):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image.astype("float32")  # original is uint16
        
        if tile:
            image = np.tile(image[..., None], [1, 1, 3])  # gray to rgb
            
        image /= image.max()

        return image

    @staticmethod
    def _load_mask(path):
        return np.load(path).astype("float32") / 255.0


# # LitDataModule

# In[ ]:


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: Optional[str],
        img_size: int,
        use_augs: bool,
        val_fold: int,
        load_images: bool,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_path)

        if test_csv_path is not None:
            self.test_df = pd.read_csv(test_csv_path)
        else:
            self.test_df = None

        self.train_transforms, self.val_test_transforms = self._init_transforms()

    def _init_transforms(self):
        img_size = self.hparams.img_size

        train_transforms = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        ]
        if self.hparams.use_augs:
            train_transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
            ]
        train_transforms += [ToTensorV2(transpose_mask=True)]
        train_transforms = A.Compose(train_transforms, p=1.0)

        val_test_transforms = [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2(transpose_mask=True),
        ]
        val_test_transforms = A.Compose(val_test_transforms, p=1.0)

        return train_transforms, val_test_transforms

    def setup(self, stage: Optional[str] = None):
        train_df = self.train_df[self.train_df.fold != self.hparams.val_fold].reset_index(drop=True)
        val_df = self.train_df[self.train_df.fold == self.hparams.val_fold].reset_index(drop=True)

        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train_df, load_mask=True, transform=self.train_transforms)
            self.val_dataset = self._dataset(val_df, load_mask=True, transform=self.val_test_transforms)

        if stage == "test" or stage is None:
            if self.test_df is not None:
                self.test_dataset = self._dataset(self.test_df, load_mask=False, transform=self.val_test_transforms)
            else:
                self.test_dataset = self._dataset(val_df, load_mask=True, transform=self.val_test_transforms)

    def _dataset(self, df: pd.DataFrame, load_mask: bool, transform: Callable) -> Dataset:
        return Dataset(df=df, load_images=self.hparams.load_images, load_mask=load_mask, transforms=transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: Dataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


# # Metrics

# In[ ]:


class DiceMetric(Metric):
    def __init__(self, thr=0.5, dim=(2, 3), epsilon=0.001):
        super().__init__(compute_on_cpu=True)

        self.thr = thr
        self.dim = dim
        self.epsilon = epsilon

        self.add_state("dice", default=[])

    def update(self, y_pred, y_true):
        self.dice.append(dice_metric_update(y_pred, y_true, self.thr, self.dim, self.epsilon))

    def compute(self):
        if len(self.dice) == 1:
            return self.dice[0]
        else:
            return torch.mean(torch.stack(self.dice))


def dice_metric_update(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_pred = torch.nn.Sigmoid()(y_pred)
    y_pred = (y_pred > thr).detach().to(torch.float32)

    y_true = y_true.detach().to(torch.float32)

    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)

    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))

    return dice


class IOUMetric(Metric):
    def __init__(self, thr=0.5, dim=(2, 3), epsilon=0.001):
        super().__init__(compute_on_cpu=True)

        self.thr = thr
        self.dim = dim
        self.epsilon = epsilon

        self.add_state("iou", default=[])

    def update(self, y_pred, y_true):
        self.iou.append(iou_metric_update(y_pred, y_true, self.thr, self.dim, self.epsilon))

    def compute(self):
        if len(self.iou) == 1:
            return self.iou[0]
        else:
            return torch.mean(torch.stack(self.iou))


def iou_metric_update(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_pred = torch.nn.Sigmoid()(y_pred)
    y_pred = (y_pred > thr).detach().to(torch.float32)

    y_true = y_true.detach().to(torch.float32)

    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)

    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))

    return iou


class CompetitionMetric(Metric):
    def __init__(self, thr=0.5):
        super().__init__(compute_on_step=False)

        self.thr = thr

        self.add_state("y_pred", default=[])
        self.add_state("y_true", default=[])

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred = torch.nn.Sigmoid()(y_pred)
        y_pred = (y_pred > self.thr).to("cpu").detach().to(torch.float32)

        y_true = y_true.to("cpu").detach().to(torch.float32)

        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def compute(self):
        y_pred = torch.cat(self.y_pred).numpy()
        y_true = torch.cat(self.y_true).numpy()

        return compute_competition_metric(y_pred, y_true)[0]


def compute_competition_metric(preds: np.ndarray, targets: np.ndarray) -> float:
    dice_ = compute_dice(preds, targets)
    hd_dist_ = compute_hd_dist(preds, targets)
    return 0.4 * dice_ + 0.6 * hd_dist_, dice_, hd_dist_


# Slightly adapted from https://www.kaggle.com/code/carnozhao?scriptVersionId=93589877&cellId=2
def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = preds.astype(np.uint8)
    targets = targets.astype(np.uint8)

    I = (targets & preds).sum((2, 3))  # noqa: E741
    U = (targets | preds).sum((2, 3))  # noqa: E741

    return np.mean((2 * I / (U + I + 1) + (U == 0)).mean(1))


def compute_hd_dist(preds: np.ndarray, targets: np.ndarray) -> float:
    return 1 - np.mean([hd_dist_batch(preds[:, i, ...], targets[:, i, ...]) for i in range(3)])


def hd_dist_batch(preds: np.ndarray, targets: np.ndarray) -> float:
    return np.mean([hd_dist(pred, target) for pred, target in zip(preds, targets)])


# From https://www.kaggle.com/code/yiheng?scriptVersionId=93883465&cellId=4
def hd_dist(pred: np.ndarray, target: np.ndarray) -> float:
    if np.all(pred == target):
        return 0.0

    edges_pred, edges_gt = get_mask_edges(pred, target)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")

    if surface_distance.shape == (0,):
        return 0.0

    dist = surface_distance.max()
    max_dist = np.sqrt(np.sum(np.array(pred.shape) ** 2))

    if dist > max_dist:
        return 1.0

    return dist / max_dist


# # LitModule

# In[ ]:


class LitModule(pl.LightningModule):
    LOSS_FNS = {
        "bce": smp.losses.SoftBCEWithLogitsLoss(),
        "dice": smp.losses.DiceLoss(mode="multilabel"),
        "focal": smp.losses.FocalLoss(mode="multilabel"),
        "jaccard": smp.losses.JaccardLoss(mode="multilabel"),
        "lovasz": smp.losses.LovaszLoss(mode="multilabel"),
        "tversky": smp.losses.TverskyLoss(mode="multilabel"),
    }

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str,
        loss: str,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[str],
        T_max: int,
        T_0: int,
        min_lr: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()

    def _init_model(self):
        return smp.create_model(
            self.hparams.arch,
            encoder_name=self.hparams.encoder_name,
            encoder_weights=self.hparams.encoder_weights,
            in_channels=3,
            classes=3,
            activation=None,
        )

    def _init_loss_fn(self):
        losses = self.hparams.loss.split("_")
        loss_fns = [self.LOSS_FNS[loss] for loss in losses]

        def criterion(y_pred, y_true):
            return sum(loss_fn(y_pred, y_true) for loss_fn in loss_fns) / len(loss_fns)

        return criterion

    def _init_metrics(self):
        val_metrics = MetricCollection({"val_dice": DiceMetric(), "val_iou": IOUMetric()})
        test_metrics = MetricCollection({"test_comp_metric": CompetitionMetric()})

        return torch.nn.ModuleDict(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizer_kwargs = dict(
            params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(**optimizer_kwargs)
        elif self.hparams.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(**optimizer_kwargs)
        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(**optimizer_kwargs)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(**optimizer_kwargs)
        elif self.hparams.optimizer == "Adamax":
            optimizer = torch.optim.Adamax(**optimizer_kwargs)
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(**optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
                )
            elif self.hparams.scheduler == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.hparams.T_0, eta_min=self.hparams.min_lr
                )
            elif self.hparams.scheduler == "ExponentialLR":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            elif self.hparams.scheduler == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            else:
                raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return {"optimizer": optimizer}

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def shared_step(self, batch, stage, log=True):
        images, masks = batch
        y_pred = self(images)

        loss = self.loss_fn(y_pred, masks)

        if stage != "train":
            metrics = self.metrics[f"{stage}_metrics"](y_pred, masks)
        else:
            metrics = None

        if log:
            self._log(loss, metrics, stage)

        return loss

    def _log(self, loss, metrics, stage):
        on_step = True if stage == "train" else False

        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)

        if metrics is not None:
            self.log_dict(metrics, on_step=on_step, on_epoch=True)

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module


# # Train

# In[ ]:


def train(
    random_seed: int = RANDOM_SEED,
    train_csv_path: str = str(INPUT_DATA_NPY_DIR / "train_preprocessed.csv"),
    img_size: int = IMG_SIZE,
    use_augs: bool = USE_AUGS,
    val_fold: int = VAL_FOLD,
    load_images: bool = LOAD_IMAGES,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    arch: str = ARCH,
    encoder_name: str = ENCODER_NAME,
    encoder_weights: str = ENCODER_WEIGHTS,
    loss: str = LOSS,
    optimizer: str = OPTIMIZER,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    scheduler: str = SCHEDULER,
    min_lr: float = MIN_LR,
    gpus: int = GPUS,
    fast_dev_run: bool = FAST_DEV_RUN,
    max_epochs: int = MAX_EPOCHS,
    precision: int = PRECISION,
    debug: bool = DEBUG,
):
    pl.seed_everything(random_seed)

    if debug:
        num_workers = 0
        max_epochs = 2

    data_module = LitDataModule(
        train_csv_path=train_csv_path,
        test_csv_path=None,
        img_size=img_size,
        use_augs=use_augs,
        val_fold=val_fold,
        load_images=load_images,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    module = LitModule(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        loss=loss,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        T_max=int(30_000 / batch_size * max_epochs) + 50,
        T_0=25,
        min_lr=min_lr,
    )

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        limit_train_batches=0.02 if debug else 1.0,
        limit_val_batches=0.02 if debug else 1.0,
        limit_test_batches=0.02 if debug else 0.5, # Metric computation takes too much memory otherwise
        logger=pl.loggers.CSVLogger(save_dir='logs/'),
        log_every_n_steps=10,
        max_epochs=max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)
    
    
    if not fast_dev_run:
        trainer.test(module, datamodule=data_module)
    
    return trainer


# In[ ]:


trainer = train()


# In[ ]:


# From https://www.kaggle.com/code/jirkaborovec?scriptVersionId=93358967&cellId=22
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")[["epoch", "train_loss_epoch", "val_loss"]]
metrics.set_index("epoch", inplace=True)

sns.relplot(data=metrics, kind="line", height=5, aspect=1.5)
plt.grid()


# # Infer

# ### Load Test Data

# In[ ]:


def extract_metadata_from_id(df):
    df[["case", "day", "slice"]] = df["id"].str.split("_", n=2, expand=True)

    df["case"] = df["case"].str.replace("case", "").astype(int)
    df["day"] = df["day"].str.replace("day", "").astype(int)
    df["slice"] = df["slice"].str.replace("slice_", "").astype(int)

    return df


def extract_metadata_from_path(path_df):
    path_df[["parent", "case_day", "scans", "file_name"]] = path_df["image_path"].str.rsplit("/", n=3, expand=True)

    path_df[["case", "day"]] = path_df["case_day"].str.split("_", expand=True)
    path_df["case"] = path_df["case"].str.replace("case", "")
    path_df["day"] = path_df["day"].str.replace("day", "")

    path_df[["slice", "width", "height", "spacing", "spacing_"]] = (
        path_df["file_name"].str.replace("slice_", "").str.replace(".png", "").str.split("_", expand=True)
    )
    path_df = path_df.drop(columns=["parent", "case_day", "scans", "file_name", "spacing_"])

    numeric_cols = ["case", "day", "slice", "width", "height", "spacing"]
    path_df[numeric_cols] = path_df[numeric_cols].apply(pd.to_numeric)

    return path_df


def add_image_paths(df, channels, stride):
    for i in range(channels):
        df[f"image_path_{i:02}"] = df.groupby(["case", "day"])["image_path"].shift(-i * stride).fillna(method="ffill")
    
    image_path_columns = [f"image_path_{i:02d}" for i in range(channels)]
    df["image_paths"] = df[image_path_columns].values.tolist()
    df = df.drop(columns=image_path_columns)
    
    return df


# In[ ]:


sub_df = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv")
test_set_hidden = not bool(len(sub_df))

if test_set_hidden:
    test_df = pd.read_csv(INPUT_DATA_DIR / "train.csv")[: 1000 * 3]
    test_df = test_df.drop(columns=["class", "segmentation"]).drop_duplicates()
    image_paths = [str(path) for path in (INPUT_DATA_DIR / "train").rglob("*.png")]
else:
    test_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
    image_paths = [str(path) for path in (INPUT_DATA_DIR / "test").rglob("*.png")]

test_df = extract_metadata_from_id(test_df)

path_df = pd.DataFrame(image_paths, columns=["image_path"])
path_df = extract_metadata_from_path(path_df)

test_df = test_df.merge(path_df, on=["case", "day", "slice"], how="left")
test_df = add_image_paths(test_df, CHANNELS, STRIDE)

print(len(test_df))
test_df.head()


# ### Save Test DataFrame

# In[ ]:


test_df.to_csv("test_preprocessed.csv", index=False)


# ## Run inference

# In[ ]:


def mask2rle(mask):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    mask = cp.array(mask)
    pixels = mask.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def masks2rles(masks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx in range(masks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        mask = cv2.resize(masks[idx], dsize=(width, height), interpolation=cv2.INTER_NEAREST)  # back to original shape

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(mask[..., midx])

        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])

    return pred_strings, pred_ids, pred_classes


@torch.no_grad()
def infer(img_size, load_images, batch_size, num_workers, model_paths, device, thr):
    data_module = LitDataModule(
        train_csv_path=str(INPUT_DATA_NPY_DIR / "train_preprocessed.csv"),
        test_csv_path="test_preprocessed.csv",
        img_size=img_size,
        use_augs=False,
        val_fold=0,
        load_images=load_images,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup(stage="test")
    test_dataloader = data_module.test_dataloader()
    
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for imgs, ids, heights, widths in tqdm(test_dataloader):
        imgs = imgs.to(device, dtype=torch.float)
        size = imgs.size()

        masks = []
        masks = torch.zeros((size[0], 3, size[2], size[3]), device=device, dtype=torch.float32)

        for path in model_paths:
            model = LitModule.load_eval_checkpoint(path, device=device)
            out = model(imgs)
            out = torch.nn.Sigmoid()(out)
            masks += out / len(model_paths)

        masks = (masks.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().detach().numpy()  # shape: (n, h, w, c)

        result = masks2rles(masks, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

    pred_df = pd.DataFrame({"id": pred_ids, "class": pred_classes, "predicted": pred_strings})

    return pred_df


# In[ ]:


model_paths = list((Path(trainer.logger.log_dir) / "checkpoints").glob("*.ckpt"))
model_paths


# In[ ]:


pred_df = infer(IMG_SIZE, LOAD_IMAGES, 32, NUM_WORKERS, model_paths, DEVICE, THR)


# ## Submit

# In[ ]:


if not test_set_hidden:
    sub_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv")
    del sub_df["predicted"]
else:
    sub_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")[: 1000 * 3]
    del sub_df["segmentation"]

sub_df = sub_df.merge(pred_df, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)
display(sub_df.head(5))


# ## 
