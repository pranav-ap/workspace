"""
Imports
"""

import os
from dataclasses import dataclass

import lightning as L
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, \
    TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

torch.set_float32_matmul_precision('medium')


@dataclass
class TrainingConfig:
    image_size = 28  # the generated image resolution

    train_batch_size = 32
    val_batch_size = 32

    max_epochs = 40
    check_val_every_n_epoch = 5
    log_every_n_steps = 5
    accumulate_grad_batches = 1
    learning_rate = 1e-5

    data_dir = "D:/workshop/data/mnist"
    output_dir = "D:/workshop/data/mnist"

    seed = 10


config = TrainingConfig()

transform = T.Compose([
    T.Normalize([0.5], [0.5]),
])

reverse_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([-0.5 / 0.5], [1 / 0.5]),
    T.ToPILImage(),
])


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, labels, images, transform=None, pred=False):
        super().__init__()
        self.labels = labels
        self.images = images
        self.pred = pred
        self.transform = transform

        if not pred:
            assert len(self.labels) == len(self.images)

    def __len__(self):
        length = len(self.images)
        return length

    def __getitem__(self, index):
        image = self.images[index]
        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        if not self.pred:
            label = self.labels[index]
            return label, image

        return image


class MNISTDataModule(L.LightningDataModule):
    def __init__(self,
                 train_transform,
                 test_transform):
        super().__init__()
        self.num_workers = os.cpu_count()  # <- use all available CPU cores

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.train_dataset: MNISTDataset | None = None
        self.val_dataset: MNISTDataset | None = None
        self.predict_dataset: MNISTDataset | None = None

    def setup(self, stage: str):
        if stage == "fit":
            all_data = pd.read_csv(f"{config.data_dir}/train.csv")

            train_data = all_data.iloc[5000:, :]
            val_data = all_data.iloc[:5000, :]

            train_labels = torch.tensor(train_data.label.values)
            val_labels = torch.tensor(val_data.label.values)

            # Reshaping data
            train_images = train_data.iloc[:, 1:].values.reshape(-1, 28, 28)
            val_images = val_data.iloc[:, 1:].values.reshape(-1, 28, 28)

            train_images = torch.from_numpy(train_images).type(torch.float32)
            val_images = torch.from_numpy(val_images).type(torch.float32)

            self.train_dataset = MNISTDataset(
                labels=train_labels,
                images=train_images,
                transform=self.train_transform
            )

            self.val_dataset = MNISTDataset(
                labels=val_labels,
                images=val_images,
                transform=self.test_transform
            )

            print(f"Total Dataset       : {len(self.train_dataset) + len(self.val_dataset)} samples")
            print(f"Train Dataset       : {len(self.train_dataset)} samples")
            print(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == 'predict':
            samples = pd.read_csv(f"{config.data_dir}/test.csv")
            images = samples.iloc[:, :].values.reshape(-1, 28, 28)
            images = torch.from_numpy(images).type(torch.float32)

            self.predict_dataset = MNISTDataset(
                labels=None,
                images=images,
                transform=self.test_transform,
                pred=True
            )

            print(f"Predict Dataset  : {len(self.predict_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


"""
Model
"""


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=512, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5),
            nn.ELU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2),
            nn.ELU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1),
        )

        self.classifier = nn.Sequential(
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        x = torch.squeeze(x)
        return x


"""
Lightning Module
"""


class LightningMNIST(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.learning_rate

        self.save_hyperparameters(ignore=['model'])

    def forward(self, images):
        labels_logits = self.model(images)
        return labels_logits

    def shared_step(self, batch):
        labels, images = batch
        labels_logits = self.model(images)

        loss = F.nll_loss(labels_logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)

        return {
            "optimizer": optimizer,
            # "monitor": "train_loss",
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            }
        }

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.00,
            patience=3,
            verbose=False,
        )

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.output_dir}/checkpoints/',
            save_top_k=1,
            save_last=True
        )

        progress_bar = TQDMProgressBar(process_position=0)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # summary = ModelSummary(max_depth=-1)
        # swa = StochasticWeightAveraging(swa_lrs=1e-2)

        return [checkpoint, progress_bar, lr_monitor, early_stop]


"""
Main Function
"""


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MNISTModel()

    lightning_module = LightningMNIST(
        model
    )

    dm = MNISTDataModule(
        train_transform=transform,
        test_transform=transform
    )

    trainer = pl.Trainer(
        default_root_dir=f"{config.output_dir}/",
        logger=TensorBoardLogger(save_dir=f'{config.output_dir}/'),
        devices='auto',
        accelerator="auto",  # auto, gpu, cpu, ...

        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # gradient_clip_val=0.1,

        # fast_dev_run=True,
        # overfit_batches=1,
        num_sanity_val_steps=1,
        enable_model_summary=False,
    )

    trainer.fit(
        lightning_module,
        datamodule=dm,
        # ckpt_path=f'{config.output_dir}\\checkpoints\\last.ckpt'
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    # "D:\workshop\data\mnist\checkpoints\epoch=49-step=7250.ckpt"
    print(f"Best model path : {best_model_path}")

    labels_logits = trainer.predict(
        ckpt_path=best_model_path,
        dataloaders=dm
    )

    pred_labels = np.argmax(torch.cat(labels_logits, dim=0).detach().numpy(), axis=1)

    submission = pd.concat([
        pd.Series(range(1, 28001), name="ImageId"),
        pd.Series(pred_labels, name="Label")],
        axis=1
    )

    submission.to_csv("mnist_submission.csv", index=False)


if __name__ == "__main__":
    main()
