from utils.logger_setup import logger
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import lightning.pytorch as pl

torch.set_float32_matmul_precision('medium')


"""
Configuration
"""


@dataclass
class TrainingConfig:
    train_batch_size = 128
    val_batch_size = 128

    max_epochs = 20
    check_val_every_n_epoch = 2
    log_every_n_steps = 200
    accumulate_grad_batches = 1
    learning_rate = 1e-5

    data_dir = "D:/workshop/data/aussie_rain"
    output_dir = "D:/workshop/data/aussie_rain"


config = TrainingConfig()


"""
Dataset Classes
"""


class AusRainDataset(torch.utils.data.Dataset):
    def __init__(self, X, stage):
        super().__init__()
        self.X = X
        self.stage = stage

    def __len__(self):
        length = len(self.X)
        return length

    def __getitem__(self, index):
        x = self.X[index]
        return x


class AusRainDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()  # <- use all available CPU cores
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            self.num_workers = 2 * num_gpus

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        self.train_dataset: AusRainDataset | None = None
        self.val_dataset: AusRainDataset | None = None
        self.test_dataset: AusRainDataset | None = None

    def setup(self, stage: str):
        if stage == "fit" or stage == "validate":
            X = pd.read_csv(f"{config.data_dir}/train.csv")
            logger.debug(f"X shape - {X.shape}")

            X = self.scaler.fit_transform(X)

            from sklearn.model_selection import train_test_split
            X_train, X_val = train_test_split(X, test_size=0.2)

            X_train = torch.from_numpy(X_train).float()
            X_val = torch.from_numpy(X_val).float()

            logger.debug(f"X_train - {X_train.shape} - {X_train.dtype}")
            logger.debug(f"X_val shape - {X_val.shape} - {X_val.dtype}")

            self.train_dataset = AusRainDataset(
                X=X_train,
                stage='train'
            )

            self.val_dataset = AusRainDataset(
                X=X_val,
                stage='val'
            )

            logger.info(f"Total Dataset       : {len(self.train_dataset) + len(self.val_dataset)} samples")
            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == 'test':
            X_test = pd.read_csv(f"{config.data_dir}/test.csv")
            X_test = self.scaler.transform(X_test)
            X_test = torch.from_numpy(X_test).float()

            self.test_dataset = AusRainDataset(
                X=X_test,
                stage='test'
            )

            logger.info(f"Test Dataset  : {len(self.test_dataset)} samples")

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

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


class AusRainAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 10),
            nn.ReLU(),
            nn.Linear(10, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # logger.debug(f'Input shape : {x.shape}')
        x = self.encoder(x)
        # logger.debug(f'Encoder Output shape : {x.shape}')
        x = self.decoder(x)
        # logger.debug(f'Decoder Output shape : {x.shape}')
        return x


"""
Lightning Module
"""


class AusRainLightning(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.learning_rate

        self.save_hyperparameters(ignore=['model'])

    def forward(self, X):
        y_pred = self.model(X)
        return y_pred

    def shared_step(self, batch):
        X, y = batch
        X_reconstructed = self.model(X)

        loss = F.mse_loss(X_reconstructed, X)

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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            }
        }

    def configure_callbacks(self):
        early_stop = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.00,
            patience=4,
            verbose=False,
        )

        checkpoint = L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.output_dir}/checkpoints/',
            save_top_k=1,
            save_last=True
        )

        progress_bar = L.pytorch.callbacks.TQDMProgressBar(process_position=0)
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        # summary = ModelSummary(max_depth=-1)
        # swa = StochasticWeightAveraging(swa_lrs=1e-2)

        return [checkpoint, progress_bar, lr_monitor, early_stop]


"""
Train Function
"""


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    input_size = 17  # Number of input features
    encoding_dim = 3  # Desired number of output dimensions

    autoencoder_model = AusRainAutoencoder(input_size, encoding_dim)
    lightning_module = AusRainLightning(autoencoder_model)
    dm = AusRainDataModule()

    trainer = pl.Trainer(
        default_root_dir=f"{config.output_dir}/",
        logger=L.pytorch.loggers.CSVLogger(save_dir=f'{config.output_dir}/'),
        devices='auto',
        accelerator="auto",  # auto, gpu, cpu, ...

        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # gradient_clip_val=0.1,

        fast_dev_run=True,
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
    logger.info(f"Best model path : {best_model_path}")



