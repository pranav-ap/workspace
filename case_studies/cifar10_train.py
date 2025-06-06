"""
Imports
"""


from utils.logger_setup import logger
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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

    max_epochs = 40
    check_val_every_n_epoch = 5
    log_every_n_steps = 200
    accumulate_grad_batches = 1
    learning_rate = 1e-5

    data_dir = "/kaggle/working/data/"
    output_dir = "/kaggle/working/"


config = TrainingConfig()


"""
Dataset Classes
"""


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, images, transform, pred=False):
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

        if self.transform:
            image = self.transform(image)

        if not self.pred:
            label = self.labels[index]
            return label, image

        return image


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.CLASS_NAMES = [
            'airplane', 'automobile', 'bird',
            'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]

        self.num_workers = os.cpu_count()  # <- use all available CPU cores
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            self.num_workers = 2 * num_gpus

        self.train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.RandomAffine(0, shear=10, scale=(0.8,1.2)), # Performs actions like zooms, change shear angles.
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

        self.test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

        self.train_dataset: CIFAR10Dataset | None = None
        self.val_dataset: CIFAR10Dataset | None = None
        self.predict_dataset: CIFAR10Dataset | None = None

    # Convert the filepath to an integer image ID
    @staticmethod
    def filename_to_id(filename: str):
        x = filename.split('/')[-1].split('.')[0]
        return int(x)

    # Load all of the images into an array, also return the image IDs
    @staticmethod
    def load_images_and_ids(img_directory_path):
        import glob

        image_filenames = glob.glob(img_directory_path + '/*.png')
        list_of_images = []
        image_ids = []

        for image_filename in image_filenames:
            # '/kaggle/working/train/44240.png' -> 44240
            image_id = CIFAR10DataModule.filename_to_id(image_filename)
            image_ids.append(image_id)

            # image = cv2.imread(image_filename)
            image = torchvision.io.read_image(image_filename)
            list_of_images.append(image)

        X = np.array(list_of_images)

        return X, image_ids

    def setup(self, stage: str):
        if stage == "fit":
            X_train, train_image_ids = CIFAR10DataModule.load_images_and_ids(
                f"{config.data_dir}/train/"
            )

            # X_train = X_train / 255.0

            # Read the labels from the CSV files and convert the array of IDs into a 0-9 code
            labels_dataframe = pd.read_csv(f"{config.data_dir}/trainLabels.csv")

            img_codes = [
                self.CLASS_NAMES.index(
                    labels_dataframe.loc[img_id - 1]['label'])
                for img_id in train_image_ids
            ]

            y_train = np.array(img_codes)

            # Splitting

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2)

            logger.debug(f'X_train  - {X_train.shape}, {X_train.dtype}')
            logger.debug(f'X_val    - {X_val.shape}, {X_val.dtype}')
            logger.debug(f'y_train  - {y_train.shape}, {y_train.dtype}')
            logger.debug(f'y_val    - {y_val.shape}, {y_val.dtype}')

            logger.debug('Convert numpy to tensor')

            X_train = torch.from_numpy(X_train).type(torch.float32)
            X_val = torch.from_numpy(X_val).type(torch.float32)

            y_train = torch.from_numpy(y_train).type(torch.float32)
            y_val = torch.from_numpy(y_val).type(torch.float32)

            logger.debug(f'X_train  - {X_train.shape}, {X_train.dtype}')
            logger.debug(f'X_val    - {X_val.shape}, {X_val.dtype}')
            logger.debug(f'y_train  - {y_train.shape}, {y_train.dtype}')
            logger.debug(f'y_val    - {y_val.shape}, {y_val.dtype}')

            self.train_dataset = CIFAR10Dataset(
                labels=y_train,
                images=X_train,
                transform=self.train_transform
            )

            self.val_dataset = CIFAR10Dataset(
                labels=y_val,
                images=X_val,
                transform=self.test_transform
            )

            logger.info(f"Total Dataset       : {len(self.train_dataset) + len(self.val_dataset)} samples")
            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

        if stage == 'predict':
            X_test, test_image_ids = CIFAR10DataModule.load_images_and_ids(
                f"{config.data_dir}/test/"
            )

            # X_test = X_test / 255.0

            logger.debug(f'X_test  - {X_test.shape}, {X_test.dtype}')
            logger.debug('Convert numpy to tensor')
            X_test = torch.from_numpy(X_test).type(torch.float32)
            logger.debug(f'X_test  - {X_test.shape}, {X_test.dtype}')

            self.predict_dataset = CIFAR10Dataset(
                labels=None,
                images=X_test,
                transform=self.test_transform,
                pred=True
            )

            logger.info(f"Predict Dataset  : {len(self.predict_dataset)} samples")

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


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )

        # Substitute the FC output layer
        resnet.fc = torch.nn.Linear(in_features=resnet.fc.in_features, out_features=10)
        torch.nn.init.xavier_uniform_(resnet.fc.weight)

        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.model = resnet

    def forward(self, x):
        # logger.debug(f'Input shape : {x.shape}')
        x = self.model(x)
        x = torch.squeeze(x)
        # logger.debug(f'Output shape : {x.shape}')

        return x


"""
Lightning Module
"""


class LightningCIFAR10(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.learning_rate = config.learning_rate

        self.save_hyperparameters(ignore=['model'])

    def forward(self, images):
        labels_logits = self.model(images)
        return labels_logits

    def shared_step(self, batch):
        labels, images = batch
        labels_logits = self.model(images)

        loss = F.cross_entropy(labels_logits, labels.long())

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
        from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.00,
            patience=4,
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
    logger.info(f"Using device: {device}")

    model = CIFAR10Model()
    lightning_module = LightningCIFAR10(model)
    dm = CIFAR10DataModule()

    from lightning.pytorch.loggers import TensorBoardLogger

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
    logger.info(f"Best model path : {best_model_path}")

    labels_logits = trainer.predict(
        ckpt_path=best_model_path,
        dataloaders=dm
    )

    pred_labels = np.argmax(torch.cat(labels_logits, dim=0).detach().numpy(), axis=1)

    ids = list(range(1, len(pred_labels) + 1))
    ids.sort(key=lambda x: str(x))

    submission = pd.concat([
        pd.Series(ids, name="id"),
        pd.Series(pred_labels, name="label")],
        axis=1
    )

    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
