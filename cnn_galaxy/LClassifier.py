import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import (
    WDMClassifierTiny,
    WDMClassifierMedium,
    WDMClassifierLarge,
    WDMClassifierHuge,
    WDMClassifierTinywithDownSampling
)

class LClassifier(pl.LightningModule):
    def __init__(self, model_type='tiny', lr=1e-3, weight_decay=1e-4, dropout=0.0):
        super().__init__()
        self.save_hyperparameters()

        if model_type == 'tiny':
            self.model = WDMClassifierTiny(dropout=dropout)
        elif model_type == 'tiny_downsample':
            self.model = WDMClassifierTinywithDownSampling(dropout=dropout)
        elif model_type == 'medium':
            self.model = WDMClassifierMedium(dropout=dropout)
        elif model_type == 'large':
            self.model = WDMClassifierLarge(dropout=dropout)
        elif model_type == 'huge':
            self.model = WDMClassifierHuge(dropout=dropout)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x).squeeze(1)  # Output shape: [B]

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",        # minimize validation loss
                factor=0.5,        # halve the LR when plateauing
                patience=5,        # wait 5 epochs of no improvement
                verbose=True,
                min_lr=1e-6
            ),
            "monitor": "val_loss",  # must match the logged name
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

