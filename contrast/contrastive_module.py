import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from models import ContrastiveCNN, NECTLoss, WDMClassifierTiny

class ContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        encoder = WDMClassifierTiny()
        
        if config['pretrained_path'] is not None:
            checkpoint = torch.load(config["pretrained_path"], map_location="cpu", weights_only=True)
            state_dict = checkpoint["model_state_dict"]

            encoder.load_state_dict(state_dict, strict=False)
                
            print(f"[INFO] Loaded pretrained weights from {config['pretrained_path']}")
        else:
            print("[INFO] No pretrained weights provided, initializing from scratch")
            
        self.model = ContrastiveCNN(encoder)
        self.loss_fn = NECTLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x1, x2, y1, y2 = batch
        z1 = self(x1)
        z2 = self(x2)
        loss = self.loss_fn(z1, z2, y1, y2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # or 'val_loss' if available
                "interval": "epoch",
                "frequency": 1,
            }
        }

