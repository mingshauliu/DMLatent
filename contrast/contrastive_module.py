import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch.nn.functional as F
from models import ContrastiveCNN, NECTLoss, NXTentLoss, WDMClassifierTiny, WDMClassifierMedium, WDMClassifierLarge

class ContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        if config['model_type'] == 'tiny':
            encoder = WDMClassifierTiny()
            self.classifier = WDMClassifierTiny()  
        elif config['model_type'] == 'medium':
            encoder = WDMClassifierMedium()
            self.classifier = WDMClassifierMedium()
        elif config['model_type'] == 'large':
            encoder = WDMClassifierLarge()
            self.classifier = WDMClassifierLarge()
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}")
        
        if config['pretrained_path'] is not None:
            checkpoint = torch.load(config["pretrained_path"], map_location="cpu", weights_only=True)
            state_dict = checkpoint["model_state_dict"]

            encoder.load_state_dict(state_dict, strict=False)
            self.classifier.load_state_dict(state_dict, strict=False)

            print(f"[INFO] Loaded pretrained weights from {config['pretrained_path']}")
        else:
            print("[INFO] No pretrained weights provided, initializing from scratch")
            
        self.model = ContrastiveCNN(encoder)
        
        if config['loss_type'] == 'nect':
            self.loss_fn = NECTLoss(temperature=config.get('temperature', 0.1))
        elif config['loss_type'] == 'nxtent':
            self.loss_fn = NXTentLoss(temperature=config.get('temperature', 0.1))
        else:
            raise ValueError(f"Unknown loss type: {config['loss_type']}")
        
        self.val_latents_list = []
        self.val_labels_list = []
        self.val_softscores_list = []
        
        self.config = config
        
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        self.cls_to_sim = nn.Linear(self.classifier.out_features, 128, bias=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pair_type = self.config.get('pair_type', 'CDMWDM')
        if pair_type == 'CDMWDM':            
            x1, x2, y1, y2 = batch
            z1 = self(x1)
            z2 = self(x2)
            loss = self.loss_fn(z1, z2, y1, y2)
            self.log("train_loss", loss, prog_bar=True)
            return loss
        elif pair_type == 'SimCLR':
            x1, x2 = batch
            z1 = self(x1)
            z2 = self(x2)
            loss_nxtent = self.loss_fn(z1, z2, None, None)
            # Compute KL alignment loss
            with torch.no_grad():
                cls_feat1 = self.classifier.forward_features(x1)  # logits from frozen classifier
                cls_feat2 = self.classifier.forward_features(x2)
                
                cls_proj1 = self.cls_to_sim(cls_feat1)
                cls_proj2 = self.cls_to_sim(cls_feat2)
                
                # Optionally normalize both spaces
                z1n = F.normalize(z1, dim=1)
                z2n = F.normalize(z2, dim=1)
                cls_proj1n = F.normalize(cls_proj1, dim=1)
                cls_proj2n = F.normalize(cls_proj2, dim=1)
                
            # KL divergence loss
            loss_kl = (1 - F.cosine_similarity(z1n, cls_proj1n, dim=1)).mean() + (1 - F.cosine_similarity(z2n, cls_proj2n, dim=1)).mean()
            loss_kl = loss_kl / 2  # Average over both views
            
            # Combine
            alpha = self.config.get("kl_weight", 0.1)
            loss = loss_nxtent + alpha * loss_kl
            self.log("train_loss", loss, prog_bar=True)
            self.log("loss_ntxent", loss_nxtent)
            self.log("loss_kl", loss_kl)
            return loss
        else:
            raise ValueError(f"Unknown pair type: {pair_type}")
    
    def validation_step(self, batch, batch_idx):
        
        if self.config.get('pair_type', 'CDMWDM') == 'CDMWDM':
            x1, x2, y1, y2 = batch
            z1 = self(x1)
            z2 = self(x2)
            loss = self.loss_fn(z1, z2, y1, y2)
            self.log("val_loss", loss, prog_bar=True)

        else:  # SimCLR
            x1, x2 = batch
            z1 = self(x1)
            z2 = self(x2)
            loss_mix = self.loss_fn(z1, z2, None, None)
            self.log("val_loss", loss_mix, prog_bar=True)
            
            with torch.no_grad():
                logits1 = self.classifier(x1)
                logits2 = self.classifier(x2)
                y1 = torch.sigmoid(logits1).squeeze()  # shape [B]
                y2 = torch.sigmoid(logits2).squeeze()
                
        # Store softscores (use y1 and y2 directly if already soft)
        self.val_softscores_list.append(y1.detach().cpu())
        self.val_softscores_list.append(y2.detach().cpu())
        # Store latents
        self.val_latents_list.append(z1.detach().cpu())
        self.val_latents_list.append(z2.detach().cpu())
        
    
    def on_validation_epoch_end(self):
        if len(self.val_latents_list) == 0:
            return  # no validation run

        self.val_latents = torch.cat(self.val_latents_list, dim=0)
        self.val_softscores = torch.cat(self.val_softscores_list, dim=0)

        # Clear for next epoch
        self.val_latents_list.clear()
        self.val_softscores_list.clear()



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     min_lr=1e-6
        # )
        
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=1e-6
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

