# contrastive_module.py - Updated with hierarchical loss support

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch.nn.functional as F
from models import ContrastiveCNN, NECTLoss, NXTentLoss, WDMClassifierTiny, WDMClassifierMedium, WDMClassifierLarge
from hierarchical_losses import HierarchicalNTXentLoss, AdaptiveHierarchicalLoss, MultiLevelContrastiveLoss

class HierarchicalContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Build encoder
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
        
        # Load pretrained weights if available
        if config['pretrained_path'] is not None:
            checkpoint = torch.load(config["pretrained_path"], map_location="cpu", weights_only=True)
            state_dict = checkpoint["model_state_dict"]

            encoder.load_state_dict(state_dict, strict=False)
            self.classifier.load_state_dict(state_dict, strict=False)

            print(f"[INFO] Loaded pretrained weights from {config['pretrained_path']}")
        else:
            print("[INFO] No pretrained weights provided, initializing from scratch")
            
        self.model = ContrastiveCNN(encoder)
        
        # Choose hierarchical loss function
        if config['loss_type'] == 'hierarchical_nxtent':
            self.loss_fn = HierarchicalNTXentLoss(
                temperature=config.get('temperature', 0.05),
                matter_weight=config.get('matter_weight', 1.0),
                cosmology_weight=config.get('cosmology_weight', 0.5),
                component_weight=config.get('component_weight', 0.25)
            )
        elif config['loss_type'] == 'adaptive_hierarchical':
            self.loss_fn = AdaptiveHierarchicalLoss(
                temperature=config.get('temperature', 0.05),
                base_matter_weight=config.get('matter_weight', 2.0),
                base_cosmology_weight=config.get('cosmology_weight', 1.0),
                base_component_weight=config.get('component_weight', 0.5)
            )
        elif config['loss_type'] == 'multilevel':
            self.loss_fn = MultiLevelContrastiveLoss(
                temperature=config.get('temperature', 0.05),
                level_weights=config.get('level_weights', [1.0, 0.7, 0.3])
            )
        elif config['loss_type'] == 'nect':
            self.loss_fn = NECTLoss(temperature=config.get('temperature', 0.1))
        elif config['loss_type'] == 'nxtent':
            self.loss_fn = NXTentLoss(temperature=config.get('temperature', 0.1))
        else:
            raise ValueError(f"Unknown loss type: {config['loss_type']}")
        
        # Storage for validation embeddings and labels
        self.val_latents_list = []
        self.val_labels_list = []
        self.val_matter_types_list = []
        self.val_cosmologies_list = []
        self.val_components_list = []
        self.val_softscores_list = []
        
        self.config = config
        
        # Freeze classifier for auxiliary tasks
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        # Projection head for auxiliary supervision (optional)
        if config.get('use_auxiliary_classifier', False):
            self.cls_to_sim = nn.Linear(self.classifier.out_features, 128, bias=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pair_type = self.config.get('pair_type', 'MultiComponent')
        
        if pair_type == 'MultiComponent':
            return self._train_multicomponent(batch)
        elif pair_type == 'CDMWDM':            
            return self._train_cdmwdm(batch)
        elif pair_type == 'SimCLR':
            return self._train_simclr(batch)
        else:
            raise ValueError(f"Unknown pair_type: {pair_type}")
    
    def _train_multicomponent(self, batch):
        """Training with hierarchical multicomponent loss"""
        x1, x2 = batch['map1'], batch['map2']
        matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
        cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2'] 
        component1, component2 = batch['component1'], batch['component2']
        
        # Get embeddings
        z1 = self(x1)
        z2 = self(x2)
        
        # Update epoch for adaptive loss
        if hasattr(self.loss_fn, 'set_epoch'):
            self.loss_fn.set_epoch(self.current_epoch)
        
        # Compute hierarchical loss
        if isinstance(self.loss_fn, MultiLevelContrastiveLoss):
            loss, loss_components = self.loss_fn(
                z1, z2, matter_type1, matter_type2, 
                cosmology1, cosmology2, component1, component2
            )
            # Log individual loss components with proper Lightning logging
            self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train/matter_loss", loss_components['matter_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train/cosmology_loss", loss_components['cosmology_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train/component_loss", loss_components['component_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            
            # Log hierarchy weights for adaptive loss
            if hasattr(self.loss_fn, 'level_weights'):
                for i, weight in enumerate(self.loss_fn.level_weights):
                    level_names = ['matter', 'cosmology', 'component']
                    self.log(f"train/weight_{level_names[i]}", weight, on_epoch=True, logger=True)
        else:
            loss = self.loss_fn(
                z1, z2, matter_type1, matter_type2,
                cosmology1, cosmology2, component1, component2
            )
            self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Log learning rate
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, logger=True)
        
        # Log adaptive weights if using adaptive loss
        if hasattr(self.loss_fn, 'matter_weight'):
            self.log("train/adaptive_matter_weight", self.loss_fn.matter_weight, on_epoch=True, logger=True)
            self.log("train/adaptive_cosmology_weight", self.loss_fn.cosmology_weight, on_epoch=True, logger=True)
            self.log("train/adaptive_component_weight", self.loss_fn.component_weight, on_epoch=True, logger=True)
        
        # Legacy logging for backward compatibility
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def _train_cdmwdm(self, batch):
        """Legacy CDM/WDM training"""
        x1, x2, y1, y2 = batch
        z1 = self(x1)
        z2 = self(x2)
        loss = self.loss_fn(z1, z2, y1, y2)
        
        # Proper Lightning logging
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, logger=True)
        
        # Legacy logging
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def _train_simclr(self, batch):
        """SimCLR-style training with auxiliary supervision"""
        x1, x2 = batch
        z1 = self(x1)
        z2 = self(x2)
        
        # Base contrastive loss
        if hasattr(self.loss_fn, '__name__') and 'Hierarchical' in self.loss_fn.__class__.__name__:
            # If using hierarchical loss, need to infer labels from classifier
            with torch.no_grad():
                logits1 = self.classifier(x1)
                logits2 = self.classifier(x2)
                # Simple binary classification: 0=CDM, 1=WDM
                y1 = torch.sigmoid(logits1).squeeze()
                y2 = torch.sigmoid(logits2).squeeze()
                
                # Convert to hierarchical labels (this is a simplification)
                matter_type1 = torch.zeros_like(y1, dtype=torch.long)  # Assume all are "total mass"
                matter_type2 = torch.zeros_like(y2, dtype=torch.long)
                cosmology1 = (y1 > 0.5).long()  # 0=CDM, 1=WDM
                cosmology2 = (y2 > 0.5).long()
                component1 = torch.zeros_like(y1, dtype=torch.long)
                component2 = torch.zeros_like(y2, dtype=torch.long)
                
            loss = self.loss_fn(z1, z2, matter_type1, matter_type2, 
                              cosmology1, cosmology2, component1, component2)
        else:
            loss = self.loss_fn(z1, z2, None, None)  # Standard NT-Xent
        
        # Optional auxiliary alignment loss
        if self.config.get('use_auxiliary_classifier', False):
            with torch.no_grad():
                cls_feat1 = self.classifier.forward_features(x1)
                cls_feat2 = self.classifier.forward_features(x2)
                cls_proj1 = self.cls_to_sim(cls_feat1)
                cls_proj2 = self.cls_to_sim(cls_feat2)
                
            z1n = F.normalize(z1, dim=1)
            z2n = F.normalize(z2, dim=1)
            cls_proj1n = F.normalize(cls_proj1, dim=1)
            cls_proj2n = F.normalize(cls_proj2, dim=1)
            
            loss_align = (1 - F.cosine_similarity(z1n, cls_proj1n, dim=1)).mean() + \
                        (1 - F.cosine_similarity(z2n, cls_proj2n, dim=1)).mean()
            loss_align = loss_align / 2
            
            alpha = self.config.get("alignment_weight", 0.1)
            loss = loss + alpha * loss_align
            
            # Log auxiliary loss
            self.log("train/alignment_loss", loss_align, on_step=True, on_epoch=True, logger=True)
            self.log("train/alignment_weight", alpha, on_epoch=True, logger=True)
        
        # Proper Lightning logging
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, logger=True)
        
        # Legacy logging
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pair_type = self.config.get('pair_type', 'MultiComponent')
        
        if pair_type == 'MultiComponent':
            return self._validate_multicomponent(batch)
        else:
            return self._validate_legacy(batch)
    
    def _validate_multicomponent(self, batch):
        """Validation with multicomponent data"""
        x1, x2 = batch['map1'], batch['map2']
        matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
        cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2'] 
        component1, component2 = batch['component1'], batch['component2']
        
        z1 = self(x1)
        z2 = self(x2)
        
        # Compute validation loss
        if isinstance(self.loss_fn, MultiLevelContrastiveLoss):
            loss, loss_components = self.loss_fn(
                z1, z2, matter_type1, matter_type2,
                cosmology1, cosmology2, component1, component2
            )
            # Log validation loss components
            self.log("val/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/matter_loss", loss_components['matter_loss'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("val/cosmology_loss", loss_components['cosmology_loss'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log("val/component_loss", loss_components['component_loss'], on_step=False, on_epoch=True, logger=True, sync_dist=True)
        else:
            loss = self.loss_fn(
                z1, z2, matter_type1, matter_type2,
                cosmology1, cosmology2, component1, component2
            )
            self.log("val/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Legacy logging for backward compatibility
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        
        # Store embeddings and labels for analysis
        self.val_latents_list.extend([z1.detach().cpu(), z2.detach().cpu()])
        self.val_matter_types_list.extend([matter_type1.detach().cpu(), matter_type2.detach().cpu()])
        self.val_cosmologies_list.extend([cosmology1.detach().cpu(), cosmology2.detach().cpu()])
        self.val_components_list.extend([component1.detach().cpu(), component2.detach().cpu()])
        
        # Generate soft scores using frozen classifier for compatibility
        with torch.no_grad():
            logits1 = self.classifier(x1)
            logits2 = self.classifier(x2)
            softscores1 = torch.sigmoid(logits1).squeeze()
            softscores2 = torch.sigmoid(logits2).squeeze()
            self.val_softscores_list.extend([softscores1.detach().cpu(), softscores2.detach().cpu()])
    
    def _validate_legacy(self, batch):
        """Legacy validation for backward compatibility"""
        if self.config.get('pair_type', 'CDMWDM') == 'CDMWDM':
            x1, x2, y1, y2 = batch
            z1 = self(x1)
            z2 = self(x2)
            loss = self.loss_fn(z1, z2, y1, y2)
            
            # Proper Lightning logging
            self.log("val/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)  # Legacy
            
            self.val_softscores_list.extend([y1.detach().cpu(), y2.detach().cpu()])
        else:  # SimCLR
            x1, x2 = batch
            z1 = self(x1)
            z2 = self(x2)
            loss_mix = self.loss_fn(z1, z2, None, None)
            
            # Proper Lightning logging
            self.log("val/total_loss", loss_mix, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_loss", loss_mix, prog_bar=True, sync_dist=True)  # Legacy
            
            with torch.no_grad():
                logits1 = self.classifier(x1)
                logits2 = self.classifier(x2)
                y1 = torch.sigmoid(logits1).squeeze()
                y2 = torch.sigmoid(logits2).squeeze()
                self.val_softscores_list.extend([y1.detach().cpu(), y2.detach().cpu()])
                
        self.val_latents_list.extend([z1.detach().cpu(), z2.detach().cpu()])
    
    def on_validation_epoch_end(self):
        if len(self.val_latents_list) == 0:
            return
            
        # Concatenate all validation data
        self.val_latents = torch.cat(self.val_latents_list, dim=0)
        self.val_softscores = torch.cat(self.val_softscores_list, dim=0)
        
        if self.config.get('pair_type') == 'MultiComponent':
            self.val_matter_types = torch.cat(self.val_matter_types_list, dim=0)
            self.val_cosmologies = torch.cat(self.val_cosmologies_list, dim=0) 
            self.val_components = torch.cat(self.val_components_list, dim=0)
        
        # Clear for next epoch
        self.val_latents_list.clear()
        self.val_softscores_list.clear()
        self.val_matter_types_list.clear()
        self.val_cosmologies_list.clear()
        self.val_components_list.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }

# Backward compatibility
class ContrastiveModel(HierarchicalContrastiveModel):
    """Backward compatibility wrapper"""
    def __init__(self, config):
        print("Using legacy ContrastiveModel interface")
        print("Consider migrating to HierarchicalContrastiveModel for full hierarchical support")
        super().__init__(config)