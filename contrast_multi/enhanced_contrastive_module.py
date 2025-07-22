# enhanced_contrastive_module.py - Module with cosmology focus (FIXED)

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch.nn.functional as F
from models import ContrastiveCNN, WDMClassifierTiny, WDMClassifierMedium, WDMClassifierLarge
from DM_coaching import CosmologyFocusedLoss, AdaptiveCosmologyWeights

class SimCLRNormalize(nn.Module):
    """L2‑normalise each sample along the feature dimension (SimCLR style)."""
    def __init__(self, dim: int = 1, eps: float = 1e-12):
        super().__init__()
        self.dim, self.eps = dim, eps
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class CosmologyEnhancedContrastiveModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Build base encoder
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
        
        # Load pretrained weights
        if config['pretrained_path'] is not None:
            checkpoint = torch.load(config["pretrained_path"], map_location="cpu", weights_only=True)
            state_dict = checkpoint["model_state_dict"]

            encoder.load_state_dict(state_dict, strict=False)
            self.classifier.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained weights from {config['pretrained_path']}")
        
        self.model = ContrastiveCNN(encoder)
        
        # DON'T create cosmology projector here - we'll create it dynamically
        self.cosmology_projector = None
        self._embedding_dim = None
        print(f"[INFO] Cosmology projector will be created dynamically on first forward pass")
        
        # Cosmology-focused loss
        self.loss_fn = CosmologyFocusedLoss(
            temperature=config.get('cosmology_temperature', 0.03),
            level_weights=config.get('initial_weights', [1.0, 0.5, 0.3]),
            pretrained_guidance_weight=config.get('pretrained_guidance_weight', 1.5),
            hard_negative_weight=config.get('hard_negative_weight', 2.0)
        )
        
        # Adaptive weight scheduler
        self.weight_scheduler = AdaptiveCosmologyWeights(
            initial_weights=config.get('initial_weights', [1.0, 0.5, 0.3]),
            target_weights=config.get('target_weights', [0.8, 4.0, 0.2]),
            warmup_epochs=config.get('cosmology_warmup_epochs', 15),
            focus_epochs=config.get('cosmology_focus_epochs', 50)
        )
        
        # Storage for validation
        self.val_latents_list = []
        self.val_cosmology_latents_list = []
        self.val_matter_types_list = []
        self.val_cosmologies_list = []
        self.val_components_list = []
        self.val_softscores_list = []
        
        self.config = config
        
        # Freeze classifier but keep it for guidance
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        # Track cosmology separation quality
        self.last_cosmology_silhouette = 0.0

    def forward(self, x):
        base_embedding = self.model(x)  # Gets actual embedding (128-dim)
        
        # Create cosmology projector on first forward pass with correct dimensions
        if self.cosmology_projector is None:
            self._embedding_dim = base_embedding.shape[1]
            self.cosmology_projector = nn.Sequential(
                nn.Linear(self._embedding_dim, max(32, self._embedding_dim // 2)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(max(32, self._embedding_dim // 2), 64),
                SimCLRNormalize(dim=1)
            ).to(base_embedding.device)
            print(f"[INFO] Created cosmology projector for embedding dim: {self._embedding_dim}")
        
        cosmology_embedding = self.cosmology_projector(base_embedding)
        
        return {
            'base': base_embedding,
            'cosmology': cosmology_embedding
        }

    def training_step(self, batch, batch_idx):
        """Enhanced training step with cosmology focus"""
        x1, x2 = batch['map1'], batch['map2']
        matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
        cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2'] 
        component1, component2 = batch['component1'], batch['component2']
        
        # Get embeddings
        out1 = self(x1)
        out2 = self(x2)
        z1, z2 = out1['base'], out2['base']
        
        # Get pretrained CNN guidance
        with torch.no_grad():
            logits1 = self.classifier(x1)
            logits2 = self.classifier(x2)
            pretrained_scores1 = torch.sigmoid(logits1).squeeze()  # 0=CDM, 1=WDM
            pretrained_scores2 = torch.sigmoid(logits2).squeeze()
        
        # Update loss weights based on epoch and cosmology quality
        current_weights = self.weight_scheduler.get_weights(
            self.current_epoch, 
            self.last_cosmology_silhouette
        )
        self.loss_fn.level_weights = current_weights
        
        # Compute enhanced loss with pretrained guidance
        loss, loss_components = self.loss_fn(
            z1, z2, matter_type1, matter_type2,
            cosmology1, cosmology2, component1, component2,
            pretrained_scores1, pretrained_scores2
        )
        
        # Additional cosmology-specific contrastive loss
        cosmology_specific_loss = self._compute_cosmology_specific_loss(
            out1['cosmology'], out2['cosmology'], 
            cosmology1, cosmology2, matter_type1, matter_type2
        )
        
        total_loss = loss + 0.5 * cosmology_specific_loss
        
        # Comprehensive logging
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/base_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train/cosmology_specific_loss", cosmology_specific_loss, on_step=True, on_epoch=True, logger=True)
        
        # Log loss components
        for component_name, component_loss in loss_components.items():
            self.log(f"train/{component_name}", component_loss, on_step=True, on_epoch=True, logger=True)
        
        # Log adaptive weights
        for i, weight in enumerate(current_weights):
            level_names = ['matter', 'cosmology', 'component']
            self.log(f"train/weight_{level_names[i]}", weight, on_epoch=True, logger=True)
        
        # Log pretrained guidance alignment
        cosine_sim = F.cosine_similarity(
            torch.cat([pretrained_scores1, pretrained_scores2]).unsqueeze(1),
            torch.cat([cosmology1, cosmology2]).float().unsqueeze(1)
        ).mean()
        self.log("train/pretrained_alignment", cosine_sim, on_epoch=True, logger=True)
        
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, logger=True)
        
        return total_loss
    
    def _compute_cosmology_specific_loss(self, cosmo_z1, cosmo_z2, cosmology1, cosmology2, 
                                       matter_type1, matter_type2):
        """Specialized contrastive loss for cosmology embeddings"""
        # Only compute for samples with same matter type (to focus on cosmology differences)
        same_matter_mask = matter_type1 == matter_type2
        
        if not same_matter_mask.any():
            return torch.tensor(0.0, device=cosmo_z1.device)
        
        # Filter to same matter type pairs
        cosmo_z1_filt = cosmo_z1[same_matter_mask]
        cosmo_z2_filt = cosmo_z2[same_matter_mask]
        cosmology1_filt = cosmology1[same_matter_mask]
        cosmology2_filt = cosmology2[same_matter_mask]
        
        # Compute similarities
        sim_matrix = torch.matmul(cosmo_z1_filt, cosmo_z2_filt.T) / 0.03  # Low temperature
        
        # Create targets: high similarity for same cosmology, low for different
        cosmo_targets = (cosmology1_filt.unsqueeze(1) == cosmology2_filt.unsqueeze(0)).float()
        
        # InfoNCE-style loss
        labels = torch.arange(len(cosmo_z1_filt), device=cosmo_z1_filt.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Enhanced validation with cosmology tracking"""
        x1, x2 = batch['map1'], batch['map2']
        matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
        cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2'] 
        component1, component2 = batch['component1'], batch['component2']
        
        out1 = self(x1)
        out2 = self(x2)
        z1, z2 = out1['base'], out2['base']
        
        # Get pretrained scores
        with torch.no_grad():
            logits1 = self.classifier(x1)
            logits2 = self.classifier(x2)
            pretrained_scores1 = torch.sigmoid(logits1).squeeze()
            pretrained_scores2 = torch.sigmoid(logits2).squeeze()
        
        # Compute validation loss
        current_weights = self.weight_scheduler.get_weights(self.current_epoch, self.last_cosmology_silhouette)
        self.loss_fn.level_weights = current_weights
        
        loss, loss_components = self.loss_fn(
            z1, z2, matter_type1, matter_type2,
            cosmology1, cosmology2, component1, component2,
            pretrained_scores1, pretrained_scores2
        )
        
        cosmology_specific_loss = self._compute_cosmology_specific_loss(
            out1['cosmology'], out2['cosmology'],
            cosmology1, cosmology2, matter_type1, matter_type2
        )
        
        total_loss = loss + 0.5 * cosmology_specific_loss
        
        # Log validation metrics
        self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/base_loss", loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("val/cosmology_specific_loss", cosmology_specific_loss, on_epoch=True, logger=True, sync_dist=True)
        
        for component_name, component_loss in loss_components.items():
            self.log(f"val/{component_name}", component_loss, on_epoch=True, logger=True, sync_dist=True)
        
        # Store for analysis
        self.val_latents_list.extend([z1.detach().cpu(), z2.detach().cpu()])
        self.val_cosmology_latents_list.extend([out1['cosmology'].detach().cpu(), out2['cosmology'].detach().cpu()])
        self.val_matter_types_list.extend([matter_type1.detach().cpu(), matter_type2.detach().cpu()])
        self.val_cosmologies_list.extend([cosmology1.detach().cpu(), cosmology2.detach().cpu()])
        self.val_components_list.extend([component1.detach().cpu(), component2.detach().cpu()])
        self.val_softscores_list.extend([pretrained_scores1.detach().cpu(), pretrained_scores2.detach().cpu()])

    def on_validation_epoch_end(self):
        if len(self.val_latents_list) == 0:
            return
            
        # Concatenate validation data
        self.val_latents = torch.cat(self.val_latents_list, dim=0)
        self.val_cosmology_latents = torch.cat(self.val_cosmology_latents_list, dim=0)
        self.val_softscores = torch.cat(self.val_softscores_list, dim=0)
        self.val_matter_types = torch.cat(self.val_matter_types_list, dim=0)
        self.val_cosmologies = torch.cat(self.val_cosmologies_list, dim=0) 
        self.val_components = torch.cat(self.val_components_list, dim=0)
        
        # Compute cosmology separation quality for adaptive weights
        try:
            from sklearn.metrics import silhouette_score
            cosmology_silhouette = silhouette_score(
                self.val_cosmology_latents.numpy(), 
                self.val_cosmologies.numpy()
            )
            self.last_cosmology_silhouette = cosmology_silhouette
            self.log("val/cosmology_silhouette", cosmology_silhouette, logger=True)
        except Exception:
            pass
        
        # Clear for next epoch
        self.val_latents_list.clear()
        self.val_cosmology_latents_list.clear()
        self.val_softscores_list.clear()
        self.val_matter_types_list.clear()
        self.val_cosmologies_list.clear()
        self.val_components_list.clear()

    def configure_optimizers(self):
        # Get all trainable parameters
        all_params = list(self.model.parameters())
        
        # Add cosmology projector parameters if it exists
        if self.cosmology_projector is not None:
            all_params.extend(list(self.cosmology_projector.parameters()))
        
        optimizer = torch.optim.Adam(
            all_params,
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
                "monitor": "train/total_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }