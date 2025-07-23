import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch.nn.functional as F
from models import WDMClassifierTiny, WDMClassifierMedium, WDMClassifierLarge
from DM_coaching import CosmologyFocusedLoss, AdaptiveCosmologyWeights

class FrozenEncoderContrastiveModel(pl.LightningModule):
    """
    Version 1: Completely freeze the encoder, only train projection heads
    Best for preserving cosmology knowledge while learning new representations
    """
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
        
        # FREEZE THE ENCODER - this is the key change!
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()  # Set to eval mode
        print(f"[INFO] FROZEN ENCODER - all encoder weights are frozen!")
        
        # Create frozen encoder wrapper that extracts features
        self.frozen_encoder = encoder
        
        # Get feature dimension from frozen encoder
        self.feature_extractor = self._create_feature_extractor()
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, config['img_size'], config['img_size'])
            dummy_features = self.feature_extractor(dummy_input)
            feature_dim = dummy_features.shape[1]
        
        print(f"[INFO] Extracted feature dimension: {feature_dim}")
        
        # Create separate projection heads for different tasks
        self.contrastive_projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),  # Standard contrastive embedding
            nn.ReLU()
        )
        
        self.cosmology_projector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(128, 64),   # Specialized cosmology embedding
        )
        
        # Matter type projector for hierarchical learning
        self.matter_projector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(128, 32),   # Matter type embedding
        )
        
        # Losses
        self.loss_fn = CosmologyFocusedLoss(
            temperature=config.get('cosmology_temperature', 0.03),
            level_weights=config.get('initial_weights', [1.0, 0.5, 0.3]),
            pretrained_guidance_weight=config.get('pretrained_guidance_weight', 2.0),
            hard_negative_weight=config.get('hard_negative_weight', 3.0)
        )
        
        self.weight_scheduler = AdaptiveCosmologyWeights(
            initial_weights=config.get('initial_weights', [1.0, 0.5, 0.3]),
            target_weights=config.get('target_weights', [0.5, 6.0, 0.1]),  # Even higher cosmology weight
            warmup_epochs=config.get('cosmology_warmup_epochs', 5),        # Faster warmup
            focus_epochs=config.get('cosmology_focus_epochs', 40)
        )
        
        # Storage
        self.val_latents_list = []
        self.val_cosmology_latents_list = []
        self.val_matter_latents_list = []
        self.val_matter_types_list = []
        self.val_cosmologies_list = []
        self.val_components_list = []
        self.val_softscores_list = []
        
        self.config = config
        
        # Freeze classifier for guidance
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        self.last_cosmology_silhouette = 0.0
    
    def _create_feature_extractor(self):
        """Extract features from the frozen encoder (before final classification layer)"""
        # For most CNN architectures, we want features before the final classifier
        if hasattr(self.frozen_encoder, 'features'):
            # If encoder has explicit features module
            return nn.Sequential(
                self.frozen_encoder.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        elif hasattr(self.frozen_encoder, 'classifier'):
            # Remove the final classification layer
            modules = list(self.frozen_encoder.children())[:-1]
            return nn.Sequential(
                *modules,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        else:
            # Fallback: use everything except potentially the last layer
            return self.frozen_encoder
    
    def forward(self, x):
        # Extract frozen features
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        # Project to different embedding spaces
        contrastive_emb = self.contrastive_projector(features)
        cosmology_emb = F.normalize(self.cosmology_projector(features), p=2, dim=1)
        matter_emb = F.normalize(self.matter_projector(features), p=2, dim=1)
        
        return {
            'features': features,
            'contrastive': contrastive_emb, 
            'cosmology': cosmology_emb,
            'matter': matter_emb
        }
    
    def training_step(self, batch, batch_idx):
        x1, x2 = batch['map1'], batch['map2']
        matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
        cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2'] 
        component1, component2 = batch['component1'], batch['component2']
        
        # Get embeddings
        out1 = self(x1)
        out2 = self(x2)
        
        # Get pretrained cosmology scores (should be very accurate since encoder is frozen!)
        with torch.no_grad():
            logits1 = self.classifier(x1)
            logits2 = self.classifier(x2)
            pretrained_scores1 = torch.sigmoid(logits1).squeeze()
            pretrained_scores2 = torch.sigmoid(logits2).squeeze()
        
        # Update weights
        current_weights = self.weight_scheduler.get_weights(
            self.current_epoch, self.last_cosmology_silhouette
        )
        self.loss_fn.level_weights = current_weights
        
        # Multi-level contrastive losses
        # 1. General contrastive loss on main embedding
        contrastive_loss, loss_components = self.loss_fn(
            out1['contrastive'], out2['contrastive'], 
            matter_type1, matter_type2, cosmology1, cosmology2, 
            component1, component2, pretrained_scores1, pretrained_scores2
        )
        
        # 2. Specialized cosmology loss - should be easier now with frozen encoder!
        cosmology_loss = self._compute_specialized_cosmology_loss(
            out1['cosmology'], out2['cosmology'], cosmology1, cosmology2, 
            pretrained_scores1, pretrained_scores2
        )
        
        # 3. Matter type separation loss
        matter_loss = self._compute_matter_separation_loss(
            out1['matter'], out2['matter'], matter_type1, matter_type2
        )
        
        # Combine losses with emphasis on cosmology since that's what we're trying to fix
        total_loss = (
            0.3 * contrastive_loss +     # Reduce general contrastive
            0.5 * cosmology_loss +       # BOOST cosmology loss
            0.2 * matter_loss            # Keep some matter separation
        )
        
        # Logging
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", contrastive_loss, on_step=True, on_epoch=True)
        self.log("train/cosmology_loss", cosmology_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/matter_loss", matter_loss, on_step=True, on_epoch=True)
        
        # Log pretrained accuracy (should stay high since encoder is frozen)
        cosmology_accuracy = ((pretrained_scores1 > 0.5) == cosmology1.bool()).float().mean()
        self.log("train/pretrained_cosmology_accuracy", cosmology_accuracy, on_epoch=True)
        
        for i, weight in enumerate(current_weights):
            level_names = ['matter', 'cosmology', 'component']
            self.log(f"train/weight_{level_names[i]}", weight, on_epoch=True)
        
        return total_loss
    
    def _compute_specialized_cosmology_loss(self, cosmo_emb1, cosmo_emb2, cosmology1, cosmology2, 
                                          pretrained_scores1, pretrained_scores2):
        """Specialized loss that leverages frozen encoder's cosmology knowledge"""
        batch_size = cosmo_emb1.shape[0]
        
        # Use pretrained scores to create strong supervision signal
        # Score difference should correlate with embedding distance
        score_diff = torch.abs(pretrained_scores1.unsqueeze(1) - pretrained_scores2.unsqueeze(0))
        embedding_dist = torch.cdist(cosmo_emb1, cosmo_emb2, p=2)
        
        # Encourage: similar scores → similar embeddings, different scores → distant embeddings
        target_dist = score_diff * 2.0  # Scale target distances
        distance_loss = F.mse_loss(embedding_dist, target_dist)
        
        # Standard contrastive loss for same/different cosmology
        same_cosmology = (cosmology1.unsqueeze(1) == cosmology2.unsqueeze(0)).float()
        sim_matrix = torch.matmul(cosmo_emb1, cosmo_emb2.T) / 0.03
        
        # Push apart different cosmologies, pull together same cosmologies
        contrastive_cosmology = -(same_cosmology * sim_matrix - (1 - same_cosmology) * sim_matrix).mean()
        
        return distance_loss + contrastive_cosmology
    
    def _compute_matter_separation_loss(self, matter_emb1, matter_emb2, matter_type1, matter_type2):
        """Simple contrastive loss for matter type separation"""
        same_matter = (matter_type1.unsqueeze(1) == matter_type2.unsqueeze(0)).float()
        sim_matrix = torch.matmul(matter_emb1, matter_emb2.T) / 0.1
        
        # InfoNCE-style loss for matter types
        positive_mask = same_matter
        negative_mask = 1 - same_matter
        
        positives = (sim_matrix * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        negatives = torch.logsumexp(sim_matrix * negative_mask, dim=1)
        
        return (-positives + negatives).mean()
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = batch['map1'], batch['map2']
        matter_type1, matter_type2 = batch['matter_type1'], batch['matter_type2']
        cosmology1, cosmology2 = batch['cosmology1'], batch['cosmology2'] 
        component1, component2 = batch['component1'], batch['component2']
        
        out1 = self(x1)
        out2 = self(x2)
        
        with torch.no_grad():
            logits1 = self.classifier(x1)
            logits2 = self.classifier(x2)
            pretrained_scores1 = torch.sigmoid(logits1).squeeze()
            pretrained_scores2 = torch.sigmoid(logits2).squeeze()
        
        # Compute validation losses (same as training)
        current_weights = self.weight_scheduler.get_weights(self.current_epoch, self.last_cosmology_silhouette)
        self.loss_fn.level_weights = current_weights
        
        contrastive_loss, _ = self.loss_fn(
            out1['contrastive'], out2['contrastive'], 
            matter_type1, matter_type2, cosmology1, cosmology2, 
            component1, component2, pretrained_scores1, pretrained_scores2
        )
        
        cosmology_loss = self._compute_specialized_cosmology_loss(
            out1['cosmology'], out2['cosmology'], cosmology1, cosmology2,
            pretrained_scores1, pretrained_scores2
        )
        
        matter_loss = self._compute_matter_separation_loss(
            out1['matter'], out2['matter'], matter_type1, matter_type2
        )
        
        total_loss = 0.3 * contrastive_loss + 0.5 * cosmology_loss + 0.2 * matter_loss
        
        self.log("val/total_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cosmology_loss", cosmology_loss, on_epoch=True, sync_dist=True)
        
        # Validation cosmology accuracy (should stay high!)
        cosmology_accuracy = ((pretrained_scores1 > 0.5) == cosmology1.bool()).float().mean()
        self.log("val/pretrained_cosmology_accuracy", cosmology_accuracy, on_epoch=True, sync_dist=True)
        
        # Store for UMAP
        self.val_latents_list.extend([out1['contrastive'].detach().cpu(), out2['contrastive'].detach().cpu()])
        self.val_cosmology_latents_list.extend([out1['cosmology'].detach().cpu(), out2['cosmology'].detach().cpu()])
        self.val_matter_latents_list.extend([out1['matter'].detach().cpu(), out2['matter'].detach().cpu()])
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
        self.val_matter_latents = torch.cat(self.val_matter_latents_list, dim=0)
        self.val_softscores = torch.cat(self.val_softscores_list, dim=0)
        self.val_matter_types = torch.cat(self.val_matter_types_list, dim=0)
        self.val_cosmologies = torch.cat(self.val_cosmologies_list, dim=0) 
        self.val_components = torch.cat(self.val_components_list, dim=0)
        
        # Compute separation metrics for all embedding spaces
        try:
            from sklearn.metrics import silhouette_score
            
            # Cosmology separation in specialized embedding
            cosmology_silhouette = silhouette_score(
                self.val_cosmology_latents.numpy(), 
                self.val_cosmologies.numpy()
            )
            
            # Matter type separation 
            matter_silhouette = silhouette_score(
                self.val_matter_latents.numpy(),
                self.val_matter_types.numpy()
            )
            
            self.last_cosmology_silhouette = cosmology_silhouette
            self.log("val/cosmology_silhouette", cosmology_silhouette, logger=True)
            self.log("val/matter_silhouette", matter_silhouette, logger=True)
            
            print(f"\nEpoch {self.current_epoch} - FROZEN ENCODER Results:")
            print(f"  Cosmology silhouette: {cosmology_silhouette:.3f} (target > 0.3)")
            print(f"  Matter silhouette: {matter_silhouette:.3f}")
            
        except Exception as e:
            print(f"Could not compute silhouette scores: {e}")
        
        # Clear for next epoch
        self.val_latents_list.clear()
        self.val_cosmology_latents_list.clear()
        self.val_matter_latents_list.clear()
        self.val_softscores_list.clear()
        self.val_matter_types_list.clear()
        self.val_cosmologies_list.clear()
        self.val_components_list.clear()

    def configure_optimizers(self):
        # Only optimize the projection heads - encoder is frozen!
        trainable_params = (
            list(self.contrastive_projector.parameters()) +
            list(self.cosmology_projector.parameters()) +
            list(self.matter_projector.parameters())
        )
        
        print(f"[INFO] Optimizing {len(trainable_params)} projection head parameters only")
        print(f"[INFO] Encoder parameters are FROZEN")
        
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing - can be more aggressive since we're only training projectors
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/cosmology_silhouette",  # Monitor cosmology separation directly
                "interval": "epoch",
                "frequency": 1,
            }
        }