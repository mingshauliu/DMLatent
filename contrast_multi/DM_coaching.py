import torch
import torch.nn as nn
import torch.nn.functional as F
from hierarchical_losses import MultiLevelContrastiveLoss

class CosmologyFocusedLoss(nn.Module):
    """
    Enhanced loss that uses pretrained CNN guidance to improve cosmology separation
    """
    def __init__(self, temperature=0.05, level_weights=[1.0, 2.0, 0.3], 
                 pretrained_guidance_weight=1.0, hard_negative_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.level_weights = level_weights  # [matter, cosmology, component]
        self.pretrained_guidance_weight = pretrained_guidance_weight
        self.hard_negative_weight = hard_negative_weight
        
        # Base multilevel loss
        self.base_loss = MultiLevelContrastiveLoss(temperature, level_weights)
        
    def forward(self, z1, z2, matter_type1, matter_type2, cosmology1, cosmology2, 
                component1, component2, pretrained_scores1=None, pretrained_scores2=None):
        """
        Enhanced loss with pretrained CNN guidance for cosmology separation
        
        Args:
            pretrained_scores1/2: [B] - CDM/WDM scores from pretrained classifier (0=CDM, 1=WDM)
        """
        # Base hierarchical loss
        base_loss, loss_components = self.base_loss(
            z1, z2, matter_type1, matter_type2, cosmology1, cosmology2, component1, component2
        )
        
        total_loss = base_loss
        
        # Add pretrained guidance if available
        if pretrained_scores1 is not None and pretrained_scores2 is not None:
            guidance_loss = self._compute_pretrained_guidance_loss(
                z1, z2, pretrained_scores1, pretrained_scores2
            )
            total_loss += self.pretrained_guidance_weight * guidance_loss
            loss_components['pretrained_guidance'] = guidance_loss
        
        # Add hard negative mining for cosmology pairs
        hard_negative_loss = self._compute_hard_negative_cosmology_loss(
            z1, z2, cosmology1, cosmology2, matter_type1, matter_type2
        )
        total_loss += self.hard_negative_weight * hard_negative_loss
        loss_components['hard_negative_cosmology'] = hard_negative_loss
        
        return total_loss, loss_components
    
    def _compute_pretrained_guidance_loss(self, z1, z2, scores1, scores2):
        """Use pretrained CNN to guide cosmology separation in embedding space"""
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Compute similarities in embedding space
        z_all = torch.cat([z1_norm, z2_norm], dim=0)  # [2B, D]
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature  # [2B, 2B]
        
        # Create target similarities based on pretrained scores
        scores_all = torch.cat([scores1, scores2], dim=0)  # [2B]
        
        # Cosine similarity in score space (higher for similar cosmologies)
        score_sim_matrix = 1 - torch.abs(scores_all.unsqueeze(0) - scores_all.unsqueeze(1))
        
        # Encourage embedding similarities to match score similarities
        # Focus on cosmology pairs by masking
        guidance_loss = F.mse_loss(torch.sigmoid(sim_matrix), score_sim_matrix)
        
        return guidance_loss
    
    def _compute_hard_negative_cosmology_loss(self, z1, z2, cosmology1, cosmology2, 
                                            matter_type1, matter_type2):
        """Focus on hard negatives: same matter type, different cosmology"""
        batch_size = z1.shape[0]
        device = z1.device
        
        # Find hard negative pairs: same matter type, different cosmology
        same_matter = (matter_type1.unsqueeze(1) == matter_type1.unsqueeze(0)) & \
                     (matter_type2.unsqueeze(1) == matter_type2.unsqueeze(0))
        diff_cosmology = (cosmology1.unsqueeze(1) != cosmology1.unsqueeze(0)) | \
                        (cosmology2.unsqueeze(1) != cosmology2.unsqueeze(0))
        
        hard_negative_mask = same_matter & diff_cosmology
        
        if not hard_negative_mask.any():
            return torch.tensor(0.0, device=device)
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Compute similarities
        sim11 = torch.matmul(z1_norm, z1_norm.T) / self.temperature
        sim22 = torch.matmul(z2_norm, z2_norm.T) / self.temperature
        sim12 = torch.matmul(z1_norm, z2_norm.T) / self.temperature
        sim21 = torch.matmul(z2_norm, z1_norm.T) / self.temperature
        
        # Push apart hard negatives
        hard_neg_loss = (
            F.relu(0.5 + sim11[hard_negative_mask]).mean() +
            F.relu(0.5 + sim22[hard_negative_mask]).mean() +
            F.relu(0.5 + sim12[hard_negative_mask]).mean() + 
            F.relu(0.5 + sim21[hard_negative_mask]).mean()
        ) / 4
        
        return hard_neg_loss

class AdaptiveCosmologyWeights:
    """
    Adaptive weight scheduler specifically designed to improve cosmology separation
    """
    def __init__(self, initial_weights=[1.0, 0.5, 0.3], target_weights=[0.8, 3.0, 0.2], 
                 warmup_epochs=20, focus_epochs=40):
        self.initial_weights = initial_weights
        self.target_weights = target_weights  
        self.warmup_epochs = warmup_epochs
        self.focus_epochs = focus_epochs
        
    def get_weights(self, epoch, cosmology_silhouette=None):
        """
        Get adaptive weights based on training progress and cosmology separation quality
        
        Strategy:
        1. Epochs 0-20: Normal hierarchy learning
        2. Epochs 20-60: Boost cosmology weight significantly  
        3. Epochs 60+: Maintain high cosmology weight, add back others
        """
        if epoch < self.warmup_epochs:
            # Phase 1: Normal hierarchical learning
            progress = epoch / self.warmup_epochs
            weights = [
                self.initial_weights[0],
                self.initial_weights[1] + progress * (self.target_weights[1] - self.initial_weights[1]) * 0.3,
                self.initial_weights[2]
            ]
        elif epoch < self.warmup_epochs + self.focus_epochs:
            # Phase 2: Focus heavily on cosmology
            focus_progress = (epoch - self.warmup_epochs) / self.focus_epochs
            
            # Boost cosmology weight significantly
            cosmology_boost = 1.0 + 2.0 * focus_progress  # Up to 3x boost
            weights = [
                self.initial_weights[0] * (1 - 0.3 * focus_progress),  # Reduce matter weight slightly
                self.target_weights[1] * cosmology_boost,              # Boost cosmology heavily
                self.initial_weights[2] * (1 - 0.5 * focus_progress)   # Reduce component weight
            ]
        else:
            # Phase 3: Maintain cosmology focus but rebalance
            weights = [
                self.target_weights[0],
                self.target_weights[1],  # Keep high cosmology weight
                self.target_weights[2]
            ]
        
        # Adaptive adjustment based on cosmology separation quality
        if cosmology_silhouette is not None:
            if cosmology_silhouette < 0.2:  # Poor separation
                weights[1] *= 1.5  # Increase cosmology weight more
            elif cosmology_silhouette > 0.4:  # Good separation
                weights[1] *= 0.8  # Can reduce cosmology weight slightly
        
        return weights

class CosmologyContrastiveModel(nn.Module):
    """
    Enhanced model that includes cosmology-specific components
    """
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Add cosmology-specific projection head
        embedding_dim = base_model.model.encoder.out_features if hasattr(base_model.model.encoder, 'out_features') else 512
        self.cosmology_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 64),  # Smaller space for cosmology
            nn.L2Norm(dim=1)
        )
        
        # Cosmology-specific loss
        self.cosmology_loss = CosmologyFocusedLoss(
            temperature=config.get('cosmology_temperature', 0.03),  # Lower temp for harder separation
            level_weights=config.get('level_weights', [1.0, 2.0, 0.3]),
            pretrained_guidance_weight=config.get('pretrained_guidance_weight', 1.0),
            hard_negative_weight=config.get('hard_negative_weight', 2.0)
        )
        
        # Adaptive weight scheduler
        self.weight_scheduler = AdaptiveCosmologyWeights(
            initial_weights=config.get('initial_weights', [1.0, 0.5, 0.3]),
            target_weights=config.get('target_weights', [0.8, 3.0, 0.2]),
            warmup_epochs=config.get('cosmology_warmup_epochs', 20),
            focus_epochs=config.get('cosmology_focus_epochs', 40)
        )
    
    def forward(self, x):
        # Get base embedding
        base_embedding = self.base_model(x)
        
        # Get cosmology-specific embedding
        cosmology_embedding = self.cosmology_projector(base_embedding)
        
        return {
            'base': base_embedding,
            'cosmology': cosmology_embedding
        }
