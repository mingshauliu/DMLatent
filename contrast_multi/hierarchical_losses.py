import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalNTXentLoss(nn.Module):
    """
    Hierarchical NT-Xent loss that encourages:
    1. Matter type separation (strongest)
    2. Cosmology separation within matter types (medium)  
    3. Component separation within baryonic matter (weakest)
    """
    def __init__(self, temperature=0.05, matter_weight=1.0, cosmology_weight=0.5, component_weight=0.25):
        super().__init__()
        self.temperature = temperature
        self.matter_weight = matter_weight
        self.cosmology_weight = cosmology_weight
        self.component_weight = component_weight
        
    def forward(self, z1, z2, matter_type1, matter_type2, cosmology1, cosmology2, component1, component2):
        """
        Args:
            z1, z2: [B, D] - embeddings for two views
            matter_type1, matter_type2: [B] - matter type labels (0=baryonic, 1=DM, 2=total)
            cosmology1, cosmology2: [B] - cosmology labels (0=CDM, 1=WDM)
            component1, component2: [B] - component labels (0=gas/DM, 1=stars)
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Concatenate all embeddings
        z_all = torch.cat([z1_norm, z2_norm], dim=0)  # [2B, D]
        
        # Create similarity matrix
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature  # [2B, 2B]
        
        # Create hierarchical similarity targets
        hierarchy_sim = self._compute_hierarchical_similarity(
            matter_type1, matter_type2, cosmology1, cosmology2, component1, component2, device
        )
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_indices = []
        for i in range(batch_size):
            pos_indices.append([i, i + batch_size])
            pos_indices.append([i + batch_size, i])
        
        # Compute loss with hierarchical weighting
        total_loss = 0.0
        count = 0
        
        for i, j in pos_indices:
            # Get positive similarity (between paired samples)
            pos_sim = sim_matrix[i, j]
            
            # Get all negative similarities for anchor i
            neg_mask = torch.ones(2 * batch_size, device=device).bool()
            neg_mask[i] = False  # Remove self
            neg_mask[j] = False  # Remove positive pair
            neg_sims = sim_matrix[i][neg_mask]
            
            # Weight negative similarities based on hierarchy
            neg_weights = self._get_negative_weights(i, j, hierarchy_sim, batch_size)
            weighted_neg_sims = neg_sims * neg_weights
            
            # Compute InfoNCE loss
            all_sims = torch.cat([pos_sim.unsqueeze(0), weighted_neg_sims])
            loss = -torch.log_softmax(all_sims, dim=0)[0]
            
            total_loss += loss
            count += 1
            
        return total_loss / count
    
    def _compute_hierarchical_similarity(self, matter_type1, matter_type2, cosmology1, cosmology2, 
                                       component1, component2, device):
        """Compute hierarchical similarity weights between all pairs"""
        batch_size = matter_type1.shape[0]
        
        # Stack all labels
        matter_all = torch.cat([matter_type1, matter_type2], dim=0)  # [2B]
        cosmology_all = torch.cat([cosmology1, cosmology2], dim=0)  # [2B]
        component_all = torch.cat([component1, component2], dim=0)  # [2B]
        
        # Compute pairwise similarities for each hierarchy level
        matter_sim = (matter_all.unsqueeze(0) == matter_all.unsqueeze(1)).float()
        cosmology_sim = (cosmology_all.unsqueeze(0) == cosmology_all.unsqueeze(1)).float()
        component_sim = (component_all.unsqueeze(0) == component_all.unsqueeze(1)).float()
        
        # Combine hierarchically: matter type is most important
        hierarchy_sim = (
            self.matter_weight * matter_sim + 
            self.cosmology_weight * cosmology_sim * matter_sim +  # Only count cosmology if matter type matches
            self.component_weight * component_sim * matter_sim    # Only count component if matter type matches
        )
        
        return hierarchy_sim
    
    def _get_negative_weights(self, anchor_idx, pos_idx, hierarchy_sim, batch_size):
        """Get weights for negative samples based on hierarchical distance"""
        # Get similarity scores between anchor and all others (excluding anchor and positive)
        anchor_sims = hierarchy_sim[anchor_idx]
        
        # Remove anchor and positive pair
        neg_mask = torch.ones_like(anchor_sims).bool()
        neg_mask[anchor_idx] = False
        neg_mask[pos_idx] = False
        
        neg_sims = anchor_sims[neg_mask]
        
        # Convert similarity to dissimilarity weights (higher weight for more dissimilar)
        max_sim = neg_sims.max()
        weights = max_sim - neg_sims + 0.1  # Add small constant to avoid zero weights
        
        return weights

class AdaptiveHierarchicalLoss(nn.Module):
    """
    Adaptive version that adjusts hierarchy weights based on training progress
    """
    def __init__(self, temperature=0.05, base_matter_weight=2.0, base_cosmology_weight=1.0, 
                 base_component_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.base_matter_weight = base_matter_weight
        self.base_cosmology_weight = base_cosmology_weight
        self.base_component_weight = base_component_weight
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """Update weights based on training progress"""
        self.epoch = epoch
        
        # Start with strong matter type separation, gradually add finer distinctions
        progress = min(epoch / 50.0, 1.0)  # Normalize by expected epochs
        
        self.matter_weight = self.base_matter_weight
        self.cosmology_weight = self.base_cosmology_weight * progress
        self.component_weight = self.base_component_weight * (progress ** 2)  # Add component separation later
        
    def forward(self, z1, z2, matter_type1, matter_type2, cosmology1, cosmology2, component1, component2):
        """Same as HierarchicalNTXentLoss but with adaptive weights"""
        # Use the same logic but with updated weights
        loss_fn = HierarchicalNTXentLoss(
            temperature=self.temperature,
            matter_weight=self.matter_weight,
            cosmology_weight=self.cosmology_weight,
            component_weight=self.component_weight
        )
        return loss_fn(z1, z2, matter_type1, matter_type2, cosmology1, cosmology2, component1, component2)

class MultiLevelContrastiveLoss(nn.Module):
    """
    Multi-level approach: separate losses for each hierarchy level
    """
    def __init__(self, temperature=0.05, level_weights=[1.0, 0.7, 0.3]):
        super().__init__()
        self.temperature = temperature
        self.level_weights = level_weights  # [matter, cosmology, component]
        
    def forward(self, z1, z2, matter_type1, matter_type2, cosmology1, cosmology2, component1, component2):
        """Compute separate contrastive losses for each hierarchy level"""
        
        # Level 1: Matter type separation
        matter_loss = self._compute_level_loss(z1, z2, matter_type1, matter_type2)
        
        # Level 2: Cosmology separation (only within same matter type)
        cosmology_loss = self._compute_conditional_loss(
            z1, z2, cosmology1, cosmology2, matter_type1, matter_type2
        )
        
        # Level 3: Component separation (only within baryonic matter)
        baryonic_mask = (matter_type1 == 0) & (matter_type2 == 0)
        if baryonic_mask.any():
            component_loss = self._compute_masked_loss(
                z1, z2, component1, component2, baryonic_mask
            )
        else:
            component_loss = torch.tensor(0.0, device=z1.device)
        
        # Weighted combination
        total_loss = (
            self.level_weights[0] * matter_loss +
            self.level_weights[1] * cosmology_loss +
            self.level_weights[2] * component_loss
        )
        
        return total_loss, {
            'matter_loss': matter_loss,
            'cosmology_loss': cosmology_loss, 
            'component_loss': component_loss
        }
    
    def _compute_level_loss(self, z1, z2, labels1, labels2):
        """Compute NT-Xent loss for a specific label level"""
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # Concatenate
        z_all = torch.cat([z1_norm, z2_norm], dim=0)
        labels_all = torch.cat([labels1, labels2], dim=0)
        
        # Similarity matrix
        sim_matrix = torch.matmul(z_all, z_all.T) / self.temperature
        
        # Create positive/negative mask based on labels
        labels_eq = labels_all.unsqueeze(0) == labels_all.unsqueeze(1)
        
        # Mask diagonal
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute loss
        pos_sim = torch.diag(sim_matrix, batch_size)  # Positive pairs
        
        # For each positive pair, compute InfoNCE
        total_loss = 0.0
        for i in range(batch_size):
            # Positive similarity
            pos_s = pos_sim[i]
            
            # All similarities for anchor i (excluding self)
            anchor_sims = sim_matrix[i]
            anchor_sims = anchor_sims[~mask[i]]  # Remove self
            
            # InfoNCE loss
            all_sims = torch.cat([pos_s.unsqueeze(0), anchor_sims])
            loss = -torch.log_softmax(all_sims, dim=0)[0]
            total_loss += loss
            
        return total_loss / batch_size
    
    def _compute_conditional_loss(self, z1, z2, labels1, labels2, condition1, condition2):
        """Compute loss only for samples where condition matches"""
        condition_match = condition1 == condition2
        if not condition_match.any():
            return torch.tensor(0.0, device=z1.device)
            
        # Filter to matching conditions
        z1_filt = z1[condition_match]
        z2_filt = z2[condition_match]
        labels1_filt = labels1[condition_match]
        labels2_filt = labels2[condition_match]
        
        if len(z1_filt) == 0:
            return torch.tensor(0.0, device=z1.device)
            
        return self._compute_level_loss(z1_filt, z2_filt, labels1_filt, labels2_filt)
    
    def _compute_masked_loss(self, z1, z2, labels1, labels2, mask):
        """Compute loss only for masked samples"""
        if not mask.any():
            return torch.tensor(0.0, device=z1.device)
            
        z1_masked = z1[mask]
        z2_masked = z2[mask]
        labels1_masked = labels1[mask]
        labels2_masked = labels2[mask]
        
        return self._compute_level_loss(z1_masked, z2_masked, labels1_masked, labels2_masked)