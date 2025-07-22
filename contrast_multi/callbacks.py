import torch
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import numpy as np

class HierarchicalUMAPCallback(Callback):
    """Enhanced UMAP callback that visualizes hierarchical structure"""
    
    def __init__(self, every_n_epochs=5):
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        # Get validation embeddings
        z = getattr(pl_module, "val_latents", None)
        if z is None:
            return

        # Different visualizations based on pair type
        if pl_module.config.get('pair_type') == 'MultiComponent':
            self._plot_hierarchical_umap(trainer, pl_module, epoch)
        else:
            self._plot_legacy_umap(trainer, pl_module, epoch)
    
    def _plot_hierarchical_umap(self, trainer, pl_module, epoch):
        """Plot UMAP with hierarchical labels"""
        z = pl_module.val_latents
        matter_types = getattr(pl_module, "val_matter_types", None)
        cosmologies = getattr(pl_module, "val_cosmologies", None)
        components = getattr(pl_module, "val_components", None)
        
        if matter_types is None:
            return
        
        # Compute UMAP projection
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = reducer.fit_transform(z.cpu().numpy())
        
        # Create a 2x2 subplot for different hierarchy levels
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Matter type
        scatter1 = axes[0,0].scatter(
            umap_proj[:, 0], umap_proj[:, 1], 
            c=matter_types.cpu().numpy(), 
            cmap='Set1', s=8, alpha=0.7
        )
        axes[0,0].set_title(f"Matter Type (Epoch {epoch})")
        axes[0,0].set_xlabel("UMAP 1")
        axes[0,0].set_ylabel("UMAP 2")
        axes[0,0].grid(True, alpha=0.3)
        
        # Add matter type legend
        matter_names = {0: 'Baryonic', 1: 'Dark Matter', 2: 'Total Mass'}
        handles1 = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=scatter1.cmap(scatter1.norm(i)), 
                              markersize=8, label=matter_names[i]) for i in matter_names.keys()]
        axes[0,0].legend(handles=handles1)
        
        # Plot 2: Cosmology
        scatter2 = axes[0,1].scatter(
            umap_proj[:, 0], umap_proj[:, 1],
            c=cosmologies.cpu().numpy(), 
            cmap='viridis', s=8, alpha=0.7
        )
        axes[0,1].set_title(f"Cosmology (Epoch {epoch})")
        axes[0,1].set_xlabel("UMAP 1")
        axes[0,1].set_ylabel("UMAP 2")
        axes[0,1].grid(True, alpha=0.3)
        
        # Add cosmology legend
        cosmo_names = {0: 'CDM', 1: 'WDM'}
        handles2 = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=scatter2.cmap(scatter2.norm(i)),
                              markersize=8, label=cosmo_names[i]) for i in cosmo_names.keys()]
        axes[0,1].legend(handles=handles2)
        
        # Plot 3: Combined matter type + cosmology
        # Create combined labels for better visualization
        combined_labels = matter_types * 2 + cosmologies
        scatter3 = axes[1,0].scatter(
            umap_proj[:, 0], umap_proj[:, 1],
            c=combined_labels.cpu().numpy(), 
            cmap='tab10', s=8, alpha=0.7
        )
        axes[1,0].set_title(f"Matter Type + Cosmology (Epoch {epoch})")
        axes[1,0].set_xlabel("UMAP 1")
        axes[1,0].set_ylabel("UMAP 2")
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Component type (for baryonic matter only)
        baryonic_mask = matter_types == 0
        if baryonic_mask.any():
            baryonic_proj = umap_proj[baryonic_mask]
            baryonic_components = components[baryonic_mask]
            
            scatter4 = axes[1,1].scatter(
                baryonic_proj[:, 0], baryonic_proj[:, 1],
                c=baryonic_components.cpu().numpy(),
                cmap='coolwarm', s=12, alpha=0.8
            )
            axes[1,1].set_title(f"Baryonic Components (Epoch {epoch})")
            axes[1,1].set_xlabel("UMAP 1")
            axes[1,1].set_ylabel("UMAP 2")
            axes[1,1].grid(True, alpha=0.3)
            
            # Add component legend
            comp_names = {0: 'Gas', 1: 'Stars'}
            handles4 = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=scatter4.cmap(scatter4.norm(i)),
                                  markersize=8, label=comp_names[i]) for i in comp_names.keys()]
            axes[1,1].legend(handles=handles4)
        else:
            axes[1,1].text(0.5, 0.5, 'No baryonic\ncomponents\nin validation set', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title(f"Baryonic Components (Epoch {epoch})")
        
        plt.tight_layout()
        plt.savefig(f"hierarchical_umap_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save embedding analysis
        self._analyze_embedding_separation(z, matter_types, cosmologies, components, epoch)
    
    def _plot_legacy_umap(self, trainer, pl_module, epoch):
        """Legacy UMAP plot for backward compatibility"""
        z = pl_module.val_latents
        softscore = getattr(pl_module, "val_softscores", None)
        if softscore is None:
            return

        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_proj = reducer.fit_transform(z.cpu().numpy())

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(umap_proj[:, 0], umap_proj[:, 1],
                         c=softscore.cpu().numpy(), cmap="viridis", s=5, alpha=0.8)
        plt.title(f"UMAP Colored by Softscore (Epoch {epoch})")
        plt.colorbar(sc, label="Softscore (CDM→0, WDM→1)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"umap_softscore_epoch_{epoch}.png", dpi=150)
        plt.close()
    
    def _analyze_embedding_separation(self, embeddings, matter_types, cosmologies, components, epoch):
        """Analyze and log embedding separation quality"""
        from sklearn.metrics import silhouette_score
        
        try:
            # Silhouette scores for different hierarchy levels
            matter_silhouette = silhouette_score(embeddings.cpu().numpy(), matter_types.cpu().numpy())
            cosmology_silhouette = silhouette_score(embeddings.cpu().numpy(), cosmologies.cpu().numpy())
            
            # Only compute component silhouette for baryonic matter
            baryonic_mask = matter_types == 0
            if baryonic_mask.sum() > 1:  # Need at least 2 samples
                baryonic_embeddings = embeddings[baryonic_mask]
                baryonic_components = components[baryonic_mask]
                if len(torch.unique(baryonic_components)) > 1:  # Need at least 2 classes
                    component_silhouette = silhouette_score(
                        baryonic_embeddings.cpu().numpy(), 
                        baryonic_components.cpu().numpy()
                    )
                else:
                    component_silhouette = 0.0
            else:
                component_silhouette = 0.0
            
            # Log metrics (if you have a logger)
            metrics = {
                f'embedding_separation/matter_silhouette': matter_silhouette,
                f'embedding_separation/cosmology_silhouette': cosmology_silhouette,
                f'embedding_separation/component_silhouette': component_silhouette,
            }
            
            # Print metrics
            print(f"\n=== Embedding Quality Metrics (Epoch {epoch}) ===")
            print(f"Matter type separation: {matter_silhouette:.3f}")
            print(f"Cosmology separation: {cosmology_silhouette:.3f}")
            print(f"Component separation: {component_silhouette:.3f}")
            
        except Exception as e:
            print(f"Could not compute silhouette scores: {e}")


# Legacy callback for backward compatibility
class UMAPCallback(HierarchicalUMAPCallback):
    """Backward compatibility wrapper"""
    def __init__(self, every_n_epochs=5):
        super().__init__(every_n_epochs)
        print("Using legacy UMAPCallback - consider upgrading to HierarchicalUMAPCallback")