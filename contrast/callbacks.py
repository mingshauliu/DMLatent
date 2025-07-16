from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import umap
import seaborn as sns

class UMAPCallback(Callback):
    def __init__(self, every_n_epochs=5):
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        z = getattr(pl_module, "val_latents", None)
        softscore = getattr(pl_module, "val_softscores", None)
        if z is None or softscore is None:
            return

        reducer = umap.UMAP(n_components=2)
        umap_proj = reducer.fit_transform(z.cpu().numpy())

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(umap_proj[:, 0], umap_proj[:, 1],
                         c=softscore.cpu().numpy(), cmap="viridis", s=5, alpha=0.8)
        plt.title(f"UMAP Colored by Softscore (Epoch {epoch})")
        plt.colorbar(sc, label="Softscore (CDM→0, WDM→1)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"umap_softscore_epoch_{epoch}.png")
        plt.close()
        