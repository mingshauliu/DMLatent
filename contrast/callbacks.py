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
        y = getattr(pl_module, "val_labels", None)
        if z is None or y is None:
            return

        reducer = umap.UMAP(n_components=2)
        umap_proj = reducer.fit_transform(z.numpy())

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=umap_proj[:, 0], y=umap_proj[:, 1], hue=y.numpy(),
                        palette="coolwarm", s=5, alpha=0.8)
        plt.title(f"UMAP at Epoch {epoch}")
        plt.legend(title="Label", loc="best")
        plt.tight_layout()
        plt.savefig(f"umap_epoch_{epoch}.png")
        plt.close()
        