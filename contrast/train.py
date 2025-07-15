import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from contrastive_module import ContrastiveModel
from data_module import ContrastiveDataModule
from callbacks import UMAPCallback

def main():
    config = {
        'img_size': 256,
        'dropout': 0.1,
        'batch_size': 64,
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'epochs': 80,
        'patience': 10,  # Early stopping patience
        'k_samples': 15000,
        'model_type': 'tiny',
        'blur_kernel': 0,
        'normalize': False,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'temperature': 0.05,  # Temperature for contrastive loss
        'pretrained_path': "/n/netscratch/iaifi_lab/Lab/msliu/small_net/best_cnn_model_blur_0_tiny.pt" # Path to pretrained model if needed
    }

    cdm_path = '/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy'
    wdm_path = '/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mtot_IllustrisTNG_WDM_z=0.00.npy'

    dm = ContrastiveDataModule(cdm_path, wdm_path, config)
    model = ContrastiveModel(config)

    early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'], mode='min')
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best-contrastive')

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        callbacks=[early_stop, checkpoint, UMAPCallback(every_n_epochs=5)],
        accelerator='auto',
        log_every_n_steps=10
    )
    
    checkpoint_path = "/n/netscratch/iaifi_lab/Lab/msliu/contrast/lightning_logs/version_10/checkpoints/best-contrastive.ckpt"  

    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path)
    # trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
