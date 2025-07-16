import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import load_contrastive_dataset
from transform import TensorAugment

class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, cdm_file, wdm_file, config):
        super().__init__()
        self.cdm_file = cdm_file
        self.wdm_file = wdm_file
        self.config = config

    def setup(self, stage=None):
        # Split indices
        all_indices = random.sample(range(15000), self.config['k_samples'])
        random.shuffle(all_indices)
        N = len(all_indices)
        train_end = int(self.config['train_ratio'] * N)
        val_end = int((self.config['train_ratio'] + self.config['val_ratio']) * N)

        self.train_indices = all_indices[:train_end]
        self.val_indices = all_indices[train_end:val_end]
        self.test_indices = all_indices[val_end:]

        self.transform = TensorAugment(
            size=(self.config['img_size'], self.config['img_size']),
            p_flip=0.5,
            p_rot=0.5,
            noise_std=0,
            blur_kernel=self.config['blur_kernel'],
            apply_log=True,
            normalize=self.config['normalize']
        )
        
        self.train_dataset = load_contrastive_dataset(
            self.train_indices, 
            transform=self.transform,
            cdm_file=self.cdm_file,
            wdm_file=self.wdm_file,
            pair_type=self.config.get('pair_type', 'CDMWDM')  # Default to 'CDMWDM' if not specified
        )
        
        self.val_dataset = load_contrastive_dataset(
            self.val_indices, 
            transform=self.transform,
            cdm_file=self.cdm_file,
            wdm_file=self.wdm_file,
            pair_type=self.config.get('pair_type', 'CDMWDM')  # Default to 'CDMWDM' if not specified
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'])

