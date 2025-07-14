import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import CDMWDMPairDataset, load_dataset
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

        cdm_data = load_dataset(self.cdm_file, self.train_indices, self.transform)
        wdm_data = load_dataset(self.wdm_file, self.train_indices, self.transform)
        self.train_dataset = CDMWDMPairDataset(cdm_data, wdm_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)
