import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import load_multicomponent_dataset, organize_component_files
from transform import TensorAugment

class EnhancedContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, data_root, config):
        super().__init__()
        self.data_root = data_root
        self.config = config
        
        # Organize component files
        self.component_files = organize_component_files(data_root)
        
    def setup(self, stage=None):
        # Split indices - same as before
        all_indices = random.sample(range(15000), self.config['k_samples'])
        random.shuffle(all_indices)
        N = len(all_indices)
        train_end = int(self.config['train_ratio'] * N)
        val_end = int((self.config['train_ratio'] + self.config['val_ratio']) * N)

        self.train_indices = all_indices[:train_end]
        self.val_indices = all_indices[train_end:val_end]
        self.test_indices = all_indices[val_end:]

        # Create transforms
        self.transform = TensorAugment(
            size=(self.config['img_size'], self.config['img_size']),
            p_flip=0.5,
            p_rot=0.5,
            noise_std=0,
            blur_kernel=self.config['blur_kernel'],
            apply_log=True,
            normalize=self.config['normalize']
        )
        
        # Create datasets with multi-component support
        pair_type = self.config.get('pair_type', 'MultiComponent')
        
        self.train_dataset = load_multicomponent_dataset(
            self.train_indices, 
            transform=self.transform,
            component_files=self.component_files,
            pair_type=pair_type
        )
        
        self.val_dataset = load_multicomponent_dataset(
            self.val_indices, 
            transform=self.transform,
            component_files=self.component_files,
            pair_type=pair_type
        )
        
        print(f"Setup complete:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Test indices: {len(self.test_indices)}")
        print(f"  Pair type: {pair_type}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
    def test_dataloader(self):
        test_dataset = load_multicomponent_dataset(
            self.test_indices,
            transform=None,  # No augmentation for test
            component_files=self.component_files,
            pair_type=self.config.get('pair_type', 'MultiComponent')
        )
        return DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
    
    def get_component_info(self):
        """Return information about available components"""
        return {
            'component_files': self.component_files,
            'train_size': len(self.train_indices),
            'val_size': len(self.val_indices), 
            'test_size': len(self.test_indices)
        }


# Legacy support - wrapper for old interface
class ContrastiveDataModule(EnhancedContrastiveDataModule):
    """Backward compatibility wrapper"""
    def __init__(self, cdm_file, wdm_file, config):
        # Convert old interface to new one
        import os
        data_root = os.path.dirname(cdm_file)
        
        # Create temporary component files dict for backward compatibility
        temp_component_files = {
            'total_cdm': cdm_file,
            'total_wdm': wdm_file,
        }
        
        super().__init__(data_root, config)
        
        # Override component files for backward compatibility
        self.component_files.update(temp_component_files)
        
        print("Using legacy ContrastiveDataModule interface")
        print("Consider migrating to EnhancedContrastiveDataModule for full multi-component support")