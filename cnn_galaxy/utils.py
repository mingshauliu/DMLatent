import torch
from torch.utils.data import Dataset, ConcatDataset
import h5py
import numpy as np
import random
import os

class HDF5KeyImageDatasetBalanced(Dataset):
    """Dataset class that returns balanced batches of CDM and WDM"""
    
    def __init__(self, cdm_path, wdm_path, transform=None):
        self.transform = transform

        # Open files
        self.cdm_file = h5py.File(cdm_path, 'r')
        self.wdm_file = h5py.File(wdm_path, 'r')

        # Collect keys
        self.cdm_keys = list(self.cdm_file.keys())
        self.wdm_keys = list(self.wdm_file.keys())

        # Match length by undersampling longer set
        self.length = min(len(self.cdm_keys), len(self.wdm_keys))

        # Shuffle to avoid same ordering
        random.shuffle(self.cdm_keys)
        random.shuffle(self.wdm_keys)

    def __len__(self):
        return self.length * 2  # Because we return interleaved CDM/WDM

    def __getitem__(self, idx):
        half_idx = idx // 2
        is_wdm = idx % 2 == 1

        if is_wdm:
            key = self.wdm_keys[half_idx]
            h5_file = self.wdm_file
            label = 1
        else:
            key = self.cdm_keys[half_idx]
            h5_file = self.cdm_file
            label = 0

        arr = h5_file[key][:]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0, None).astype(np.float32)

        if arr.max() > 0:
            arr = arr / arr.max()

        img = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

    def __del__(self):
        if hasattr(self, 'cdm_file'):
            self.cdm_file.close()
        if hasattr(self, 'wdm_file'):
            self.wdm_file.close()
            
def load_multiple_hdf5_datasets(indices, transform=None, base_path='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images'):
    """Load multiple HDF5 datasets and concatenate them, skipping corrupted or missing files"""
    datasets = []
    for idx in indices:
        cdm_path = f'{base_path}/CDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        wdm_path = f'{base_path}/WDM/MW_zooms/box_{idx}/CDM/Galaxy_{idx}.hdf5'
        
        if not (os.path.exists(cdm_path) and os.path.exists(wdm_path)):
            print(f"Warning: Skipping missing files for box_{idx}")
            continue
        
        try:
            # Attempt to open files to detect corruption
            with h5py.File(cdm_path, 'r') as f:
                _ = list(f.keys())
            with h5py.File(wdm_path, 'r') as f:
                _ = list(f.keys())
        except Exception as e:
            print(f"Warning: Skipping corrupted file for box_{idx}: {e}")
            continue

        try:
            ds = HDF5KeyImageDatasetBalanced(cdm_path, wdm_path, transform=transform)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: Skipping box_{idx} due to dataset init failure: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets found!")
    
    return ConcatDataset(datasets)