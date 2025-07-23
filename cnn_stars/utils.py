import torch
from torch.utils.data import Dataset
import numpy as np

class StarDataset(Dataset):
    """Dataset class for CDM/WDM cosmic web classification"""
    
    def __init__(self, cdm_data, wdm_data, indices, transform=None):
        """
        Args:
            cdm_data: numpy array of shape [N, H, W] - CDM samples
            wdm_data: numpy array of shape [N, H, W] - WDM samples  
            indices: list of indices to use from the datasets
            transform: optional transform to apply to samples
        """
        self.cdm_data = cdm_data
        self.wdm_data = wdm_data
        self.indices = indices
        self.transform = transform
        
        # Create labels: 0 for CDM, 1 for WDM
        # We'll alternate between CDM and WDM samples
        self.samples = []
        self.labels = []
        
        for idx in indices:
            if idx < len(cdm_data):
                self.samples.append(('cdm', idx))
                self.labels.append(0.0)  # CDM = 0
                # self.labels.append(0.05)  # CDM = 0.05 for balanced dataset
            if idx < len(wdm_data):
                self.samples.append(('wdm', idx))
                self.labels.append(1.0)  # WDM = 1
                # self.labels.append(0.95)  # WDM = 0.95 for balanced dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_type, data_idx = self.samples[idx]
        label = self.labels[idx]
        
        # Get the appropriate sample
        if data_type == 'cdm':
            image = self.cdm_data[data_idx]
        else:  # wdm
            image = self.wdm_data[data_idx]
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Ensure image has shape [H, W] (remove extra dimensions if needed)
        if image.dim() > 2:
            image = image.squeeze()
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)



def load_dataset(indices, transform=None, cdm_file='cdm_data.npy', wdm_file='wdm_data.npy'):
    """
    Load CDM and WDM datasets from .npy files and create a PyTorch dataset
    
    Args:
        indices: list of indices to use for sampling
        transform: optional transform to apply to samples
        cdm_file: path to CDM .npy file
        wdm_file: path to WDM .npy file
    
    Returns:
        CosmicWebDataset: PyTorch dataset containing CDM and WDM samples
    """
    try:
        # Load the data files
        print(f"Loading CDM data from {cdm_file}...")
        cdm_data = np.load(cdm_file)
        print(f"CDM data shape: {cdm_data.shape}")
        
        print(f"Loading WDM data from {wdm_file}...")
        wdm_data = np.load(wdm_file)
        print(f"WDM data shape: {wdm_data.shape}")
        
        # Validate data shapes
        if len(cdm_data.shape) != 3 or len(wdm_data.shape) != 3:
            raise ValueError("Data should have shape [N, H, W]")
        
        # Create and return dataset
        dataset = StarDataset(cdm_data, wdm_data, indices, transform)
        print(f"Created dataset with {len(dataset)} samples")
        
        return dataset
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise