import random
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import Pk_library as PKL

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def power_spectrum(image):

    # parameters
    BoxSize = 25.0     #Mpc/h
    MAS     = 'None'  #MAS used to create the image; 'NGP', 'CIC', 'TSC', 'PCS' o 'None'
    threads = 1       #number of openmp threads
    delta = np.log1p(image)

    # compute the Pk of that image
    Pk2D = PKL.Pk_plane(delta, BoxSize, MAS, threads, verbose=False)
    Pk = Pk2D.Pk     #Pk in (Mpc/h)^2
    return Pk

class CosmicWebPk(Dataset):
    """Power spectrum for CDM/WDM cosmic web classification"""
    
    def __init__(self, cdm_data, wdm_data, indices, transform=None):
        """
        Args:
            cdm_data: numpy array of shape [N, H, W]
            wdm_data: numpy array of shape [N, H, W]
            indices: list of indices to use
            transform: optional transform to apply to images before computing P(k)
        """
        self.cdm_data = cdm_data
        self.wdm_data = wdm_data
        self.indices = indices
        self.transform = transform
        
        self.samples = []
        self.labels = []
        
        for idx in indices:
            if idx < len(cdm_data):
                self.samples.append(('cdm', idx))
                self.labels.append(0.0)
            if idx < len(wdm_data):
                self.samples.append(('wdm', idx))
                self.labels.append(1.0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data_type, data_idx = self.samples[idx]
        label = self.labels[idx]

        # Get the image
        image = self.cdm_data[data_idx] if data_type == 'cdm' else self.wdm_data[data_idx]
        image = torch.from_numpy(image).float()

        # Ensure correct shape
        if image.dim() > 2:
            image = image.squeeze()

        # Apply optional preprocessing
        if self.transform:
            image = self.transform(image)
        
        # Compute power spectrum (you must define this function elsewhere)
        pk = power_spectrum(image.numpy()) 
        
        # Optional: Normalize for stability
        pk_log = np.log1p(np.abs(pk))  # Avoids log(0), stabilizes outliers
        
        return torch.tensor(pk_log, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
