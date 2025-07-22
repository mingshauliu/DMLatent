import torch
import numpy as np
from torch.utils.data import Dataset

class MultiComponentDataset(Dataset):
    """
    Dataset that loads multiple component maps (gas, stars, DM, total) 
    and creates appropriate labels for hierarchical contrastive learning
    """
    def __init__(self, indices, transform, component_files, pair_type='MultiComponent'):
        self.indices = indices
        self.transform = transform
        self.pair_type = pair_type
        
        # Load all component maps - 8 total component types
        self.gas_cdm_maps = np.load(component_files['gas_cdm'])      # Gas in CDM cosmology
        self.gas_wdm_maps = np.load(component_files['gas_wdm'])      # Gas in WDM cosmology
        self.stars_cdm_maps = np.load(component_files['stars_cdm'])  # Stars in CDM cosmology
        self.stars_wdm_maps = np.load(component_files['stars_wdm'])  # Stars in WDM cosmology
        self.dm_cdm_maps = np.load(component_files['dm_cdm'])        # DM in CDM cosmology
        self.dm_wdm_maps = np.load(component_files['dm_wdm'])        # DM in WDM cosmology
        self.total_cdm_maps = np.load(component_files['total_cdm'])  # Total mass CDM
        self.total_wdm_maps = np.load(component_files['total_wdm'])  # Total mass WDM
        
        print(f"Loaded dataset with {len(self.indices)} samples")
        print(f"Available components: {list(component_files.keys())}")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        map_idx = self.indices[idx]
        
        if self.pair_type == 'MultiComponent':
            return self._get_multicomponent_pair(map_idx)
        elif self.pair_type == 'CDMWDM':
            return self._get_cdmwdm_pair(map_idx)  # Original functionality
        elif self.pair_type == 'SimCLR':
            return self._get_simclr_pair(map_idx)
        else:
            raise ValueError(f"Unknown pair_type: {self.pair_type}")
    
    def _get_multicomponent_pair(self, map_idx):
        """
        Create pairs for multi-component contrastive learning
        Returns two different component maps from the same simulation
        """
        # Define component types with proper cosmology distinction
        # All components exist in both CDM and WDM cosmologies
        components = {
            # Gas components
            'gas_cdm': {'map': self.gas_cdm_maps[map_idx], 'matter_type': 0, 'cosmology': 0, 'component': 0},  # Gas-CDM
            'gas_wdm': {'map': self.gas_wdm_maps[map_idx], 'matter_type': 0, 'cosmology': 1, 'component': 0},  # Gas-WDM
            
            # Stellar components  
            'stars_cdm': {'map': self.stars_cdm_maps[map_idx], 'matter_type': 0, 'cosmology': 0, 'component': 1},  # Stars-CDM
            'stars_wdm': {'map': self.stars_wdm_maps[map_idx], 'matter_type': 0, 'cosmology': 1, 'component': 1},  # Stars-WDM
            
            # Dark matter components
            'dm_cdm': {'map': self.dm_cdm_maps[map_idx], 'matter_type': 1, 'cosmology': 0, 'component': 0},  # DM-CDM
            'dm_wdm': {'map': self.dm_wdm_maps[map_idx], 'matter_type': 1, 'cosmology': 1, 'component': 0},  # DM-WDM
            
            # Total mass components
            'total_cdm': {'map': self.total_cdm_maps[map_idx], 'matter_type': 2, 'cosmology': 0, 'component': 0},  # Total-CDM
            'total_wdm': {'map': self.total_wdm_maps[map_idx], 'matter_type': 2, 'cosmology': 1, 'component': 0},  # Total-WDM
        }
        
        # Sample two different components (positive pairs from same simulation)
        component_names = list(components.keys())
        comp1_name, comp2_name = np.random.choice(component_names, size=2, replace=False)
        
        comp1 = components[comp1_name]
        comp2 = components[comp2_name]
        
        # Convert to tensors and apply transforms
        map1 = torch.from_numpy(comp1['map'].copy()).float().unsqueeze(0)
        map2 = torch.from_numpy(comp2['map'].copy()).float().unsqueeze(0)
        
        if self.transform:
            map1 = self.transform(map1)
            map2 = self.transform(map2)
        
        # Create hierarchical labels
        matter_type1 = torch.tensor(comp1['matter_type'], dtype=torch.long)
        matter_type2 = torch.tensor(comp2['matter_type'], dtype=torch.long)
        cosmology1 = torch.tensor(comp1['cosmology'], dtype=torch.long)
        cosmology2 = torch.tensor(comp2['cosmology'], dtype=torch.long)
        component1 = torch.tensor(comp1['component'], dtype=torch.long)
        component2 = torch.tensor(comp2['component'], dtype=torch.long)
        
        # Create combined labels for different hierarchy levels
        # Level 1: Matter type (0=Baryonic, 1=DM, 2=Total)
        # Level 2: Cosmology within matter type (0=CDM, 1=WDM)  
        # Level 3: Component within baryonic matter (0=gas, 1=stars)
        
        return {
            'map1': map1, 'map2': map2,
            'matter_type1': matter_type1, 'matter_type2': matter_type2,
            'cosmology1': cosmology1, 'cosmology2': cosmology2,
            'component1': component1, 'component2': component2,
            'component1_name': comp1_name, 'component2_name': comp2_name
        }
    
    def _get_cdmwdm_pair(self, map_idx):
        """Original CDM/WDM pair generation for backward compatibility"""
        cdm_map = torch.from_numpy(self.total_cdm_maps[map_idx].copy()).float().unsqueeze(0)
        wdm_map = torch.from_numpy(self.total_wdm_maps[map_idx].copy()).float().unsqueeze(0)
        
        if self.transform:
            cdm_map = self.transform(cdm_map)
            wdm_map = self.transform(wdm_map)
        
        cdm_label = torch.tensor(0.0, dtype=torch.float)
        wdm_label = torch.tensor(1.0, dtype=torch.float)
        
        return cdm_map, wdm_map, cdm_label, wdm_label
    
    def _get_simclr_pair(self, map_idx):
        """SimCLR-style augmentation pairs from same map"""
        # Randomly choose a component type - now 8 options
        components = ['gas_cdm', 'gas_wdm', 'stars_cdm', 'stars_wdm', 
                     'dm_cdm', 'dm_wdm', 'total_cdm', 'total_wdm']
        comp_name = np.random.choice(components)
        
        # Get the appropriate map
        component_map_dict = {
            'gas_cdm': self.gas_cdm_maps,
            'gas_wdm': self.gas_wdm_maps,
            'stars_cdm': self.stars_cdm_maps,
            'stars_wdm': self.stars_wdm_maps,
            'dm_cdm': self.dm_cdm_maps,
            'dm_wdm': self.dm_wdm_maps,
            'total_cdm': self.total_cdm_maps,
            'total_wdm': self.total_wdm_maps
        }
        
        base_map = component_map_dict[comp_name][map_idx]
            
        map1 = torch.from_numpy(base_map.copy()).float().unsqueeze(0)
        map2 = torch.from_numpy(base_map.copy()).float().unsqueeze(0)
        
        if self.transform:
            map1 = self.transform(map1)
            map2 = self.transform(map2)
            
        return map1, map2


def load_multicomponent_dataset(indices, transform, component_files, pair_type='MultiComponent'):
    """
    Load dataset with multiple component types
    
    Args:
        indices: List of simulation indices to use
        transform: Data augmentation transforms
        component_files: Dict with paths to component files
        pair_type: Type of pairs to generate
    """
    return MultiComponentDataset(indices, transform, component_files, pair_type)


# Example usage and data organization
def organize_component_files(data_root):
    """
    Helper function to organize component file paths
    Now includes 8 component types (all matter components in both cosmologies)
    """
    component_files = {
        # Gas in both cosmologies
        'gas_cdm': f'{data_root}/msliu/CMD/data/IllustrisTNG/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy',
        'gas_wdm': f'{data_root}/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mgas_IllustrisTNG_WDM_z=0.00.npy',
        
        # Stars in both cosmologies
        'stars_cdm': f'{data_root}/msliu/CMD/data/IllustrisTNG/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy', 
        'stars_wdm': f'{data_root}/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mstar_IllustrisTNG_WDM_z=0.00.npy',
        
        # Dark matter in both cosmologies
        'dm_cdm': f'{data_root}/msliu/CMD/data/IllustrisTNG/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy',
        'dm_wdm': f'{data_root}/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mcdm_IllustrisTNG_WDM_z=0.00.npy',
        
        # Total mass in both cosmologies
        'total_cdm': f'{data_root}/msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy',
        'total_wdm': f'{data_root}/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mtot_IllustrisTNG_WDM_z=0.00.npy',
    }
    
    # Verify all files exist
    import os
    for comp_name, filepath in component_files.items():
        if not os.path.exists(filepath):
            print(f"Warning: {comp_name} file not found at {filepath}")
    
    return component_files


def create_balanced_sampling_strategy(component_files, indices):
    """
    Create sampling strategy to ensure balanced representation of all component types
    """
    n_components = len(component_files)
    n_samples = len(indices)
    
    # Calculate how many samples per component type
    samples_per_component = n_samples // n_components
    
    print(f"Total samples: {n_samples}")
    print(f"Components: {n_components}")
    print(f"Samples per component: {samples_per_component}")
    
    return {
        'samples_per_component': samples_per_component,
        'total_samples': n_samples,
        'component_names': list(component_files.keys())
    }