from data_module import EnhancedContrastiveDataModule
from utils import organize_component_files, create_balanced_sampling_strategy

def create_multicomponent_config():
    """
    Configuration for multi-component contrastive learning
    """
    config = {
        # Data parameters
        'img_size': 256,
        'batch_size': 32,
        'k_samples': 15000,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'num_workers': 4,
        
        # Model parameters  
        'model_type': 'large',
        'dropout': 0.1,
        
        # Training parameters
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'epochs': 80,
        'patience': 10,
        
        # Contrastive learning parameters
        'pair_type': 'MultiComponent',  # 'MultiComponent', 'CDMWDM', 'SimCLR'
        'loss_type': 'hierarchical_nxtent',  # We'll implement this in next steps
        'temperature': 0.05,
        
        # Multi-component specific parameters
        'matter_type_weight': 1.0,  # Weight for matter type separation
        'subtype_weight': 0.5,      # Weight for subtype separation
        'component_balance': True,   # Whether to balance component sampling
        
        # Augmentation parameters
        'blur_kernel': 0,
        'normalize': False,
        
        # Paths
        'pretrained_path': "/path/to/your/pretrained/model.pt"
    }
    return config

def setup_data_paths():
    """
    Define your actual data paths here - now with 8 component types
    """
    data_paths = {
        # Update these paths to match your actual file locations
        'data_root': '/n/netscratch/iaifi_lab/Lab/',
        'component_files': {
            # Gas components in both cosmologies
            'gas_cdm': 'msliu/CMD/data/IllustrisTNG/Maps_Mgas_IllustrisTNG_LH_z=0.00.npy',
            'gas_wdm': 'ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mgas_IllustrisTNG_WDM_z=0.00.npy',
            
            # Stellar components in both cosmologies
            'stars_cdm': 'msliu/CMD/data/IllustrisTNG/Maps_Mstar_IllustrisTNG_LH_z=0.00.npy',
            'stars_wdm': 'ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mstar_IllustrisTNG_WDM_z=0.00.npy',
            
            # Dark matter components in both cosmologies
            'dm_cdm': 'msliu/CMD/data/IllustrisTNG/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy',
            'dm_wdm': 'ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mcdm_IllustrisTNG_WDM_z=0.00.npy',
            
            # Total mass components (your existing files)
            'total_cdm': 'msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy',
            'total_wdm': 'ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mtot_IllustrisTNG_WDM_z=0.00.npy'
        }
    }
    return data_paths

def test_dataset_loading():
    """
    Test function to verify your dataset loads correctly
    """
    config = create_multicomponent_config()
    data_paths = setup_data_paths()
    
    # Create data module
    dm = EnhancedContrastiveDataModule(data_paths['data_root'], config)
    dm.setup()
    
    # Test data loading
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print("=== Dataset Loading Test ===")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    
    if config['pair_type'] == 'MultiComponent':
        print(f"Batch keys: {batch.keys()}")
        print(f"Map1 shape: {batch['map1'].shape}")
        print(f"Map2 shape: {batch['map2'].shape}")
        print(f"Matter types: {batch['matter_type1'][0].item()}, {batch['matter_type2'][0].item()}")
        print(f"Cosmologies: {batch['cosmology1'][0].item()}, {batch['cosmology2'][0].item()}")
        print(f"Components: {batch['component1'][0].item()}, {batch['component2'][0].item()}")
        print(f"Component names: {batch['component1_name'][0]}, {batch['component2_name'][0]}")
        
        # Decode the hierarchical labels
        matter_type_names = {0: 'Baryonic', 1: 'DarkMatter', 2: 'Total'}
        cosmology_names = {0: 'CDM', 1: 'WDM'}
        component_names = {0: 'Gas/DM', 1: 'Stars'}
        
        print(f"Decoded: {matter_type_names[batch['matter_type1'][0].item()]}-{cosmology_names[batch['cosmology1'][0].item()]} vs {matter_type_names[batch['matter_type2'][0].item()]}-{cosmology_names[batch['cosmology2'][0].item()]}")
    
    return dm, batch

def analyze_dataset_distribution(dm, num_batches=10):
    """
    Analyze the distribution of component types in your dataset
    """
    from collections import defaultdict
    
    component_counts = defaultdict(int)
    matter_type_counts = defaultdict(int)
    pair_combinations = defaultdict(int)
    
    train_loader = dm.train_dataloader()
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
            
        if dm.config['pair_type'] == 'MultiComponent':
            # Count component types
            for j in range(len(batch['component1_name'])):
                comp1 = batch['component1_name'][j] 
                comp2 = batch['component2_name'][j]
                
                component_counts[comp1] += 1
                component_counts[comp2] += 1
                
                # Count matter types
                matter_type_counts[batch['matter_type1'][j].item()] += 1
                matter_type_counts[batch['matter_type2'][j].item()] += 1
                
                # Count pair combinations
                pair_key = tuple(sorted([comp1, comp2]))
                pair_combinations[pair_key] += 1
    
    print("=== Dataset Distribution Analysis ===")
    print("Component counts:", dict(component_counts))
    print("Matter type counts:", dict(matter_type_counts))
    print("Top pair combinations:", dict(list(pair_combinations.items())[:10]))

if __name__ == "__main__":
    # Test the enhanced dataset
    dm, batch = test_dataset_loading()
    
    # Analyze distribution
    analyze_dataset_distribution(dm)
    
    print("\n=== Next Steps ===")
    print("1. Verify all your component files exist and have the same number of samples")
    print("2. Check that the component maps have the expected shapes and value ranges")
    print("3. Implement hierarchical contrastive loss (next step)")
    print("4. Test the training loop with a small dataset first")