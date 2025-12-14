import torch
import random
import yaml
import os
import numpy as np

def calculate_field_stats(data):
    flat_data = data.flatten()
    
    mean_val = np.mean(flat_data)
    std_val = np.std(flat_data)
    rms_val = np.sqrt(np.mean(np.square(flat_data)))
    
    return {
        'mean': float(f"{mean_val:.6g}"),
        'std': float(f"{std_val:.6g}"),
        'rms': float(f"{rms_val:.6g}")
    }

def generate_stats_yaml(output_path, description, grid_type, spatial_resolution, field_stats, trajectories, 
                        n_trajectories, n_fields, n_timeSteps):
    data_stats = {
        'description': description,
        'grid_type': grid_type,
        'spatial_resolution': list(spatial_resolution),
        'n_trajectories': n_trajectories,
        'n_fields': n_fields,
        'n_timeSteps': n_timeSteps,
        'fields': field_stats,
        'trajectories': trajectories
    }

    yaml_content = yaml.dump(
        data_stats,
        sort_keys=False,         
        width=120,               
        indent=2,                
        default_flow_style=False
    )

    with open(output_path, 'w') as f:
        f.write(yaml_content)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True