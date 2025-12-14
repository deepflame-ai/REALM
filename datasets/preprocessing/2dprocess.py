import os
import sys
import numpy as np
from typing import List, Tuple
from glob import glob
from utils import *


def load_pde_dataset(root_dir: str, data_size: List[int], spatial_size: Tuple[int, int] = (128, 128)):
    group_dirs = sorted([
        os.path.relpath(d, root_dir)
        for d in glob(os.path.join(root_dir, "*", "*"))
        if os.path.isdir(d)
    ])

    all_data = []
    var_list = None
    time_list = None
    coord_names = {"Cx.npy", "Cy.npy", "Cz.npy"}
    coord_dict = {}

    expected_size = np.prod(spatial_size)

    for group_idx, group in enumerate(group_dirs):
        group_path = os.path.join(root_dir, group)
        time_dirs = sorted(
            [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))],
            key=lambda x: float(x)
        )
        group_data = []

        for time_idx, time in enumerate(time_dirs):
            time_path = os.path.join(group_path, time)
            var_files = sorted([f for f in os.listdir(time_path) if f.endswith(".npy")])

            for v in var_files:
                if v == 'U.npy':
                    U = np.load(os.path.join(time_path, "U.npy"))
                    np.save(os.path.join(time_path, "Ux.npy"), U[:, 0])
                    np.save(os.path.join(time_path, "Uy.npy"), U[:, 1])
                    np.save(os.path.join(time_path, "Uz.npy"), U[:, 2])

            if var_list is None:
                raw_vars = [v for v in var_files if v not in coord_names]
                raw_vars = [v for v in raw_vars if (v != 'N2.npy' and v != 'p.npy' and v != 'U.npy' and v != 'vorticityz.npy')]  
                chem_vars = [v for v in raw_vars if v not in ('Ux.npy', 'Uy.npy', 'Uz.npy', 'T.npy', 'rho.npy', 'pMax.npy')]
                vel_vars = ['Ux.npy', 'Uy.npy']
                temp_var = ['T.npy']
                rho_var = ['rho.npy']
                pres_var = [] #['pMax.npy']
                var_list = chem_vars + temp_var + rho_var + vel_vars + pres_var
                time_list = time_dirs

            if group_idx == 0 and time_idx == 0:
                coordx_path = os.path.join(time_path, "Cx.npy")
                coordy_path = os.path.join(time_path, "Cy.npy")

                cdata_x = np.load(coordx_path)
                cdata_y = np.load(coordy_path)

                sort_indices = np.lexsort((-cdata_x, -cdata_y))

                cdata_x = cdata_x[sort_indices].reshape(spatial_size, order='F')
                cdata_y = cdata_y[sort_indices].reshape(spatial_size, order='F')

                coord_dict["Cx"] = cdata_x
                coord_dict["Cy"] = cdata_y

            vars_at_t = []
            for var in var_list:
                file_path = os.path.join(time_path, var)
                data = np.load(file_path)
                if data.shape == (expected_size,):
                    data = data[sort_indices].reshape(spatial_size, order='F')
                elif data.shape != spatial_size:
                    raise ValueError(f"Unexpected shape {data.shape} in {file_path}, expected {spatial_size}")
                vars_at_t.append(data[np.newaxis, ...])  # (1, H, W)

            group_data.append(np.concatenate(vars_at_t, axis=0)[np.newaxis, ...])  # (1, C, H, W)

        group_tensor = np.concatenate(group_data, axis=0)  # (T, C, H, W)
        all_data.append(group_tensor[np.newaxis, ...])  # (1, T, C, H, W)

    all_data = np.concatenate(all_data, axis=0)  # (N, T, C, H, W)
    group_dirs = np.array(group_dirs)
    perm = np.random.permutation(len(all_data))
    shuffled_data = all_data[perm]
    shuffled_groups = group_dirs[perm].tolist()

    field_stats = []
    for idx, name in enumerate(var_list):
        combined_data = np.concatenate(all_data[:, :, idx, :, :])
        stats = calculate_field_stats(combined_data)
        
        field_stats.append({
            'field_name': name[:-4],
            'stats': stats
        })

    train_traj = shuffled_groups[:data_size[0]]
    val_traj = shuffled_groups[data_size[0]:data_size[0] + data_size[1]]
    test_traj = shuffled_groups[data_size[0] + data_size[1]:]
    trajectories = []
    trajectories.append({
        'group': 'train',
        'n_trajectories': len(train_traj),
        'trajectories': train_traj
    })
    trajectories.append({
        'group': 'val',
        'n_trajectories': len(val_traj),
        'trajectories': val_traj
    })
    trajectories.append({
        'group': 'test',
        'n_trajectories': len(test_traj),
        'trajectories': test_traj
    })

    return ODE_PDEDataSet.from_raw(
        data=shuffled_data,
        variables=var_list,
        groups=shuffled_groups,
        times=time_list,
        coords=coord_dict,
        train_size=data_size[0],
        val_size=data_size[1],
        test_size=data_size[2]
    ), field_stats, trajectories


if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'XDEBench'))
    sys.path.insert(0, target_dir)

    from data.dataset import *

    setup_seed(0)
    data_size = [24, 3, 3]
    spatial_size = (256, 256)
    root_dir = '/aisi-nas/zhangteng/comb-train/evoJet-npy-3phi-10Re'
    save_path = '/aisi-nas/baixuan/XDEBench_Data/2dEvojet/data'
    dataset_name = f'evojet'

    dataset, field_stats, trajectories = load_pde_dataset(root_dir=root_dir, data_size=data_size, spatial_size=spatial_size)

    output_path = f'{save_path}/{dataset_name}_stats.yaml'
    description = f'2D evojet dataset'
    grid_type = f'uniform, cartesian coordinates'
    n_trajectories = trajectories[0]['n_trajectories'] + trajectories[1]['n_trajectories'] + trajectories[2]['n_trajectories']
    n_fields = len(dataset.variables)
    n_timeSteps = len(dataset.times)
    generate_stats_yaml(output_path, description, grid_type, spatial_size, field_stats, 
                        trajectories, n_trajectories, n_fields, n_timeSteps)
    print(f"✅ Dataset statistics saved to: {output_path}")

    dataset.save(f"{save_path}")