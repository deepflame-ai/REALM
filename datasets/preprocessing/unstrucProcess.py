import os
import sys
import numpy as np
from typing import List, Tuple
from glob import glob
import tqdm

import os
import sys
import numpy as np
from typing import List, Tuple
from glob import glob
from tqdm import tqdm          # ← 新增
from utils import *

def load_pde_dataset(root_dir: str,
                     data_size: List[int],
                     spatial_size: Tuple[int, int] = (128, 128, 128)):
    group_dirs = sorted([
        os.path.relpath(d, root_dir)
        for d in glob(os.path.join(root_dir, "*", "*"))
        if os.path.isdir(d)
    ])

    all_data = []
    var_list = None
    time_list = None
    coord_names = set()
    coord_dict = {}
    trajectories = []

    expected_size = np.prod(spatial_size)

    # 最外层进度条：遍历 group
    for group_idx, group in enumerate(tqdm(group_dirs,
                                           desc="[load_pde_dataset] Processing groups",
                                           unit="group")):
        group_path = os.path.join(root_dir, group)
        trajectories.append(group)
        time_dirs = sorted(
            [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))],
            key=lambda x: float(x)
        )
        group_data = []

        # 内层进度条：遍历 time
        for time_idx, time in enumerate(tqdm(time_dirs,
                                             desc=f"[load_pde_dataset]   Group {group}",
                                             leave=False,
                                             unit="step")):
            time_path = os.path.join(group_path, time)
            var_files = sorted([f for f in os.listdir(time_path) if f.endswith(".npy")])

            for v in var_files:
                if v == 'U.npy':
                    U = np.load(os.path.join(time_path, "U.npy"))
                    np.save(os.path.join(time_path, "Ux.npy"), U[:, 0])
                    np.save(os.path.join(time_path, "Uy.npy"), U[:, 1])
                    np.save(os.path.join(time_path, "Uz.npy"), U[:, 2])

            if var_list is None:
                raw_vars = [v for v in var_files]
                raw_vars = [v for v in raw_vars if (v != 'N2.npy' and v != 'p.npy' and v != 'U.npy' and v != 'Cx.npy' and v != 'Cy.npy' and v != 'Cz.npy' and v != 'Uz.npy')]
                chem_vars = [v for v in raw_vars if v not in ('Ux.npy', 'Uy.npy', 'T.npy', 'rho.npy')]
                vel_vars = ['Ux.npy', 'Uy.npy']
                temp_var = ['T.npy']
                rho_var = ['rho.npy']
                var_list = chem_vars + vel_vars + temp_var + rho_var
                time_list = time_dirs

            if group_idx == 0 and time_idx == 0:
                coordx_path = os.path.join(time_path, "Cx.npy")
                coordy_path = os.path.join(time_path, "Cy.npy")
                # coordz_path = os.path.join(time_path, "Cz.npy")

                cdata_x = np.load(coordx_path)
                cdata_y = np.load(coordy_path)
                # cdata_z = np.load(coordz_path)

                sort_indices = np.lexsort((-cdata_x, -cdata_y))

                cdata_x = cdata_x[sort_indices]
                cdata_y = cdata_y[sort_indices]
                # cdata_z = cdata_z[sort_indices]

                coord_dict["Cx"] = cdata_x
                coord_dict["Cy"] = cdata_y
                # coord_dict["Cz"] = cdata_z

            vars_at_t = []
            for var in var_list:
                file_path = os.path.join(time_path, var)
                data = np.load(file_path)
                if data.shape == (expected_size,):
                    data = data[sort_indices]
                elif data.shape != spatial_size:
                    raise ValueError(f"Unexpected shape {data.shape} in {file_path}, expected {spatial_size}")
                vars_at_t.append(data[np.newaxis, ...])

            group_data.append(np.concatenate(vars_at_t, axis=0)[np.newaxis, ...])

        # 原来的 DEBUG 信息保留，方便调试
        print(f"[DEBUG] var_list for group {group}: {var_list}")
        group_tensor = np.concatenate(group_data, axis=0)
        print(f'group_tensor:{group_tensor.shape}')
        all_data.append(group_tensor[np.newaxis, ...])

    # 后续流程与原代码完全一致
    all_data = np.concatenate(all_data, axis=0)
    print(f'all_data: {all_data.shape}')
    
    group_dirs = np.array(group_dirs)
    perm = np.random.permutation(len(all_data))
    shuffled_data = all_data[perm]
    shuffled_groups = group_dirs[perm].tolist()

    field_stats = []
    for idx, name in enumerate(var_list):
        combined_data = np.concatenate(all_data[:, :, idx, :])
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
    data_size = [8, 2, 2]
    spatial_size = (294900,)
    root_dir = '/aisi-nas/zhangteng/dfr-2d/dfr2d-npy'
    save_path = '/aisi-nas/baixuan/XDEBench_Data/U2dRocket/data'
    dataset_name = f'rocket'

    dataset, field_stats, trajectories = load_pde_dataset(root_dir=root_dir, data_size=data_size,
                                                          spatial_size=spatial_size)

    output_path = f'{save_path}/{dataset_name}_stats.yaml'
    description = f'rocket dataset'
    grid_type = f'unstructured mesh'
    n_trajectories = trajectories[0]['n_trajectories'] + trajectories[1]['n_trajectories'] + trajectories[2]['n_trajectories']
    n_fields = len(dataset.variables)
    n_timeSteps = len(dataset.times)
    generate_stats_yaml(output_path, description, grid_type, spatial_size, field_stats,
                        trajectories, n_trajectories, n_fields, n_timeSteps)
    print(f"✅ Dataset statistics saved to: {output_path}")

    dataset.save(f"{save_path}")

