import numpy as np
import os
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataSplit:
    def __init__(self, data: np.ndarray, variables: List[str]):
        self.data = data  # shape: (N, T, C, H, W) or (N, T, C, H, W, L)
        self.variables = variables
        self.var_idx = {v: i for i, v in enumerate(variables)}

        # 分组索引
        self.chemical_idx = [i for i, v in enumerate(variables)
                             if v not in {'Ux.npy', 'Uy.npy', 'Uz.npy', 'T.npy', 'rho.npy', 'pMax.npy', 'p.npy'}]
        self.velocity_idx = [self.var_idx[v] for v in ['Ux.npy', 'Uy.npy', 'Uz.npy'] if v in self.var_idx]
        self.temperature_idx = [self.var_idx['T.npy']] if 'T.npy' in self.var_idx else []
        self.density_idx = [self.var_idx['rho.npy']] if 'rho.npy' in self.var_idx else []

        if 'pMax.npy' in self.var_idx:
            self.pressure_idx = [self.var_idx['pMax.npy']] if 'pMax.npy' in self.var_idx else []
        elif 'p.npy' in self.var_idx:
            self.pressure_idx = [self.var_idx['p.npy']] if 'p.npy' in self.var_idx else []
        else:
            self.pressure_idx = []


    def __getattr__(self, name):
        if name in ['chemical', 'velocity', 'temperature', 'density', 'pressure']:
            idx = getattr(self, f"{name}_idx")
            return self.data[:, :, idx, ...]  # (N, T, C_group, H, W) or (N, T, C_group, H, W, L)
        raise AttributeError(f"'DataSplit' object has no attribute '{name}'")

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class ODE_PDEDataSet:
    train_data: DataSplit
    val_data: DataSplit
    test_data: DataSplit
    variables: List[str]
    groups: List[str]
    times: List[str]
    spatial_size: Tuple[int, int]
    coords: Dict[str, np.ndarray]

    def __repr__(self):
        return (f"PDEDataSet(train={self.train_data.data.shape}, "
                f"val={self.val_data.data.shape}, test={self.test_data.data.shape}, "
                f"{len(self.variables)} variables, {len(self.groups)} groups, "
                f"{len(self.times)} time steps)\n"
                f"  chemical:   {[v for v in self.variables if v not in {'Ux.npy', 'Uy.npy', 'Uz.npy', 'T.npy', 'rho.npy', 'pMax.npy', 'p.npy'}]}\n"
                f"  velocity:   {[v for v in ['Ux.npy', 'Uy.npy', 'Uz.npy'] if v in self.variables]}\n"
                f"  temperature:{['T.npy'] if 'T.npy' in self.variables else []}\n"
                f"  pressure:   {[v for v in ['pMax.npy', 'p.npy'] if v in self.variables]}\n"
                f"  density:    {['rho.npy'] if 'rho.npy' in self.variables else []}"
                )

    @staticmethod
    def from_raw(data: np.ndarray,
                 variables: List[str],
                 groups: List[str],
                 times: List[str],
                 coords: Dict[str, np.ndarray],
                 train_size: int,
                 val_size: int,
                 test_size: int) -> "ODE_PDEDataSet":
        assert data.shape[0] == train_size + val_size + test_size

        train_data = data[:train_size].astype(np.float32)
        val_data = data[train_size:train_size + val_size].astype(np.float32)
        test_data = data[train_size + val_size:].astype(np.float32)
        groups_name = [s.replace('/', '_') for s in groups]

        return ODE_PDEDataSet(
            train_data=DataSplit(train_data, variables),
            val_data=DataSplit(val_data, variables),
            test_data=DataSplit(test_data, variables),
            variables=variables,
            groups=groups_name,
            times=times,
            spatial_size=data.shape[3:],
            coords=coords,
        )

    def save(self, filepath: str):
        if 'Cz' in self.coords:
            coord_stack = np.stack([self.coords['Cx'], self.coords['Cy'], self.coords['Cz']], axis=0).astype(np.float32)
        else:
            coord_stack = np.stack([self.coords['Cx'], self.coords['Cy']], axis=0).astype(np.float32)

        train_size = self.train_data.data.shape[0]
        val_size = self.val_data.data.shape[0]
        test_size = self.test_data.data.shape[0]
        spatial_size = self.train_data.data.shape[3:]

        num_chemical = self.train_data.chemical.shape[2]
        num_temperature = self.train_data.temperature.shape[2]
        num_density = self.train_data.density.shape[2]
        num_velocity = self.train_data.velocity.shape[2]
        num_pressure = self.train_data.pressure.shape[2]
        print(f'num_chemical: {num_chemical} / num_temperature: {num_temperature} / num_density: {num_density} / num_velocity: {num_velocity} / num_pressure: {num_pressure}')

        train_groups = self.groups[:train_size]
        val_groups = self.groups[train_size:train_size + val_size]
        test_groups = self.groups[train_size + val_size:]

        os.makedirs(f'{filepath}/train', exist_ok=True)
        os.makedirs(f'{filepath}/val', exist_ok=True)
        os.makedirs(f'{filepath}/test', exist_ok=True)

        for iTrain in range(train_size):
            np.savez_compressed(f'{filepath}/train/{train_groups[iTrain]}.npz',
                data=self.train_data.data[iTrain])
            
        for iVal in range(val_size):
            np.savez_compressed(f'{filepath}/val/{val_groups[iVal]}.npz',
                data=self.val_data.data[iVal])  
                      
        for iTest in range(test_size):
            np.savez_compressed(f'{filepath}/test/{test_groups[iTest]}.npz',
                data=self.test_data.data[iTest])
        
        np.savez_compressed(f'{filepath}/data.npz',
                coords=coord_stack,
                times=self.times,
                variables=self.variables,
                train_groups=train_groups,
                val_groups=val_groups,
                test_groups=test_groups,
                spatial_size=spatial_size,
                num_chemical=num_chemical,
                num_temperature=num_temperature,
                num_density=num_density,
                num_velocity=num_velocity,
                num_pressure=num_pressure)
        
        print(f"✅ Saved dataset to: {filepath}")