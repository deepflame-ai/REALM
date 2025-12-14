import numpy as np
from typing import Dict, Literal, Optional
from pathlib import Path
from sklearn.neighbors import NearestNeighbors   # 仅 FPS 用到，pip install scikit-learn
import time
import os
import sys
import torch


# ---------------- 采样策略实现 ----------------
def _random_choice(n_total: int, ratio: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_new = int(n_total * ratio)
    idx = rng.choice(n_total, n_new, replace=False)
    return np.sort(idx)

def _uniform_stride(n_total: int, ratio: float) -> np.ndarray:
    """等距均匀采样：计算 stride 后向下取整"""
    stride = max(1, int(1 / ratio))
    return np.arange(0, n_total, stride)

def _fps(xyz: np.ndarray, n_sample: int) -> np.ndarray:
    """
    最远点采样（FPS）。
    xyz: (N, 2) or (N, 3)
    返回索引
    """
    N = xyz.shape[0]
    idx = np.zeros(n_sample, dtype=np.int64)
    idx[0] = np.random.randint(N)
    dist = np.full(N, np.inf)
    for i in range(1, n_sample):
        last = xyz[idx[i-1]]
        dist = np.minimum(dist, np.linalg.norm(xyz - last, axis=1))
        idx[i] = np.argmax(dist)
    return np.sort(idx)

def _grid_sample(xyz: np.ndarray, ratio: float) -> np.ndarray:
    """
    网格均匀采样：把坐标范围切成若干网格，每个网格随机保留一点。
    ratio 仅用于估算网格尺寸，不保证严格比例。
    """
    N = xyz.shape[0]
    n_target = int(N * ratio)

    # 计算网格边长
    bbox_min, bbox_max = xyz.min(0), xyz.max(0)
    extent = bbox_max - bbox_min
    # 假设近似均匀，估计每维 cell 数
    grid_cells_1d = int(np.ceil((N / n_target) ** (1 / extent.shape[0])))
    cell_size = extent / grid_cells_1d

    # 计算每个点所属网格索引
    grid_idx = np.floor((xyz - bbox_min) / cell_size).astype(np.int32)
    # 把多维索引压缩成一维 key
    key = np.ravel_multi_index(grid_idx.T, [grid_cells_1d]*extent.shape[0])
    unique_key, inverse = np.unique(key, return_inverse=True)

    # 每个网格随机保留一点
    mask = np.zeros(len(unique_key), dtype=bool)
    rng = np.random.default_rng(42)
    for k in range(len(unique_key)):
        ids = np.where(inverse == k)[0]
        mask[k] = ids[rng.integers(0, len(ids))]
    return np.where(mask[inverse])[0]

# ---------------- 主函数 ----------------
def downsample_npz(in_path, out_path, 
                   method: Literal["random", "stride", "fps", "grid"] = "random",
                   ratio: float = 0.10, seed: int = 42):
    """
    通用降采样入口
    """
    print(f"🔍 Loading original dataset ...")
    ds = np.load(in_path + '/data/data.npz')
    n_total = ds["spatial_size"][0]

    # 构造 (N, 2/3) 坐标矩阵
    xyz = torch.tensor(ds["coords"], dtype=torch.float)
    print(f'xyz: {xyz.shape}')
    
    print(f"📐 Original spatial points: {n_total}")
    print(f"🎲 Sampling method: {method}, ratio: {ratio}")

    if method == "random":
        idx = _random_choice(n_total, ratio, seed)
    elif method == "stride":
        idx = _uniform_stride(n_total, ratio)
    elif method == "fps":
        n_sample = max(1, int(n_total * ratio))
        idx = _fps(xyz, n_sample)
    elif method == "grid":
        idx = _grid_sample(xyz, ratio)
    else:
        raise ValueError("Unknown sampling method")

    n_new = len(idx)
    print(f"✅ Selected {n_new} points ({n_new/n_total:.2%})")

    # 降采样坐标
    coords_new = xyz[:, idx]
    print(f'coords_new: {coords_new.shape}')
    coord_dict = {}
    coord_dict["Cx"] = coords_new[0]
    coord_dict["Cy"] = coords_new[1]
    
    train_data = load_data(in_path, ds, datatype="train")
    val_data = load_data(in_path, ds, datatype="val")
    test_data = load_data(in_path, ds, datatype="test")

    # 降采样数据
    train_data_new = train_data[:, :, :, idx]
    val_data_new   = val_data[:, :, :, idx]
    test_data_new  = test_data[:, :, :, idx]
    print(f'train_data_new: {train_data_new.shape}')
    print(f'val_data_new: {val_data_new.shape}')
    print(f'test_data_new: {test_data_new.shape}')
    
    groups = []
    for iTrain in range(train_data_new.shape[0]):
        groups.append(ds["train_groups"][iTrain].tolist())
    for iVal in range(val_data_new.shape[0]):
        groups.append(ds["val_groups"][iVal].tolist())    
    for iTest in range(test_data_new.shape[0]):
        groups.append(ds["test_groups"][iTest].tolist())
    print(f'groups: {groups}')

    ds_down = ODE_PDEDataSet(
        train_data=DataSplit(train_data_new, ds["variables"]),
        val_data=DataSplit(val_data_new, ds["variables"]),
        test_data=DataSplit(test_data_new, ds["variables"]),
        variables=ds["variables"],
        groups=groups,
        times=ds["times"],
        spatial_size=(n_new,),
        coords=coord_dict,
        )

    print(f"💾 Saving to {out_path} ...")
    ds_down.save(out_path)
    print("✅ All done.")

# ---------------- 命令行示例 ----------------
if __name__ == "__main__":
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'XDEBench'))
    sys.path.insert(0, target_dir)

    from data.dataset import *
    from utils.tools import *

    downsample_npz(in_path="/aisi-nas/baixuan/XDEBench_Data/U2dRocket", 
                   out_path="/aisi-nas/baixuan/XDEBench_Data/U2dRocket3k/data",
                   method="random", ratio=0.010, seed=0)

