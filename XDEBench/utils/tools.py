import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def param_flops(net):
    params = 0
    for p in net.parameters():
        params += p.numel() * (2 if p.is_complex() else 1)
    
    print(' params: %.3f M' % (params / 1000000.0))
    out_params = params / 1000000.0
    return out_params


def apply_colormap(tensor_pred, tensor_gt, cmap='jet'):
    if isinstance(tensor_pred, torch.Tensor):
        np_array_gt = tensor_gt.detach().cpu().numpy()
        np_array_pred = tensor_pred.detach().cpu().numpy()
    else:
        np_array_gt = tensor_gt 
        np_array_pred = tensor_pred
    norm_array_gt = (np_array_gt - np_array_gt.min()) / (np_array_gt.max() - np_array_gt.min() + 1e-8)
    norm_array_pred = (np_array_pred - np_array_gt.min()) / (np_array_gt.max() - np_array_gt.min() + 1e-8)
    
    cmap = plt.get_cmap(cmap)
    colored_gt = cmap(norm_array_gt)[:, :, :3]
    colored_pred = cmap(norm_array_pred)[:, :, :3]
    
    colored_tensor_gt = torch.tensor(colored_gt).permute(2, 0, 1)
    colored_tensor_pred = torch.tensor(colored_pred).permute(2, 0, 1)
    return colored_tensor_pred, colored_tensor_gt

def random_crop_3d(data, coords, patch_size, n_patches):
    # data: [N, T, C, x, y, z]
    N, T, C, X, Y, Z = data.shape
    patches = []
    for _ in range(n_patches):
        n_idx = random.randint(0, N-1)
        t_idx = random.randint(0, T-2)  # 保证t+1不越界
        x0 = random.randint(0, X - patch_size)
        y0 = random.randint(0, Y - patch_size)
        z0 = random.randint(0, Z - patch_size)
        patch_input = data[n_idx, t_idx, :, x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size]
        patch_coords = coords[0, :, x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size]
        patch_output = data[n_idx, t_idx+1, :, x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size]
        patches.append((patch_input, patch_output, patch_coords))
    input_batch = torch.stack([p[0] for p in patches])
    output_batch = torch.stack([p[1] for p in patches])
    coords_batch = torch.stack([p[2] for p in patches])
    return input_batch, output_batch, coords_batch

def fixed_grid_crop_3d(data, coords, patch_size, n_patches):
    # data: [N, T, C, X, Y, Z]
    N, T, C, X, Y, Z = data.shape
    
    # 确保数据尺寸可以被patch_size整除
    assert X % patch_size == 0 and Y % patch_size == 0 and Z % patch_size == 0, \
        "Data dimensions must be divisible by patch_size"
    
    # 计算每个维度上的块数
    n_blocks_x = X // patch_size
    n_blocks_y = Y // patch_size
    n_blocks_z = Z // patch_size
    total_blocks = n_blocks_x * n_blocks_y * n_blocks_z
    
    # 生成所有可能的块索引
    all_blocks = []
    for bx in range(n_blocks_x):
        for by in range(n_blocks_y):
            for bz in range(n_blocks_z):
                all_blocks.append((bx, by, bz))
    
    # 随机选择n_patches个块
    selected_blocks = random.sample(all_blocks, min(n_patches, total_blocks))
    
    patches = []
    for block in selected_blocks:
        bx, by, bz = block
        n_idx = random.randint(0, N-1)
        t_idx = random.randint(0, T-2)  # 保证t+1不越界
        
        # 计算块的起始坐标
        x0 = bx * patch_size
        y0 = by * patch_size
        z0 = bz * patch_size
        
        patch_input = data[n_idx, t_idx, :, x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size]
        patch_coords = coords[0, :, x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size]
        patch_output = data[n_idx, t_idx+1, :, x0:x0+patch_size, y0:y0+patch_size, z0:z0+patch_size]
        patches.append((patch_input, patch_output, patch_coords))
    
    input_batch = torch.stack([p[0] for p in patches])
    output_batch = torch.stack([p[1] for p in patches])
    coords_batch = torch.stack([p[2] for p in patches])
    
    return input_batch, output_batch, coords_batch

def sliding_window_inference(net, input_data, coords, patch_size, overlap, device, args):
    # input_data: [N, C, X, Y, Z]
    N, C, X, Y, Z = input_data.shape
    stride = patch_size - overlap
    xs = list(range(0, X - patch_size + 1, stride))
    ys = list(range(0, Y - patch_size + 1, stride))
    zs = list(range(0, Z - patch_size + 1, stride))
    if xs[-1] != X - patch_size: xs.append(X - patch_size)
    if ys[-1] != Y - patch_size: ys.append(Y - patch_size)
    if zs[-1] != Z - patch_size: zs.append(Z - patch_size)

    output = torch.zeros((N, args.out_dim, X, Y, Z), device=device)
    weight = torch.zeros_like(output)

    # 生成高斯权重
    sigma = patch_size / 8
    grid = np.meshgrid(
        np.linspace(-1, 1, patch_size),
        np.linspace(-1, 1, patch_size),
        np.linspace(-1, 1, patch_size),
        indexing='ij'
    )
    gauss = np.exp(-(grid[0]**2 + grid[1]**2 + grid[2]**2) / (2 * (sigma/patch_size)**2))
    gauss = torch.tensor(gauss, dtype=torch.float32, device=device)[None, None, :, :, :]

    for xi in xs:
        for yi in ys:
            for zi in zs:
                patch = input_data[:, :, xi:xi+patch_size, yi:yi+patch_size, zi:zi+patch_size]
                patch_coords = coords[:, :, xi:xi+patch_size, yi:yi+patch_size, zi:zi+patch_size]
                pred = net(patch, patch_coords)
                pred = pred.view(N, -1, patch_size, patch_size, patch_size)
                output[:, :, xi:xi+patch_size, yi:yi+patch_size, zi:zi+patch_size] += pred * gauss
                weight[:, :, xi:xi+patch_size, yi:yi+patch_size, zi:zi+patch_size] += gauss

    output = output / (weight + 1e-8)
    return output

def load_data(data_path, dataset, datatype="train", sub=1):
    all_data = []
    num = len(dataset[f"{datatype}_groups"])
    for i in range(num):
        npz = np.load(f'{data_path}/data/{datatype}/{dataset[f"{datatype}_groups"][i]}.npz')
        data = npz["data"][::sub]
        all_data.append(data)
    return torch.tensor(np.array(all_data), dtype=torch.float, device='cpu')

def normalize_coords(coords):

    num_dims = coords.shape[1]
    normalized_axes = []
    axis_range = None
    
    for dim in range(num_dims):
        axis_coords = coords[0, dim, :]
        
        axis_min = torch.min(axis_coords)
        axis_max = torch.max(axis_coords)
        
        if axis_range is None: # 保留坐标相对长度
            axis_range = axis_max - axis_min
            if axis_range == 0:
                axis_range = 1e-8

        normalized_axis = (axis_coords - axis_min) / axis_range
        normalized_axes.append(normalized_axis)
        
        print(f'dim{dim}_min: {axis_min}, dim{dim}_max: {axis_max}')
    
    normalized_coords = torch.stack(normalized_axes, dim=0).unsqueeze(0)
    
    for dim in range(num_dims):
        axis_coords = normalized_coords[0, dim, :]
        axis_min = torch.min(axis_coords)
        axis_max = torch.max(axis_coords)
        
        print(f'norm: dim{dim}_min: {axis_min}, dim{dim}_max: {axis_max}')
    
    return normalized_coords

def create_grayscale_scatter(x, y, values, figsize=(8, 1), dpi=100, 
                             vmin=None, vmax=None, s=10):
    """
    创建灰度散点图并返回图像数组和Figure对象
    """
    # 自动确定值范围
    if vmin is None or vmax is None:
        vmin = np.min(values) if vmin is None else vmin
        vmax = np.max(values) if vmax is None else vmax
    
    # 创建图形
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    
    # 使用灰度颜色映射
    plt.scatter(x, y, c=values, cmap='gray', s=s, vmin=vmin, vmax=vmax)
    plt.tight_layout(pad=0)
    
    # 将图形转换为图像数组
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_array = np.array(Image.open(buf).convert('L'))  # 转换为灰度图
    
    plt.close(fig)  # 及时关闭图形释放内存
    return img_array
