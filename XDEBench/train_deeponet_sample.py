import os
import sys
import random
import csv
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from datetime import datetime
from runner.train import *
from utils.tools import setup_seed, param_flops
from evaluator.metrics import *
from utils.tools import *
from models.xde_model import XDESolver


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"

def test_deeponet(args, net, dataset, test_data, coords, times, num_chemical):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    coords_dim = len(test_data.shape) - 3

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    num_points = args.num_points  # 每次送进网络的点数
    B = test_data.shape[0]

    with torch.no_grad():
        test_data = test_data.to(device)
        coords = coords.to(device)
        times = times.to(device)

        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0,
                            velocity=0.0, pressure=0.0)

        # 扩展 times 到 batch 维
        if coords_dim == 2:
            times = times.repeat(B, 1, 1, 1)
        elif coords_dim == 3:
            times = times.repeat(B, 1, 1, 1, 1)
        else:
            times = times.repeat(B, 1, 1)

        # 编码初始场（完整点）
        input_data = net.encoder(test_data[:, 0])   # 形状示例: [B, C, NPoints]
        NPoints = input_data.shape[-1]

        for t in range(steps):
            # 准备这一时刻的 GT 编码（完整点）
            output_gth_full = net.encoder(test_data[:, t + 1])  # [B, C_out, NPoints]

            # 为这一时刻准备一个容器，逐块写入预测
            output_pred_full = torch.zeros_like(output_gth_full)

            # 对所有点分块：0..NPoints 以 num_points 为步长滑动
            for start in range(0, NPoints, num_points):
                end = min(start + num_points, NPoints)
                idx = slice(start, end)

                # 取本 chunk 的坐标，形状示例: [?, ?, num_points_chunk]
                coords_chunk = coords[..., idx].to(device)

                # 取本 chunk 的输入特征（如果 encoder 输出有点维）
                input_data_chunk = input_data[..., idx]           # [B, C, num_points_chunk]

                # 时间特征按需要复制到点维（视 net 的实现而定）
                # 这里只是保持你原来的用法：input_time 不含点维
                input_time = times[:, t + 1].unsqueeze(1)         # 你的原逻辑
                times_chunk = input_time[..., idx]               # [B, 1, num_points_chunk]
                
                # 前向：每次只处理 num_points_chunk 个点
                output_pred_chunk = net(input_data_chunk, coords_chunk, times_chunk)
                # output_pred_chunk: [B, C_out, num_points_chunk]

                # 回填到完整预测张量
                output_pred_full[..., idx] = output_pred_chunk

            # 现在 output_pred_full 已经包含所有点的预测
            diff = output_pred_full - output_gth_full

            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:
                                                 num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:
                                               num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:]
                                             ** 2).mean().item()

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss

    
def train_deeponet(args, net, dataset):

    device = args.device
    net.to(device)

    coords = torch.tensor(dataset["coords"], dtype=torch.float)[None]
    dim = coords.shape[1]
            
    train_data = load_data(args.data_path, dataset, datatype="train", sub=args.sub)
    val_data = load_data(args.data_path, dataset, datatype="val", sub=args.sub)
    test_data = load_data(args.data_path, dataset, datatype="test", sub=args.sub)

    coords = normalize_coords(coords).to(device)
    coords_dim = len(train_data.shape) - 3
    
    # if coords_dim <= 2:
    #     train_data = train_data.to(device)
    #     val_data = val_data.to(device)
    #     test_data = test_data.to(device)

    print(f'train_data: {train_data.shape}, {len(train_data.shape)}')
    
    base = torch.arange(0, train_data.shape[1], dtype=torch.float32) / (train_data.shape[1] - 1)
    if coords_dim == 2:
        time = (base[None, :, None, None].expand(1, train_data.shape[1], train_data.shape[-2], train_data.shape[-1]).contiguous())
    elif coords_dim == 3:
        time = (base[None, :, None, None, None].expand(1, train_data.shape[1], train_data.shape[-3], train_data.shape[-2], train_data.shape[-1]).contiguous()) 
    else:
        time = (base[None, :, None].expand(1, train_data.shape[1], train_data.shape[-1]).contiguous())
    time = time.to(device)    
    print(f'time shape: {time.shape}')
    print(f'coords shape: {coords.shape}')

    num_train = len(dataset["train_groups"])
    T = time.shape[1]
    batch_size = args.batch_size
    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    print(f'num_chemical: {num_chemical} / num_temperature: {num_temperature} / num_density: {num_density} / num_velocity: {num_velocity} / num_pressure: {num_pressure}')

    mean, std = compute_mean_std(train_data, num_chemical)

    net.set_normalizer(mean.to(device), std.to(device), num_chemical)
    save_dir = f"{args.data_path}/log_model/experiment_results_{args.experiment}_model_{args.model}"
    os.makedirs(save_dir, exist_ok=True) 

    if args.optim_type == "adam":
        optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == "adamw":
        optimizer = optim.AdamW(net.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.num_iterations+1, max_lr=args.lr)
    train_loss_list = []
    val_error_list = []
    best_val_error = None

    args.num_points = 5000000
    num_points = args.num_points
    for step in range(args.num_iterations):
        net.train()
        indices = list(range(num_train))
        random.shuffle(indices)

        for i in range(num_train // batch_size):
            idx = indices[i * batch_size: (i + 1) * batch_size]
            random_ts = random.choices(range(1, T - 1), k=batch_size)  # output time step

            # 先按 batch / time 取出 [B, C, NPoints]
            input_batch_full = torch.stack([train_data[idx[b], 0] for b in range(batch_size)]).to(device)            # [B, C, NPoints]
            output_batch_full = torch.stack([train_data[idx[b], random_ts[b]] for b in range(batch_size)]).to(device)            # [B, C, NPoints]

            # 在 NPoints 维度上随机采样 num_points 个点
            # 若是「不放回」采样，num_points 需要 <= NPoints
            NPoints = input_batch_full.shape[-1]
            point_idx = torch.randint(
                low=0, high=NPoints, size=(batch_size, num_points), device=device
            )   # [B, num_points]
            point_idx_ = point_idx[0]

            # 利用高级索引对每个样本单独采样点: 先扩展到 [B, C, num_points]
            input_batch = input_batch_full[:, :, point_idx_]   # [B, C, num_points]
            output_batch = output_batch_full[:, :, point_idx_] # [B, C, num_points]
            coords_batch = coords[:, :, point_idx_]  # [1, dim, num_points]

            # 时间特征如果也需要按点复制，可以这样（若不需要点维度，可保持 [B, ...]）
            input_time = torch.stack([time[:, random_ts[b]] for b in range(batch_size)])  # 你的原逻辑
            time_batch = input_time[..., point_idx_]  # [B, 1, num_points]

            input_batch = net.encoder(input_batch)
            output_gth = net.encoder(output_batch)
            output_pred = net(input_batch, coords_batch, time_batch)

            loss = 0.0
            if num_pressure > 0 and num_density > 0:
                for start, count, key in zip(
                        [0, num_chemical, num_chemical + num_temperature, num_chemical + num_temperature + num_density, num_chemical + num_temperature + num_density + num_velocity],
                        [num_chemical, num_temperature, num_density, num_velocity, num_pressure],
                        ['chemical', 'temperature', 'density', 'velocity', 'pressure']):
                    diff = output_pred[:, start:start+count] - output_gth[:, start:start+count]
                    if args.loss_modes == 'mae':
                        loss += diff.abs().mean()
                    elif args.loss_modes == 'mse':
                        loss += (diff ** 2).mean()
                    else:
                        raise ValueError(f"Unknown loss mode: {args.loss_modes}")
            elif num_density > 0:
                for start, count, key in zip(
                    [0, num_chemical, num_chemical + num_temperature, num_chemical + num_temperature + num_density],
                    [num_chemical, num_temperature, num_density, num_velocity],
                    ['chemical', 'temperature', 'density', 'velocity']):
                    diff = output_pred[:, start:start+count] - output_gth[:, start:start+count]
                    if args.loss_modes == 'mae':
                        loss += diff.abs().mean()
                    elif args.loss_modes == 'mse':
                        loss += (diff ** 2).mean()
                    else:
                        raise ValueError(f"Unknown loss mode: {args.loss_modes}")
            elif num_pressure > 0:
                for start, count, key in zip(
                    [0, num_chemical, num_chemical + num_temperature, num_chemical + num_temperature + num_velocity],
                    [num_chemical, num_temperature, num_velocity, num_pressure],
                    ['chemical', 'temperature', 'velocity', 'pressure']):
                    diff = output_pred[:, start:start+count] - output_gth[:, start:start+count]
                    if args.loss_modes == 'mae':
                        loss += diff.abs().mean()
                    elif args.loss_modes == 'mse':
                        loss += (diff ** 2).mean()
                    else:
                        raise ValueError(f"Unknown loss mode: {args.loss_modes}")
            else:
                for start, count, key in zip(
                    [0, num_chemical, num_chemical + num_temperature],
                    [num_chemical, num_temperature, num_velocity],
                    ['chemical', 'temperature', 'velocity']):
                    diff = output_pred[:, start:start+count] - output_gth[:, start:start+count]
                    if args.loss_modes == 'mae':
                        loss += diff.abs().mean()
                    elif args.loss_modes == 'mse':
                        loss += (diff ** 2).mean()
                    else:
                        raise ValueError(f"Unknown loss mode: {args.loss_modes}")   

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        if step % 50 == 0:
            print(f"\n{step} ########################")
            print(f"Training loss: {loss.item():.6f}")
            print("validation error---------------------")
            val_error_now = test_deeponet(args, net, dataset, val_data, coords, time, num_chemical)
            train_loss_list.append(loss.item())
            val_error_list.append(val_error_now)

            if best_val_error is None or val_error_now < best_val_error:
                best_val_error = val_error_now
                print("test error---------------------")
                best_test_error = test_deeponet(args, net, dataset, test_data, coords, time, num_chemical)
                print('-----------SAVING NEW MODEL-----------')
                torch.save({
                    'model_state': net.state_dict(),
                    'mean': mean.cpu(),
                    'std': std.cpu(),
                    'num_chemical': num_chemical
                }, f'{save_dir}/best_{args.model_name}.pt')
            sys.stdout.flush()

    print('\n----------------------------FINAL_RESULT-----------------------------')
    ckpt = torch.load(f'{save_dir}/best_{args.model_name}.pt')
    net.load_state_dict(ckpt['model_state'])
    net.set_normalizer(ckpt['mean'].to(device), ckpt['std'].to(device), ckpt['num_chemical'])
    net.to(device)
    
    print("validation error---------------------")
    final_val_error = test_deeponet(args, net, dataset, val_data, coords, time, num_chemical)
    print("test error---------------------")
    final_test_error = test_deeponet(args, net, dataset, test_data, coords, time, num_chemical)

    os.makedirs(f"{args.data_path}/log_train_history", exist_ok=True)
    os.makedirs(f"{args.data_path}/log_results", exist_ok=True)
    np.savez_compressed(f"{args.data_path}/log_train_history/{args.model_name}_history.npz",
                        train_loss=np.array(train_loss_list),
                        val_error=np.array(val_error_list))
    
    params = param_flops(net)
    
    csv_path = f"{args.data_path}/log_results/experiment_results_{args.experiment}_model_{args.model}.csv"
    args_dict = vars(args)
    headers = list(args_dict.keys()) + ["val_error", "test_error", "params"]
    row = list(args_dict.values()) + [final_val_error, final_test_error, params]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


def main(args):
    os.makedirs(f'{args.data_path}/log_train/log_{args.experiment}', exist_ok=True)
    os.makedirs(f'{args.data_path}/log_model', exist_ok=True)

    now = datetime.now()
    timestring = f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
    rand_suffix = f'{random.randint(0, 99999):05d}'
    timestring = f'{timestring}_{rand_suffix}'
    args.model_name = (
        f"{args.model}-seed-{args.seed}-{timestring}"
    )

    writerPath = f'{args.data_path}/log_train/deeponetTrainer'
    os.makedirs(writerPath, exist_ok=True)
    
    setup_seed(args.seed) 
    dataset = np.load(args.data_path + '/data/data.npz')
    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]
    total_data_channels = num_chemical + num_temperature + num_density + num_velocity + num_pressure

    args.shape_list = tuple(dataset["spatial_size"].tolist())
    dim = dataset["coords"].shape[0]
    print(f'data shape: {args.shape_list}, dim: {dim}')
    args.num_chemical = num_chemical
    args.in_dim = total_data_channels + dim
    args.out_dim = total_data_channels

    net = XDESolver(args)

    logfile = f'{writerPath}/log_{args.model_name}.txt'
    sys.stdout = open(logfile, 'w')
    print('--------args----------')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('--------args----------\n')

    param_flops(net)
    sys.stdout.flush()
    train_deeponet(args, net, dataset)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-parameters')
    parser.add_argument('--data_path', type=str,
                        default='/home/zhangrui/zhangruiC/CoupledXDEBench_data/2dCoupledXDEBench',
                        help='Path to .npz data file (auto-resolved from main.py location)')
    parser.add_argument('--experiment', type=str, default='E1')

    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--model', type=str, default='FNO', help='Model type')
    parser.add_argument('--width', type=int, default=60)
    parser.add_argument('--act', type=str, default='gelu', help='Activation function')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of layers for neural operator')
    parser.add_argument('--modes', type=int, default=40, help='Number of Fourier modes')
    parser.add_argument('--in_dim', type=int, default=0)
    parser.add_argument('--out_dim', type=int, default=0)
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_chemical', type=int, default=0)
    
    # graph
    parser.add_argument('--vertex_dim', type=int, default=0)
    parser.add_argument('--edge_dim', type=int, default=0)
    parser.add_argument('--noise_std', type=float, default=0.0)
    
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--loss_modes', type=str, default='mae', help='Loss function: mae, mse')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--optim_type', type=str, default='adam')
    
    # Random Crop
    parser.add_argument('--random_crop', type=bool, default=False, help='Use random cropping for training')
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--overlap_size', type=int, default=32)
    parser.add_argument('--n_patches', type=int, default=10)

    # Start from best
    parser.add_argument('--start_from_best', type=bool, default=False, help='Load weights from best model')
    parser.add_argument('--load_model_name', type=str, default='')
    
    # Downsample factor for time step
    parser.add_argument('--sub', type=int, default=1)
    
    args = parser.parse_args()
    main(args)

