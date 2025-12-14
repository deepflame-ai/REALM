import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse
from utils.tools import setup_seed, param_flops
from models.xde_model import XDESolver

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def main(args):
    os.makedirs(f'{args.data_path}/log_train/log_{args.experiment}', exist_ok=True)
    os.makedirs(f'{args.data_path}/log_model', exist_ok=True)

    if args.start_from_best:
        args.model_name = args.load_model_name
    else:
        now = datetime.now()
        timestring = f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        rand_suffix = f'{random.randint(0, 99999):05d}'
        timestring = f'{timestring}_{rand_suffix}'
        args.model_name = (
            f"{args.model}-seed-{args.seed}-{timestring}"
        )

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
    args.vertex_dim = int(total_data_channels)
    args.edge_dim = dim + 1

    if dim == 2:
        args.modes = args.modes[0]
    
    net = XDESolver(args)

    if args.start_from_best:
        logfile = f'{args.data_path}/log_train/log_{args.experiment}/log_{args.model_name}_restart.txt'
    else:
        logfile = f'{args.data_path}/log_train/log_{args.experiment}/log_{args.model_name}.txt'
    sys.stdout = open(logfile, 'w')
    print('--------args----------')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('--------args----------\n')

    param_flops(net)
    sys.stdout.flush()
    if args.random_crop:
        print("Using random cropping for training")
        from runner.train_3d_randomCrop import train
        train(args, net, dataset)
    else:
        if args.model in ['MGN', 'GAT', 'GraphSAGE', 'GraphUNet']:
            from runner.train_graph import train
            train(args, net, dataset)
        else:
            from runner.train import train
            train(args, net, dataset)

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
    parser.add_argument('--modes', type=lambda x: int(x), nargs='*', default=[40], help='Number of Fourier modes')
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

