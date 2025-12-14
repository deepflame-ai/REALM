import os
import sys
import csv
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from utils.tools import *
from models.xde_model import BCTNormalizer
from data.graph_dataset import ODEGraphDataset, ODEGraphDatasetRollout

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def test(args, net, dataset, test_loader, coords, num_chemical):
    device = args.device
    net.eval()

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0)
        
        for b_i, graph_batch in enumerate(test_loader):
            y = graph_batch.y
            steps = y.shape[1]
            graph_batch.input_fields = net.encoder(graph_batch.input_fields)
            for t in range(steps):
                output_pred = net.model(graph_batch, roll_out=True)
                output_gth = net.encoder(y[:, t])
                
                total_losses['chemical'] += ((output_pred - output_gth)[:, :num_chemical] ** 2).mean().item()
                total_losses['temperature'] += ((output_pred - output_gth)[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
                total_losses['density'] += ((output_pred - output_gth)[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
                total_losses['velocity'] += ((output_pred - output_gth)[:, num_chemical + num_temperature + num_density:] ** 2).mean().item()

                graph_batch.input_fields = output_pred

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss


def compute_mean_std(train_data, num_chemical):
    normalizer = BCTNormalizer(num_chemical=num_chemical, eps=1e-40)
    data = train_data.clone()
    data[:, :, :num_chemical] = normalizer.BCT(data[:, :, :num_chemical])
    mean = data.mean(dim=(0, 1, 3))
    std = data.std(dim=(0, 1, 3))
    return mean, std


def train(args, net, dataset):
    device = args.device
    net.to(device)

    coords = torch.tensor(dataset["coords"], dtype=torch.float)[None]
    coords = normalize_coords(coords).to(device)
    print(f'coords: {coords.shape}')
    
    batch_size = args.batch_size
    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]
    
    train_data = load_data(args.data_path, dataset, datatype="train").to(device)
    val_data = load_data(args.data_path, dataset, datatype="val").to(device)
    test_data = load_data(args.data_path, dataset, datatype="test").to(device)

    mean, std = compute_mean_std(train_data, num_chemical)
    net.set_normalizer(mean.to(device), std.to(device), num_chemical)
    save_dir = f"{args.data_path}/log_model/experiment_results_{args.experiment}_model_{args.model}"
    os.makedirs(save_dir, exist_ok=True) 

    # print info
    tr_dataset = ODEGraphDataset(train_data, coords)
    print(f'train_data: {train_data.shape}, {coords.shape}')
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size)
    val_dataset = ODEGraphDatasetRollout(val_data, coords=coords)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    te_dataset = ODEGraphDatasetRollout(test_data, coords)
    te_loader = DataLoader(te_dataset, batch_size=1)
    optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.num_iterations+1, max_lr=args.lr)
    train_loss_list = []
    val_error_list = []
    best_val_error = None

    for step in range(args.num_iterations):
        net.train()

        for i, graph_batch in enumerate(tr_loader):
            graph_batch.input_fields = net.encoder(graph_batch.input_fields)
            output_gth = net.encoder(graph_batch.y)
            output_pred = net.model(graph_batch)
    
            loss = 0.0
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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        if step % 20 == 0:
            print(f"\n{step} ########################")
            print(f"Training loss: {loss.item():.6f}")
            print("validation error---------------------")
            val_error_now = test(args, net, dataset, val_loader, coords, num_chemical)
            train_loss_list.append(loss.item())
            val_error_list.append(val_error_now)

            if best_val_error is None or val_error_now < best_val_error:
                best_val_error = val_error_now
                print("test error---------------------")
                best_test_error = test(args, net, dataset, te_loader, coords, num_chemical)
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
    final_val_error = test(args, net, dataset, val_loader, coords, num_chemical)
    print("test error---------------------")
    final_test_error = test(args, net, dataset, te_loader, coords, num_chemical)

    os.makedirs(f"{args.data_path}/log_train_history", exist_ok=True)
    os.makedirs(f"{args.data_path}/log_results", exist_ok=True)
    np.savez_compressed(f"{args.data_path}/log_train_history/{args.model_name}_history.npz",
                        train_loss=np.array(train_loss_list),
                        val_error=np.array(val_error_list))

    csv_path = f"{args.data_path}/log_results/experiment_results_{args.experiment}_model_{args.model}.csv"
    args_dict = vars(args)
    headers = list(args_dict.keys()) + ["val_error", "test_error"]
    row = list(args_dict.values()) + [final_val_error, final_test_error]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)