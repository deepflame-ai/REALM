import os
import sys
import csv
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from models.xde_model import BCTNormalizer
from evaluator.metrics import *
from utils.tools import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"

def test(args, net, test_data, coords, num_chemical, num_temperature, num_density, num_velocity, num_pressure):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    crop_size = args.crop_size
    overlap_size = args.overlap_size

    with torch.no_grad():
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)

        input_data = net.encoder(test_data[:, 0].to(device))
        for t in range(steps):
            output_pred = sliding_window_inference(net, input_data, coords, crop_size, overlap_size, device, args)
            output_gth = net.encoder(test_data[:, t + 1].to(device))
            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()

            input_data = output_pred

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss


def compute_mean_std(train_data, num_chemical):
    normalizer = BCTNormalizer(num_chemical=num_chemical, eps=1e-40)
    data = train_data.cpu().clone()
    data[:, :, :num_chemical] = normalizer.BCT(data[:, :, :num_chemical])
    mean = data.mean(dim=(0, 1, 3, 4, 5))
    std = data.std(dim=(0, 1, 3, 4, 5))
    return mean, std


def train(args, net, dataset):
    device = args.device
    net.to(device)

    coords = torch.tensor(dataset["coords"], dtype=torch.float).to(device)[None]
    print(f'coords: {coords.shape}')
    num_train = len(dataset["train_groups"])
    T = len(dataset["times"])

    batch_size = args.batch_size
    crop_size = args.crop_size
    n_patches = args.n_patches
    train_data = load_data(args.data_path, dataset, datatype="train")

    print(f'coords shape: {coords.shape}')
    print(f'train_data shape: {train_data.shape}')

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

    if args.start_from_best:
        ckpt = torch.load(f'{save_dir}/best_{args.model_name}.pt')
        net.load_state_dict(ckpt['model_state'])
        net.set_normalizer(ckpt['mean'].to(device), ckpt['std'].to(device), ckpt['num_chemical'])
        net.to(device)

    for step in range(args.num_iterations):
        net.train()
        for i in range(num_train):
            input_batch, output_batch, coords_batch = random_crop_3d(train_data, coords, crop_size, n_patches)
            # input_batch, output_batch, coords_batch = fixed_grid_crop_3d(train_data, coords, crop_size, n_patches)
            
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            input_batch = net.encoder(input_batch)
            output_gth = net.encoder(output_batch)
            output_pred = net(input_batch, coords_batch)

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
            val_data = load_data(args.data_path, dataset, datatype="val")
            val_error_now = test(args, net, val_data, coords, num_chemical, num_temperature, num_density, num_velocity, num_pressure)
            del val_data
            train_loss_list.append(loss.item())
            val_error_list.append(val_error_now)

            if best_val_error is None or val_error_now < best_val_error:
                best_val_error = val_error_now
                print("test error---------------------")
                test_data = load_data(args.data_path, dataset, datatype="test")
                best_test_error = test(args, net, test_data, coords, num_chemical, num_temperature, num_density, num_velocity, num_pressure)
                del test_data
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
    val_data = load_data(args.data_path, dataset, datatype="val")
    final_val_error = test(args, net, val_data, coords, num_chemical, num_temperature, num_density, num_velocity, num_pressure)
    print("test error---------------------")
    test_data = load_data(args.data_path, dataset, datatype="test")
    final_test_error = test(args, net, test_data, coords, num_chemical, num_temperature, num_density, num_velocity, num_pressure)

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