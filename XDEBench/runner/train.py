import os
import sys
import csv
import torch
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluator.metrics import *
from utils.tools import *
from models.xde_model import BCTNormalizer
from transformers import get_cosine_schedule_with_warmup

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"

def test(args, net, dataset, test_data, coords, num_chemical):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)

        input_data = net.encoder(test_data[:, 0])
        for t in range(steps):
            output_pred = net(input_data, coords)
            output_gth = net.encoder(test_data[:, t + 1])

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
    data = train_data.clone()
    data[:, :, :num_chemical] = normalizer.BCT(data[:, :, :num_chemical])
    dims = tuple([0, 1] + list(range(3, data.dim())))
    mean = data.mean(dim=dims)
    std = data.std(dim=dims)
    return mean, std


def train(args, net, dataset):
    device = args.device
    net.to(device)
    sub = args.sub

    coords = torch.tensor(dataset["coords"], dtype=torch.float)[None]
    print(f'coords: {coords.shape}')
    num_train = len(dataset["train_groups"])
    dim = coords.shape[1]
    
    coords = normalize_coords(coords).to(device)

    batch_size = args.batch_size
    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    print(f'num_chemical: {num_chemical} / num_temperature: {num_temperature} / num_density: {num_density} / num_velocity: {num_velocity} / num_pressure: {num_pressure}')

    train_data = load_data(args.data_path, dataset, datatype="train", sub=sub)
    val_data = load_data(args.data_path, dataset, datatype="val", sub=sub)
    test_data = load_data(args.data_path, dataset, datatype="test", sub=sub)
    if dim == 2:
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)

    T = train_data.shape[1]
    print(f'train_data: {train_data.shape}')

    mean, std = compute_mean_std(train_data, num_chemical)
        
    net.set_normalizer(mean.to(device), std.to(device), num_chemical)

    save_dir = f"{args.data_path}/log_model/experiment_results_{args.experiment}_model_{args.model}"
    os.makedirs(save_dir, exist_ok=True) 

    if args.optim_type == "adam":
        optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == "adamw":
        optimizer = optim.AdamW(net.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.num_iterations+1, max_lr=args.lr)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=args.num_iterations+1)
    train_loss_list = []
    val_error_list = []
    best_val_error = None

    for step in range(args.num_iterations):
        net.train()
        indices = list(range(num_train))
        random.shuffle(indices)

        for i in range(num_train // batch_size):
            idx = indices[i * batch_size: (i + 1) * batch_size]
            random_ts = random.choices(range(0, T - 2), k=batch_size)
            input_batch = torch.stack([train_data[idx[b], random_ts[b]] for b in range(batch_size)]).to(device)
            output_batch = torch.stack([train_data[idx[b], random_ts[b] + 1] for b in range(batch_size)]).to(device)

            # # 动态加载每个batch的train_data
            # input_batch_list = []
            # output_batch_list = []
            # for b in range(batch_size):
            #     group_idx = idx[b]
            #     t = random_ts[b]
            #     npz = np.load(f'{args.data_path}/data/train/{dataset["train_groups"][group_idx]}.npz')
            #     data = npz["data"]  # shape: [T, C, H, W]
            #     input_batch_list.append(torch.tensor(data[t]))
            #     output_batch_list.append(torch.tensor(data[t + 1]))
            # input_batch = torch.stack(input_batch_list).to(device)
            # output_batch = torch.stack(output_batch_list).to(device)

            input_batch = net.encoder(input_batch)
            output_gth = net.encoder(output_batch)
            output_pred = net(input_batch, coords)

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
            val_error_now = test(args, net, dataset, val_data, coords, num_chemical)

            train_loss_list.append(loss.item())
            val_error_list.append(val_error_now)

            if best_val_error is None or val_error_now < best_val_error:
                best_val_error = val_error_now
                print("test error---------------------")
                best_test_error = test(args, net, dataset, test_data, coords, num_chemical)
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
    final_val_error = test(args, net, dataset, val_data, coords, num_chemical)
    print("test error---------------------")
    final_test_error = test(args, net, dataset, test_data, coords, num_chemical)

    os.makedirs(f"{args.data_path}/log_train_history", exist_ok=True)
    os.makedirs(f"{args.data_path}/log_results", exist_ok=True)
    np.savez_compressed(f"{args.data_path}/log_train_history/{args.model_name}_history.npz",
                        train_loss=np.array(train_loss_list),
                        val_error=np.array(val_error_list))

    # temp for compare with rollout
    args.random_crop = True
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