import os
import csv
import glob
import torch
import ast
import numpy as np
import pandas as pd
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
from runner.eval import *
from utils.tools import setup_seed, param_flops
from models.xde_model import XDESolver

def extract_best_results(data_path=".", experiment_name="hit", evalutor="test_error"):

    log_dir = f"{data_path}"
    output_path = os.path.join(log_dir, "best_results.csv")
    os.makedirs(log_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(log_dir, f"log_results/experiment_results_{experiment_name}_model_*.csv"))
    print(f"Found {len(csv_files)} CSV files for experiment '{experiment_name}'")
    
    all_best_rows = []
    headers = None
    
    for file_path in csv_files:
        if os.stat(file_path).st_size == 0: 
            continue
            
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if headers is None and reader.fieldnames:
                headers = reader.fieldnames
            
            best_row = None
            min_error = float('inf')
            
            for row in reader:
                try:
                    test_error = float(row.get(evalutor, float('inf')))
                    if test_error < min_error:
                        min_error = test_error
                        best_row = row
                except (ValueError, TypeError):
                    continue
            
            if best_row:
                all_best_rows.append(best_row)
    
    if headers and all_best_rows:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_best_rows)
        
        print(f"Generated best_results.csv with {len(all_best_rows)} best records")

    else:
        print("No valid records found in any files")


def extract_best_results_rollout(data_path=".", experiment_name="hit", evalutor="test_error"):
    log_dir = f"{data_path}"
    os.makedirs(log_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(log_dir, f"log_results/experiment_results_{experiment_name}_model_*.csv"))
    print(f"Found {len(csv_files)} CSV files for experiment '{experiment_name}'")
    
    rollout_true_rows = []
    rollout_false_rows = []
    
    selected_fields = ['model', 'params', 'width', 'n_layers', 'modes', 'n_heads', 'val_error', 'test_error']
    
    skipped_deeponet_count = 0
    
    for file_path in csv_files:
        if os.stat(file_path).st_size == 0: 
            continue
            
        skip_file = False
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            try:
                first_row = next(reader)
                model_val = first_row.get('model', '')
                if model_val == 'DeepONet_deepxde' or model_val == 'DeepONet_deepxde3d' or model_val == 'DeepONet_deepxdeU':
                    skip_file = True
                    skipped_deeponet_count += 1
                    print(f"Skipping deeponet model file: {os.path.basename(file_path)}")
            except StopIteration:
                continue
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                continue
        
        if skip_file:
            continue
            
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            best_by_width_true = {}
            best_by_width_false = {}
            
            for row in reader:
                try:
                    width_val = row.get('width')
                    rollout_val = row.get('random_crop', '').lower()
                    if not width_val:
                        continue
                    
                    eval_str = row.get(evalutor, '')
                    if eval_str.lower() == 'nan' or eval_str == '':
                        continue
                    
                    test_error = float(eval_str)
                    
                    if rollout_val == 'false':
                        if width_val not in best_by_width_true:
                            best_by_width_true[width_val] = row
                        else:
                            current_best_error = float(best_by_width_true[width_val].get(evalutor, float('inf')))
                            if test_error < current_best_error:
                                best_by_width_true[width_val] = row
                    elif rollout_val == 'true':
                        if width_val not in best_by_width_false:
                            best_by_width_false[width_val] = row
                        else:
                            current_best_error = float(best_by_width_false[width_val].get(evalutor, float('inf')))
                            if test_error < current_best_error:
                                best_by_width_false[width_val] = row
                except (ValueError, TypeError):
                    continue
            
            for row in sorted(best_by_width_true.values(), key=lambda x: float(x['width'])):
                selected_row = {}
                for field in selected_fields:
                    value = row.get(field, '')
                    try:
                        if field in ['val_error', 'test_error', 'params']:
                            formatted_value = f"{float(value):.3f}"
                            selected_row[field] = formatted_value
                        elif field in ['width', 'modes', 'heads']:
                            try:
                                selected_row[field] = str(int(float(value)))
                            except:
                                selected_row[field] = value
                        else:
                            selected_row[field] = value
                    except (ValueError, TypeError):
                        selected_row[field] = value
                rollout_true_rows.append(selected_row)
            
            for row in sorted(best_by_width_false.values(), key=lambda x: float(x['width'])):
                selected_row = {}
                for field in selected_fields:
                    value = row.get(field, '')
                    try:
                        if field in ['val_error', 'test_error', 'params']:
                            formatted_value = f"{float(value):.3f}"
                            selected_row[field] = formatted_value
                        elif field in ['width', 'modes', 'heads']:
                            try:
                                selected_row[field] = str(int(float(value)))
                            except:
                                selected_row[field] = value
                        else:
                            selected_row[field] = value
                    except (ValueError, TypeError):
                        selected_row[field] = value
                rollout_false_rows.append(selected_row)
    
    if rollout_true_rows:
        output_path_true = os.path.join(log_dir, "best_results_rollout.csv")
        with open(output_path_true, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=selected_fields)
            writer.writeheader()
            writer.writerows(rollout_true_rows)
        
        print(f"Generated best_results_rollout.csv with {len(rollout_true_rows)} best records (rollout=True)")
    
    if rollout_false_rows:
        output_path_false = os.path.join(log_dir, "best_results_singleStep.csv")
        with open(output_path_false, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=selected_fields)
            writer.writeheader()
            writer.writerows(rollout_false_rows)
        
        print(f"Generated best_results_singleStep.csv with {len(rollout_false_rows)} best records (rollout=False)")
    
    if not rollout_true_rows and not rollout_false_rows:
        print("No valid records found in any files")
    else:
        total_records = len(rollout_true_rows) + len(rollout_false_rows)
        print(f"Total records processed: {total_records}")
        print(f"Skipped {skipped_deeponet_count} deeponet model files")
        print(f"Expected records: {(len(csv_files) - skipped_deeponet_count)} files x 3 widths x 2 rollout types = {(len(csv_files) - skipped_deeponet_count)*3 * 2} records")


def evaluate_model(data_path, experiment_name="hit", device='cuda', seed=0):
    best_results_path = f"{data_path}/best_results.csv"
    base_model_dir = f"{data_path}/log_model"

    setup_seed(seed)
    dataset = np.load(data_path + '/data/data.npz')
    varlist = dataset["variables"]
    coords = torch.tensor(dataset["coords"]).to(device)[None]
    
    coords = normalize_coords(coords)

    df = pd.read_csv(best_results_path)
    print(f"已加载 {len(df)} 个模型的最佳结果")

    models_evalutors = []
    all_train_losses = []
    all_val_errors = []
    model_names = []    
    for i, row in df.iterrows():
        args_dict = row.to_dict()
        
        if 'shape_list' in args_dict and isinstance(args_dict['shape_list'], str):
            args_dict['shape_list'] = ast.literal_eval(args_dict['shape_list'])
        
        if 'modes' in args_dict:
            value = args_dict['modes']
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                modes_list = value[1:-1].split(',')
                args_dict['modes'] = [int(m.strip()) for m in modes_list]
            elif isinstance(value, str):
                args_dict['modes'] = int(value)
            elif isinstance(value, list):
                args_dict['modes'] = [int(x) for x in value]
        
        args = Namespace(**args_dict)
        
        if i == 0:
            test_data = load_data(data_path, dataset, datatype="test", sub=args.sub)
            base = torch.arange(0, test_data.shape[1], dtype=torch.float32) / (test_data.shape[1] - 1) 
            dim = len(test_data.shape) - 3
            if dim == 3:
                time = (base[None, :, None, None, None].expand(1, test_data.shape[1], test_data.shape[-3], test_data.shape[-2], test_data.shape[-1]).contiguous())
            elif dim == 2:
                time = (base[None, :, None, None].expand(1, test_data.shape[1], test_data.shape[-2], test_data.shape[-1]).contiguous())
            else:
                time = (base[None, :, None].expand(1, test_data.shape[1], test_data.shape[-1]).contiguous())
            time = time.to(device)  
            
        model_dir = os.path.join(
            base_model_dir, 
            f"experiment_results_{args.experiment}_model_{args.model}"
        )
        model_path = os.path.join(model_dir, f"best_{args.model_name}.pt")
        
        print(f"\n处理模型 {i+1}/{len(df)}: {args.model_name}")
        print(f"模型路径: {model_path}")

        os.makedirs(f'{data_path}/bestModelResults', exist_ok=True)
        writer = SummaryWriter(f'{data_path}/bestModelResults/{args.model_name}')
                
        net = XDESolver(args).to(device)

        ckpt = torch.load(model_path)
        net.load_state_dict(ckpt['model_state'])
        net.set_normalizer(ckpt['mean'].to(device), ckpt['std'].to(device), ckpt['num_chemical'])
        net.to(device)

        raw_out_dir = os.path.join(data_path, "bestModelResults", args.model_name, "raw_npz")
        os.makedirs(raw_out_dir, exist_ok=True)

        num_chemical = ckpt['num_chemical']

        if dim == 3:
            if args.model == 'DeepONet_deepxde3d':
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModel_deeponet_3d(args, net, dataset, test_data, coords, time, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir)
            else:
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModel_3d(args, net, dataset, test_data, coords, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir)
        elif dim == 2:
            if args.model == 'DeepONet_deepxde':
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModel_deeponet(args, net, dataset, test_data, coords, time, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir)
            else:
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModel(args, net, dataset, test_data, coords, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir)
        else:
            if args.model == 'DeepONet_deepxdeU':
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModel_deeponetU(args, net, dataset, test_data, coords, time, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir)
            elif args.model in ['TransolverU', 'PointNet', 'LSM']:
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModelU(args, net, dataset, test_data, coords, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir) 
            else: 
                test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime = evalModelGraph(args, net, dataset, test_data, coords, num_chemical, writer, 0, True, varlist, save_raw=True, raw_out_dir=raw_out_dir) 
        
        params = param_flops(net)

        models_evalutor = {}
        models_evalutor['models'] = args.model
        models_evalutor['params'] = params
        models_evalutor['test_error'] = test_error
        models_evalutor['PSNR'] = PSNR
        models_evalutor['SSIM'] = SSIM
        models_evalutor['MSSSIM'] = MSSSIM
        models_evalutor['GMSD'] = GMSD
        models_evalutor['MSGMSD'] = MSGMSD
        models_evalutor['inferTime'] = inferTime
        models_evalutors.append(models_evalutor)
        
        loss_data = np.load(f"{data_path}/log_train_history/{args.model_name}_history.npz")
        train_loss_list = loss_data['train_loss'].tolist()
        val_error_list = loss_data['val_error'].tolist()
        
        all_train_losses.append(train_loss_list)
        all_val_errors.append(val_error_list)
        model_names.append(args.model)     
           
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(train_loss_list, color=color, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        color = 'red'
        ax2.set_ylabel('Validation Error', color=color)
        ax2.plot(val_error_list, color=color, label='Validation Error')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        plt.title(f'{args.model_name} - Training Loss & Validation Error')
        plt.grid(True)
        comparison_path = f"{data_path}/bestModelResults/{args.model}_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        # 显式释放资源
        del net
        if 'checkpoint' in locals():
            del checkpoint 
        
        writer.close()
        
        import gc
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


    plt.figure(figsize=(12, 10))
    
    # 绘制训练损失（对数坐标）
    plt.subplot(2, 1, 1)
    for i, train_loss in enumerate(all_train_losses):
        plt.plot(train_loss, label=f"{model_names[i]} (Train)")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss Comparison (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.yscale('log')  # 设置为对数坐标

    # 绘制验证误差（对数坐标）
    plt.subplot(2, 1, 2)
    for i, val_error in enumerate(all_val_errors):
        plt.plot(val_error, label=f"{model_names[i]} (Val)")
    plt.xlabel('Epoch')
    plt.ylabel('Error (log scale)')
    plt.title('Validation Error Comparison (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.yscale('log')  # 设置为对数坐标

    plt.tight_layout()
    comparison_path = f"{data_path}/bestModelResults/all_models_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    outputCSV_path = f"{data_path}/bestModelsEvaluator.csv"
    with open(outputCSV_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["models", "params(M)", "test_error", "PSNR", "SSIM", 'MSSSIM', 'GMSD', 'MSGMSD', "inferTime(s)"])
    
    for model in models_evalutors:
        model_name = model["models"]
        params = model["params"]
        test_error = model["test_error"]
        PSNR = model["PSNR"]
        SSIM = model["SSIM"]
        MSSSIM = model["MSSSIM"]
        GMSD = model["GMSD"]
        MSGMSD = model["MSGMSD"]
        inferTime = model["inferTime"]
        
        with open(outputCSV_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_name, params, test_error, PSNR, SSIM, MSSSIM, GMSD, MSGMSD, inferTime])
    
    print(f"✅ 所有模型结果已保存到: {outputCSV_path}")