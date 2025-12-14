import os
import itertools
import subprocess
import time
from multiprocessing import Process, Queue

XDEBench_DIR = f'../XDEBench'

# 可用 GPU 列表
gpus = [1]  # 根据实际机器情况修改

# =========================================
# Detonation Best
# =========================================
# model_list = ['FactFormerv2']     # gpu1 - 8h
# batch_size_list = [4]         # 确认 batch_size <= num_train_traj
# width_list = [64]             # 可调
# n_layers_list = [3]            # 可调
# modes_list = [0]                  # 无用参数
# n_heads_list = [8]          # 可调    
# lr_list = [0.001]                 # 可调

# model_list = ['Transolver']       # gpu7 - 8h
# batch_size_list = [4]         # 确认 batch_size <= num_train_traj
# width_list = [64]       # 可调
# n_layers_list = [4]            # 可调 
# modes_list = [0]                  # 无用参数
# n_heads_list = [6]             # 可调 
# lr_list = [0.0005]         # 可调

# model_list = ['DPOT']             # gpu3 - 0.7h
# batch_size_list = [7]        # 确认 batch_size <= num_train_traj
# width_list = [512]     # 可调
# n_layers_list = [6]            # 可调
# modes_list = [64]             # 确认 modes <= img_size // 2
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]                 # 可调

# model_list = ['CROP']             # gpu6 - 2.5h
# batch_size_list = [7]     # 确认 batch_size <= num_train_traj
# width_list = [192]           # 可调
# n_layers_list = [0]               # 无用参数
# modes_list = [32]         # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]                 # 可调   

# model_list = ['FFNO']             # gpu2 - 4h
# batch_size_list = [7]     # 确认 batch_size <= num_train_traj
# width_list = [64]        # 可调
# n_layers_list = [3]            # 可调
# modes_list = [32]             # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.005]          # 可调

# model_list = ['GNOT']             # gpu5 - 4h
# batch_size_list = [1]          # 确认 batch_size <= num_train_traj
# width_list = [64]            # 可调
# n_layers_list = [4]            # 可调
# modes_list = [0]                  # 无用参数
# n_heads_list = [4]             # 确认 width // heads 能整除
# lr_list = [0.001]                 # 可调

model_list = ['CNextv2']   # gpu0 - 6.5h
batch_size_list = [7]      # 确认 batch_size <= num_train_traj
width_list = [96]       # 确认 width 至少能被2整除 n_layers+1 次
n_layers_list = [6]         # 确认 img_size 至少能被2整除 n_layers 次
modes_list = [0]                  # 无用参数
n_heads_list = [0]                # 无用参数
lr_list = [0.001]           # 可调

# model_list = ['UNOv2']            # gpu7 - 8h
# batch_size_list = [2]          # 确认 batch_size <= num_train_traj
# width_list = [64]            # 可调
# n_layers_list = [0]               # 无用参数
# modes_list = [32]             # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.005]          # 可调

# model_list = ['UNO']              # gpu4 - 8h
# batch_size_list = [7]     # 确认 batch_size <= num_train_traj
# width_list = [64]            # 可调
# n_layers_list = [0]               # 无用参数
# modes_list = [32]             # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.005]         # 可调

# model_list = ['FNO']              # gpu4 - 1.5h
# batch_size_list = [4, 7]         # 确认 batch_size <= num_train_traj
# width_list = [96]      # 可调
# n_layers_list = [0]               # 无用参数
# modes_list = [32, 16]         # 确认 modes <= img_size // 2
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]                 # 可调

# model_list = ['CNext']            # gpu3 - 10h
# batch_size_list = [4]         # 确认 batch_size <= num_train_traj
# n_layers_list = [6]         # 确认 img_size 至少能被2整除 n_layers 次
# width_list = [64]        # 确认 width 至少能被2整除 n_layers+1 次
# modes_list = [0]                  # 无用参数
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]          # 可调

# model_list = ['FactFormer']       # gpu5 - 2h
# batch_size_list = [4]     # 确认 batch_size <= num_train_traj
# width_list = [64]            # 可调
# n_layers_list = [4]            # 可调
# modes_list = [0]                  # 无用参数
# n_heads_list = [4]             # 确认 width // heads 能整除
# lr_list = [0.001]          # 可调

# model_list = ['ONO']              # gpu5 - 3h
# batch_size_list = [4]          # 确认 batch_size <= num_train_traj
# width_list = [96]            # 可调
# n_layers_list = [3]            # 可调   
# modes_list = [0]                  # 无用参数
# n_heads_list = [8]             # 可调
# lr_list = [0.001]           # 可调

####################################################
####################################################

# 构建所有组合（共 3*2*3*3 = 54 个任务）
grid = list(itertools.product(model_list, batch_size_list, modes_list, width_list, n_layers_list, n_heads_list, lr_list))

# 构建任务队列
task_queue = Queue()
for model, batch_size, modes, width, n_layers, n_heads, lr in grid:
    cmd = (
        f"python {XDEBench_DIR}/train_xdebench_rollout.py "
        f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dDetonation "
        f"--experiment detonation "   
        f"--model {model} "
        f"--batch_size {batch_size} "
        f"--modes {modes} "
        f"--width {width} "
        f"--n_layers {n_layers} "
        f"--n_heads {n_heads} "
        f"--lr {lr} "
        f"--num_iterations 20000 "
        f"--device cuda"
    )
    task_queue.put(cmd)

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dEvojet "
        # f"--experiment evojet "

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dDetonation "
        # f"--experiment detonation "

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dHIT "
        # f"--experiment hit " 

# 每个 GPU 跑一个任务，不断从队列中取任务运行
def gpu_worker(gpu_id, task_queue):
    while not task_queue.empty():
        try:
            cmd = task_queue.get_nowait()
        except:
            break
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🚀 [GPU {gpu_id}] {cmd}")
        subprocess.run(cmd, shell=True, env=env)
        time.sleep(1)  # 可选：避免爆发式调度
    print(f"✅ GPU {gpu_id} 完成所有任务")

if __name__ == "__main__":
    print(f"Total jobs: {task_queue.qsize()} using {len(gpus)} GPUs")
    workers = []
    for gpu in gpus:
        p = Process(target=gpu_worker, args=(gpu, task_queue))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()
    print("🎉 All tasks completed.")
