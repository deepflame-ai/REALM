import os
import itertools
import subprocess
import time
from multiprocessing import Process, Queue

XDEBench_DIR = f'../XDEBench'

# 可用 GPU 列表
gpus = [4]  # 根据实际机器情况修改

# =========================================
### HIT best
# model_list = ['DeepONet_deepxde'] # gpu - 3h
# batch_size_list = [16]     # 确认 batch_size <= num_train_traj
# width_list = [256]       # 可调
# n_layers_list = [4]            # 可调
# modes_list = [256]         # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]                 # 可调

# =========================================
### Evojet best
# model_list = ['DeepONet_deepxde'] # gpu - 3h
# batch_size_list = [8]     # 确认 batch_size <= num_train_traj
# width_list = [256]       # 可调
# n_layers_list = [4]            # 可调
# modes_list = [96]         # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]                 # 可调

# =========================================
### Detonation best
# model_list = ['DeepONet_deepxde'] # gpu - 3h
# batch_size_list = [7]     # 确认 batch_size <= num_train_traj
# width_list = [128]       # 可调
# n_layers_list = [4]            # 可调
# modes_list = [64]         # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001]                 # 可调
# =========================================

# =========================================
### Poolfire best
model_list = ['DeepONet_deepxde3d'] # gpu - 3h
batch_size_list = [1]     # 确认 batch_size <= num_train_traj
width_list = [128]       # 可调
n_layers_list = [3]            # 可调
modes_list = [64]         # 可调
n_heads_list = [0]                # 无用参数
lr_list = [0.001]                 # 可调

# =========================================

# model_list = ['DeepONet_deepxdeU'] # gpu - 3h
# batch_size_list = [4, 8]     # 确认 batch_size <= num_train_traj
# width_list = [64, 128, 256]       # 可调
# n_layers_list = [3, 5]            # 可调
# modes_list = [32, 64]         # 可调
# n_heads_list = [0]                # 无用参数
# lr_list = [0.001, 0.01]                 # 可调

# 构建所有组合（共 3*2*3*3 = 54 个任务）
grid = list(itertools.product(model_list, batch_size_list, modes_list, width_list, n_layers_list, n_heads_list, lr_list))

# 构建任务队列
task_queue = Queue()
for model, batch_size, modes, width, n_layers, n_heads, lr in grid:
    cmd = (
        f"python {XDEBench_DIR}/train_deeponet.py "
        f"--data_path /aisi-nas/baixuan/XDEBench_Data/3dTGV128 "
        f"--experiment TGV "
        f"--model {model} "
        f"--batch_size {batch_size} "
        f"--modes {modes} "
        f"--width {width} "
        f"--n_layers {n_layers} "
        f"--n_heads {n_heads} "
        f"--lr {lr} "
        f"--num_iterations 10000 "
        f"--device cuda"
    )
    task_queue.put(cmd)

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dEvojet "
        # f"--experiment evojet "

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dDetonation "
        # f"--experiment detonation "

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/2dHIT "
        # f"--experiment hit "

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/3dPoolfire "
        # f"--experiment poolfire "
        
        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/3dTGV128 "
        # f"--experiment TGV "        
        
        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/U2dCavity30w "
        # f"--experiment cavity "      
        
        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/U2dRocket "
        # f"--experiment rocket "          

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
