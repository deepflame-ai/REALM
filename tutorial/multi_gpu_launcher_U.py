import os
import itertools
import subprocess
import time
from multiprocessing import Process, Queue

XDEBench_DIR = f'../XDEBench'

gpus = [3]  # modify your GPU ids here

# model_list = ['TransolverU']  
# batch_size_list = [4, 8]
# n_layers_list = [6, 8, 12]  
# width_list = [96, 128, 192]
# n_heads_list = [4, 8]  
# lr_list = [0.001, 0.01]
# modes_list = [0]

# model_list = ['PointNet']  
# batch_size_list = [4, 8]
# n_layers_list = [0]  
# width_list = [64, 96]
# n_heads_list = [0]  
# lr_list = [0.001]
# modes_list = [0]

# model_list = ['MGN']  
# batch_size_list = [1]
# n_layers_list = [5]  
# width_list = [64]
# n_heads_list = [0]  
# lr_list = [0.001]
# modes_list = [0]

# model_list = ['GAT']  
# batch_size_list = [1]
# n_layers_list = [0]  
# width_list = [32]
# n_heads_list = [2]  
# lr_list = [0.005]
# modes_list = [0]

# model_list = ['GraphSAGE']  
# batch_size_list = [1]
# n_layers_list = [4]  
# width_list = [96]
# n_heads_list = [0]  
# lr_list = [0.001, 0.01]
# modes_list = [0]

model_list = ['GraphUNet']  
batch_size_list = [1]
n_layers_list = [3]  
width_list = [64]
n_heads_list = [0]  
lr_list = [0.001]
modes_list = [0]

# model_list = ['LSM']  
# batch_size_list = [1]
# n_layers_list = [0]  
# width_list = [64]
# n_heads_list = [4]  
# lr_list = [0.001]
# modes_list = [32, 16]


# 构建所有组合（共 3*2*3*3 = 54 个任务）
grid = list(itertools.product(model_list, batch_size_list, modes_list, width_list, n_layers_list, n_heads_list, lr_list))

# 构建任务队列
task_queue = Queue()
for model, batch_size, modes, width, n_layers, n_heads, lr in grid:
    cmd = (
        f"python {XDEBench_DIR}/train_xdebench.py "
        f"--data_path /aisi-nas/baixuan/XDEBench_Data/U2dRocket3k "
        f"--experiment rocket "    
        f"--model {model} "
        f"--batch_size {batch_size} "
        f"--modes {modes} "
        f"--width {width} "
        f"--n_layers {n_layers} "
        f"--n_heads {n_heads} "
        f"--lr {lr} "
        f"--num_iterations 5000 "
        f"--optim_type adam "
        f"--device cuda"
    )
    task_queue.put(cmd)

        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/U2dCavity3w "
        # f"--experiment cavity "
        
        # f"--data_path /aisi-nas/baixuan/XDEBench_Data/U2dRocket3w "
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
