import os
import sys 

if __name__ == "__main__":

    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'XDEBench'))
    sys.path.insert(0, target_dir)

    from evaluator.utils import *
    
    data_path = "/aisi-nas/baixuan/XDEBench_FinalData/2dHIT"  # Change as needed
    experiment_name = "hit"  # Change as needed
    
    extract_best_results_rollout(data_path=data_path, experiment_name=experiment_name, evalutor="test_error")

    extract_best_results(data_path=data_path, experiment_name=experiment_name, evalutor="test_error")

    evaluate_model(data_path=data_path, experiment_name=experiment_name, device='cuda', seed=0)