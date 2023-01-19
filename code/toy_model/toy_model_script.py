import os
"""
Main script for launching different tasks, activation functions and random seeds
"""

task = 'cls'
activation = 'tanh'
max_wd = 20 if task == 'cls' else 10
data_seeds = [41, 42, 43]
model_seeds = [101, 102, 103]

for data_seed in data_seeds:
    for model_seed in model_seeds:
        save_dir = f'experiments/{task}_{activation}_data{data_seed}_model{model_seed}'
        os.system(
            f'python3 toy_model_run.py --data_seed {data_seed} --model_seed {model_seed} '
            f'--task {task} --max_wd {max_wd} --save_dir {save_dir} --activation {activation}'
        )
        print(f'Finished fitting {save_dir}')
