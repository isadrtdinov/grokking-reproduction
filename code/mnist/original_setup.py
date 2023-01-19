import os

min_lr_log = -6
max_lr_log = 0
min_wd_log = -5
max_wd_log = 1
num_lr_vals = 4
num_wd_vals = 4
max_iters = 10**5

save_path = f'mnist_results/results_original.csv'

if not os.path.isdir('mnist_results'):
    os.makedirs('mnist_results')

os.system(
    f'python3 mnist_run.py --min_lr_log {min_lr_log} --max_lr_log {max_lr_log} '
    f'--min_wd_log {min_wd_log} --max_wd_log {max_wd_log} --num_lr_vals {num_lr_vals} '
    f'--num_wd_vals {num_wd_vals} --max_iters {max_iters}'
)
