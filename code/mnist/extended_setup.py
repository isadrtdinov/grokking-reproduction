import os

min_lr_log = -8.888
max_lr_log = 2.666
min_wd_log = -9
max_wd_log = 3
num_lr_vals = 15
num_wd_vals = 14
max_iters = 1.5 * 10**5

save_path = f'mnist_results/results_extended.csv'

if not os.path.isdir('mnist_results'):
    os.makedirs('mnist_results')

os.system(
    f'python3 mnist_run.py --min_lr_log {min_lr_log} --max_lr_log {max_lr_log}'
    f'--min_wd_log {min_wd_log} --max_wd_log {max_wd_log} --num_lr_vals {num_lr_vals} 
    f'--num_wd_vals {num_wd_vals} --max_iters {max_iters}'
)
