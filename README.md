# Reproducibility of "Towards Understanding Grokking"

## Contents

* `code/` &mdash; contains all necessary code for reproducing experiments
    * `code/mnist/` &mdash; MNIST experiments code
    * `code/toy_model/` &mdash; toy model experiments code
* `images/` &mdash; contains all figures from the submitted pdf report

## Python environment

The code from this repo required the following list of libraries:
```requirements
torch
torchvision
numpy
pandas
sklearn
joblib
tqdm
matplotlib
seaborn
```

## Running toy model experiments

The toy model code consists of two files, `toy_model_run.py` and `toy_model_script.py` found in `code/toy_model/` directory. The former is required to construct a single phase diagram, while the latter is a script for multiple diagrams. The notebook `toy_model_plotting.ipynb` contains code fragments for plotting the diagrams. To launch the experiments, follow this list of commands in shell:

```shell
cd code/toy_model
python toy_model_script.py
```

This will create a directory `code/toy_model/experiments/` saving all necessary data for reproducing the figures from the report. 

## Running MNIST experiments

Code for MNIST experiments is stored in `code/mnist` folder. `mnist_run.py` file is intended for launching training on MNIST for multiple weight decays and learning rates. `original_setup.py` in a launching file that contains all hyperparameters for setup proposed by authors, while `extended_setup.py` stores hyperparameters for our larger setup. The notebook `mnist_plotting.ipynb` contains the code for visualization of results from this work. To launch the experiments, follow this list of commands in shell:

```shell
cd code/mnist
python original_setup.py
python extended_setup.py
```

This code will generate two `.csv` files for original and extended setups respectively. The results can be found in `mnist_results` folder.

