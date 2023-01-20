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

Toy model code consists of two files, `toy_model_run.py` and `toy_model_script.py` found in `code/toy_model/` directory. The former is required to construct a single phase diagram, while the latter is a script for multiple diagrams. The notebook `toy_model_plotting.ipynb` contains code fragments for plotting the diagrams. To launch the experiments, follow this list of commands in shell:

```shell
cd code/toy_model
python toy_model_script.py
```

This will create a directory `code/toy_model/experiments` saving all necessary data for reproducing the figures from the report. 

## Running MNIST experiments

TODO
