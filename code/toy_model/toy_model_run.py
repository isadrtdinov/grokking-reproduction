import os
import random
import shutil
import argparse

import torch
import numpy as np
from torch import nn
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from itertools import product
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs', type=int, default=8,
                    help="Number of parallel processing jobs")
parser.add_argument('--save_dir', type=str, required=True,
                    help="Path to save experimental results")
parser.add_argument('--task', type=str, default='cls',
                    help="Task to solve, either \'cls\' for classification "
                         "or \'regr\' regression")
parser.add_argument('--activation', type=str, default='tanh',
                    help="Activation function in MLP decoder, "
                         "available option: \'tanh\', \'relu\', \'leaky_relu\'")
parser.add_argument('--max_wd', type=float, default=20.0,
                    help="Maximum value of decoder weight decay")
parser.add_argument('--num_vals', type=int, default=21,
                    help="Number of values in the search grid")
parser.add_argument('--max_iters', type=int, default=10 ** 5,
                    help="Maximum number of optimization iterations")
parser.add_argument('--data_seed', type=int, default=42,
                    help="Train-validation split random seed")
parser.add_argument('--model_seed', type=int, default=100,
                    help="Model initialization random seed")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Whether to verbose information during running")
args = parser.parse_args()
device = torch.device('cpu')


class ToyModel(nn.Module):
    act_class = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}
    act_kwargs = {'leaky_relu': {'negative_slope': 0.1}}

    def __init__(self, p: int, embed_dim: int, mlp_dims: Tuple[int, ...],
                 embed_scale: float = 1., activation='relu'):
        """
        Initialize toy model.
        :param p: number of embeddings
        :param embed_dim: dimensionality of embeddings
        :param mlp_dims: inner dimensionality of MLP decoder
        :param embed_scale: embeddings initialization scale
        :param activation: activation function in MLP decoder
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=p, embedding_dim=embed_dim)
        torch.nn.init.uniform_(self.embedding.weight, a=-embed_scale / 2, b=embed_scale / 2)
        self.decoder = [nn.Linear(embed_dim, mlp_dims[0])]
        for i in range(1, len(mlp_dims)):
            self.decoder += [
                self.act_class[activation](**self.act_kwargs.get(activation, {})),
                nn.Linear(mlp_dims[i - 1], mlp_dims[i])
            ]
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the toy model.
        :param x:
        :return: logits for classification or output vectors for regression
        """
        embeds = self.embedding(x)
        output = self.decoder(embeds.sum(dim=1))
        return output


def generate_additive_data(p: int, sizes: Tuple[int, int], target_dim: int = 30, seed: Optional[int] = None):
    """
    Generate data for addition toy setup,
    split to train and validation,
    and convert to torch tensors.
    :param p: number of unique integers in range [0, ..., p - 1]
    :param sizes: tuple containing train size and validation size
    :param target_dim: dimensionality of target vectors (for regression)
    :param seed: data generation random seed
    :return: (x_train, x_test, y_train, y_test, target_mat)
        x_train: torch.Tensor with a training set of size (sizes[0], 2)
        x_test: torch.Tensor with a validation set of size (sizes[1], 2)
        y_train: torch.Tensor with training targets of size (sizes[0], )
        y_test: torch.Tensor with validation targets of size (sizes[0], )
        target_mat: torch.Tensor with target vectors of size (2p - 1, target_dim)
    """
    set_random_seed(seed)
    a = np.arange(p)
    b = np.arange(p)
    a, b = np.meshgrid(a, b)
    mask = a <= b
    x = np.stack([a[mask], b[mask]], axis=1)
    y = x[:, 0] + x[:, 1]
    x, y = torch.tensor(x), torch.tensor(y)

    total = len(y)
    assert sizes[0] + sizes[1] == total
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, train_size=sizes[0] / total, random_state=seed)
    target_mat = torch.randn(size=(y.max() + 1, target_dim))

    return x_train, x_test, y_train, y_test, target_mat


def set_random_seed(seed: int):
    """
    Seed everything
    :param seed: random seed to fix
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_setup(data: Tuple[torch.Tensor, ...], kind: str = 'regr',
                emb_lr: float = 1e-3, emb_wd: float = 0, dec_lr: float = 1e-3, dec_wd: float = 0,
                num_iters: int = 10 ** 4, model_seed: int = 100, activation: str = 'relu',
                acc_thr: float = 0.9, grokking_diff: int = 10 ** 3, embed_log_freq: int = 10):
    """
    Train toy model and determine the phase
    :param data: data tuple outputted by `generate_additive_data` function
    :param kind: which task to solve (either \'cls\' or \'regr\')
    :param emb_lr: embeddings encoder learning rate
    :param emb_wd: embeddings encoder weight decay
    :param dec_lr: decoder MLP learning rate
    :param dec_wd: decoder MLP weight decay
    :param num_iters: maximum number of training iterations
    :param model_seed: model initialization random seed
    :param activation: decoder MLP activation function
    :param acc_thr: convergence accuracy threshold
    :param grokking_diff: iterations difference for the phase to be considered grokking
    :param embed_log_freq: frequency of embeddings logging
    """
    # Set random seed
    set_random_seed(model_seed)

    # Unpack datasets
    x_train, x_test, y_train, y_test, target_mat = data
    x_train, x_test = x_train.to(device), x_test.to(device)
    target_mat = target_mat.to(device)

    # Set up loss functions
    if kind == 'regr':
        criterion = nn.MSELoss()
        target = target_mat[y_train]
    elif kind == 'cls':
        criterion = nn.CrossEntropyLoss()
        target = y_train.to(device)
    else:
        raise ValueError('Unknown task')

    # Initialize model and optimizer
    y_train, y_test = y_train.to(device), y_test.to(device)
    num_outputs = 30 if kind == 'regr' else 19
    model = ToyModel(p=10, embed_dim=1, mlp_dims=(200, 200, num_outputs), activation=activation).to(device)

    opt = torch.optim.AdamW([
        {'params': model.embedding.parameters(), 'lr': emb_lr, 'weight_decay': 0},
        {'params': model.decoder.parameters(), 'lr': dec_lr, 'weight_decay': dec_wd}
    ], lr=emb_lr)

    # Prepare logging lists
    embeds_history = [torch.clone(model.embedding.weight).detach().cpu()]
    train_accs, test_accs = [], []
    train_iter, test_iter = None, None

    def save_state(phase: str, it: int):
        """
        A function to save experimental results.
        :param phase: observed phase
        :param it: training interation
        """
        if args.verbose:
            print(f'Finished lr={dec_lr:.3e}, wd={dec_wd:.3e}, phase={phase}')

        torch.save({
            'train_acc': train_accs, 'test_acc': test_accs,
            'train_iter': train_iter, 'test_iter': test_iter,
            'embeds_history': torch.stack(embeds_history, dim=0),
            'kind': kind, 'iter': it, 'phase': phase, 'emb_lr': emb_lr, 'emb_wd': emb_wd,
            'dec_lr': dec_lr, 'dec_wd': dec_wd, 'model_seed': model_seed,
            'num_iters': num_iters, 'grokking_diff': grokking_diff,
            'acc_thr': acc_thr, 'activation': activation,
        }, os.path.join(args.save_dir, f'lr={dec_lr:.3e}-wd={dec_wd:.3e}.state'))
        return None

    # Main training loop
    for i in range(1, num_iters + 1):
        # Do forward pass and calculate loss
        model.train()
        opt.zero_grad()

        output = model(x_train)
        loss = criterion(output, target)
        loss.backward()
        opt.step()

        # Generate predictions for both datasets
        model.eval()
        with torch.no_grad():
            if kind == 'regr':
                # Regression: class with the closest target vector
                train_preds = torch.cdist(output, target_mat).argmin(dim=1)
                test_preds = torch.cdist(model(x_test), target_mat).argmin(dim=1)
            else:
                # Classification: class with the greatest logit
                train_preds = output.argmax(dim=1)
                test_preds = model(x_test).argmax(dim=1)

            train_acc = (train_preds == y_train).to(torch.float).mean().item()
            test_acc = (test_preds == y_test).to(torch.float).mean().item()

        train_accs += [train_acc]
        test_accs += [test_acc]

        # Record embeddings state
        if i % embed_log_freq == 0:
            embeds_history += [torch.clone(model.embedding.weight).detach().cpu()]

        # Check the accuracy threshold for both datasets
        if train_iter is None and train_acc >= acc_thr:
            train_iter = i

        if test_iter is None and test_acc >= acc_thr:
            test_iter = i

        # If generalized to the validation set
        if train_iter and test_iter:
            # Difference of iters and grokking iteration threshold => memorization
            if test_iter - train_iter < grokking_diff:
                return save_state('comprehension', i)

            # Otherwise => grokking
            return save_state('grokking', i)

    if not train_iter:
        # Not converged on the training set => confusion
        return save_state('confusion', num_iters)

    # Converged on the training set => memorization
    return save_state('memorization', num_iters)


# Generate toy data and set decoder learning rate and weight decay ranges
data = generate_additive_data(p=10, sizes=(45, 10), target_dim=30, seed=args.data_seed)
lrs = np.logspace(-5, -2, args.num_vals)
wds = np.linspace(0, args.max_wd, args.num_vals)
prod = list(product(lrs, wds))

# Create experiments dir
if os.path.isdir(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

# Save data state
torch.save({
    'x_train': data[0], 'y_train': data[2],
    'x_test': data[1], 'y_test': data[3],
    'target_mat': data[4], 'data_seed': args.data_seed
}, os.path.join(args.save_dir, 'data.state'))

# Launch grid training in parallel
Parallel(n_jobs=args.n_jobs)(
    delayed(train_setup)(
        data, kind=args.task, dec_lr=lr, dec_wd=wd,
        num_iters=args.max_iters, model_seed=args.model_seed,
        acc_thr=0.9 - 1e-3,  # threshold equal to 0.9 but subtract eps=1e-3 for floating point numbers
    ) for lr, wd in prod
)
