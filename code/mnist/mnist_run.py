import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision
import random
import pandas as pd
import argparse

from tqdm.auto import tqdm


# define model's architecture
class MNIST_MLP(nn.Module):
    def __init__(self, n_classes=10, dim=200):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def train(model, optimizer, dataloader):
    acc = []
    for batch in dataloader:
        image, label = batch
        image, label = image.to(device), label.to(device)
        logits = model(image)
        
        # use MSE loss with one-hot targets instead of Cross-Entropy
        target = F.one_hot(label, 10).type(dtype)
        loss = F.mse_loss(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(-1) == label).detach().cpu()
        acc.extend(accuracy)
        
    return np.mean(acc)


def evaluate(model, dataloader):
    acc = []
    for batch in dataloader:
        image, label = batch
        image, label = image.to(device), label.to(device)
        logits = model(image)
        
        # use MSE loss with one-hot targets instead of Cross-Entropy
        target = F.one_hot(label, 10).type(dtype)
        loss = F.mse_loss(logits, target)

        accuracy = (logits.argmax(-1) == label).detach().cpu()
        acc.extend(accuracy)

    return np.mean(acc)
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
seed = 0


def train_model(lr, wd, max_iters=10**5, acc_th=0.6, batch_size=200, model_scale=9., early_stop=True):
    # fix random seed
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # load dataset
    train_data = torchvision.datasets.MNIST(root='./data', train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, 
        transform=torchvision.transforms.ToTensor(), download=True)

    # take random sample of size 1000 from training dataset
    idxs = np.random.choice(range(len(train_data)), 1000, replace=False)
    train_data = torch.utils.data.Subset(train_data, idxs)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # create model
    mlp = MNIST_MLP().to(device)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = model_scale * p.data  # scale initial weights

    # weight decay is common for both encoder and decoder,
    # learning rate for encoder is fixed
    optimizer = torch.optim.AdamW([
            {'params': mlp.encoder.parameters(), 'lr': 1e-3},
            {'params': mlp.decoder.parameters(), 'lr': lr}
         ], weight_decay=wd
    )

    iters_per_epoch = len(train_data) // batch_size

    # logarithmic grid that defines evaluation iterations to reduce the total training time
    log_steps = np.unique([int(step) for step in np.logspace(0, np.log10(max_iters), 1000) / iters_per_epoch])
    
    train_accuracies, test_accuracies = [], []
    train_converge_step, test_converge_step = 0, max_iters
    steps = 0
    phase = None
    for epoch in tqdm(range(max_iters // iters_per_epoch)):
        train_acc = train(mlp, optimizer, train_loader)
        steps += iters_per_epoch

        if epoch in log_steps:
            test_acc = evaluate(mlp, test_loader)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # save the first iteration, when accuracy on the training set reached accuracy threshold
            if train_acc > acc_th and train_converge_step == 0:
                train_converge_step = steps

            # save the first iteration, when accuracy on the validation set reached accuracy threshold
            if test_acc > acc_th and test_converge_step == max_iters:
                test_converge_step = steps

            if phase is None and train_converge_step != 0 and test_converge_step != max_iters:
                if steps - train_converge_step > 1e+3:
                    phase = 'grokking'
                else:
                    phase = 'comprehension'

                if early_stop:
                    return phase, (train_converge_step, test_converge_step), (train_accuracies, test_accuracies)

    if phase is None:
        if train_converge_step > 0:
            phase = 'memorization'
        else:
            phase = 'confusion'

    return phase, (train_converge_step, test_converge_step), (train_accuracies, test_accuracies)


if __name__ == '__main__':
    # parse training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='results.csv')
    parser.add_argument('--min_lr_log', type=float, default=-6)
    parser.add_argument('--max_lr_log', type=float, default=0)
    parser.add_argument('--min_wd_log', type=float, default=-5)
    parser.add_argument('--max_wd_log', type=float, default=1)
    parser.add_argument('--num_lr_vals', type=int, default=10)
    parser.add_argument('--num_wd_vals', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=10 ** 5)
    args = parser.parse_args()
    
    
    results = pd.DataFrame()
    results = pd.read_csv(args.save_path, index_col='wd')

    lrs = np.logspace(args.min_lr_log, args.max_lr_log, args.num_lr_vals)
    wds = np.logspace(args.min_wd_log, args.max_wd_log, args.num_wd_vals)

    for lr in lrs:
        for wd in wds:
            _, convergence_steps, _ = train_model(lr, wd, max_iters=args.max_iters)

            # log the results of the training process 
            results = pd.read_csv(args.save_path, index_col='wd')
            results.loc[wd, str(lr)] = str(convergence_steps)
            results.to_csv(args.save_path, index_label='wd')
