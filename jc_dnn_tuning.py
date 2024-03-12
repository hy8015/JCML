import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import jc_torch_library as lib
from tqdm import tqdm
import optuna
import logging
import sys

datatype = 'sq_diff'
cond = 'cond2'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_iter = 500

def objective(trial):
    params = {
        'n_layers': trial.suggest_int('n_layers', 2, 7),
        'n_units': trial.suggest_int('n_units', 1300, 2400, step=100),
        'lr': trial.suggest_float('lr', 0.00005, 0.01, log=True),
        'act': trial.suggest_categorical('act', ['ReLU', 'GELU', 'LeakyReLU']),
        'loss': trial.suggest_categorical('loss', ['L1Loss', 'HuberLoss', 'SmoothL1Loss']),
        't': trial.suggest_int('t', 70, 230, step=10),
        't_mul': trial.suggest_int('t_mul', 5, 18),
        'batch_size': 1024
    }

    path = f'data/testset_size_30000_{cond}.npy'

    if datatype == 'sq':
        model = lib.DNN(N, params['n_units'], params['n_layers'], device, '', params['act'], load_param=False)
    else:
        model = lib.DNN(N-1, params['n_units'], params['n_layers'], device, '', params['act'], load_param=False)

    loss = train(trial, path, model, params)

    return loss

def validation_loss(dataloader, model):
    n = len(dataloader)
    running_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        model.eval()
        with tqdm(dataloader, unit='batch', ascii=True) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f'Validation')
                inputs, targets = data.to(device), target.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
    
    model.train()
    return running_loss / n

def train(trial, path, model, param):
    torch.manual_seed(0)
    trainloader, valloader = load_dataset(path, N, datatype, param['batch_size'])
    check_early = lib.EarlyStoppingCheck()

    if param['loss'] == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(beta=2.0)
    else:
        criterion = getattr(nn, param['loss'])()
    optimizer = optim.Adam(model.parameters(), lr=param['lr'], amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=param['t'], T_mult=param['t_mul'], eta_min=0)

    for epoch in range(max_iter):
        with tqdm(trainloader, unit='batch', ascii=True) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f'Epoch {epoch+1}')
                inputs, targets = data.to(device), target.to(device)

                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            scheduler.step()
           
        val_loss = validation_loss(valloader, model)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if check_early(val_loss):
            break

    return val_loss

def load_dataset(path, n_input, datatype, batch_size):
    dataset = lib.JCDataset(path, n_input, datatype)
    num_train = int(len(dataset) * 0.8)
    trainset, valset = random_split(dataset, [num_train, len(dataset) - num_train])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    return trainloader, valloader

def main():
    global N
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    for N in iter([12, 24, 48, 96]):
        while(True):
            print(f'{device} - N: {N} is started.')
            study_name = f'JC_dnn_{cond}_{datatype}_N_{N}'
            storage_name = "sqlite:///data/tune/{}.db".format(study_name)
            study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(max_resource=max_iter), load_if_exists=True)
            study.optimize(objective, n_trials=500, gc_after_trial=True, n_jobs=2)

            best_trial = study.best_trial

            if (study.trials[-1].number - 500) > best_trial.number:
                break

        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

if __name__ == '__main__':
    main()
