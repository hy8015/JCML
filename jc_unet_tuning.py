import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import jc_torch_library as lib
from tqdm import tqdm
import optuna
import logging
import sys

cond = 'cond2'
max_iter = 1000
datatype = 'sq_diff'
mean = 0.28   # cond2, sq_diff

def objective(trial):
    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 5, step=2),
        'lr': trial.suggest_float('lr', 1e-7, 0.001, log=True),
        'nodes': trial.suggest_int('nodes', 1200, 2800, step=100),
        'multiple': trial.suggest_float('multiple', 1.5, 2.0, step=0.5),
        'act': trial.suggest_categorical('act', ['ReLU', 'Hardswish', 'GELU', 'LeakyReLU']),
        'loss': trial.suggest_categorical('loss', ['L1Loss', 'MSELoss', 'HuberLoss', 'SmoothL1Loss']),
        't': trial.suggest_int('t', 20, 220, step=10),
        't_mul': trial.suggest_int('t_mul', 4, 20),
        'alpha': trial.suggest_categorical('alpha', [0, 0.1, 0.2]),
        'batch_size': 1024,
        'eps': 0.5
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = f'data/testset_size_30000_{cond}.npy'
    n_units = lib.cal_units_dae(params['n_layers'], params['nodes'], params['multiple'])
    if params['n_layers'] == 1:
        model = lib.DenoisingUNet1(N, n_units, device, params['act'], params['alpha'])
    elif params['n_layers'] == 3:
        model = lib.DenoisingUNet3(N, n_units, device, params['act'], params['alpha'])
    elif params['n_layers'] == 5:
        model = lib.DenoisingUNet5(N, n_units, device, params['act'], params['alpha'])
    elif params['n_layers'] == 7:
        model = lib.DenoisingUNet7(N, n_units, device, params['act'], params['alpha'])

    loss = train(trial, path, device, model, params)

    return loss

def validation_loss(dataloader, model, device, eps):
    n = len(dataloader)
    running_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        model.eval()
        with tqdm(dataloader, unit='batch', ascii=True) as tepoch:
            for data, _ in tepoch:
                tepoch.set_description(f'Validation')
                noise = torch.normal(0, mean/3, data.shape)
                noise_inputs = data + eps * noise
                inputs = data.to(device)
                noise_inputs = noise_inputs.to(device)
                outputs = model(noise_inputs)
                loss = criterion(outputs, inputs)
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
    
    model.train()
    return running_loss / n

def train(trial, path, device, model, param):
    torch.manual_seed(0)
    if 'diff' in datatype:
        trainloader, valloader = load_dataset(path, N+1, param['batch_size'], datatype)
    else:
        trainloader, valloader = load_dataset(path, N, param['batch_size'], datatype)
    check_early = lib.EarlyStoppingCheck()

    if param['loss'] == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(beta=2.0)
    else:
        criterion = getattr(nn, param['loss'])()
    optimizer = optim.Adam(model.parameters(), lr=param['lr'], amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=param['t'], T_mult=param['t_mul'], eta_min=0)

    cnt = 0
    for epoch in range(max_iter):
        with tqdm(trainloader, unit='batch', ascii=True) as tepoch:
            for data, _ in tepoch:
                tepoch.set_description(f'Epoch {epoch+1}')
                noise = torch.normal(0, mean/3, data.shape)
                noise_inputs = data + param['eps'] * noise
                inputs = data.to(device)
                noise_inputs = noise_inputs.to(device)
                optimizer.zero_grad()
                outputs = model(noise_inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            scheduler.step()
        val_loss = validation_loss(valloader, model, device, param['eps'])
        
        trial.report(val_loss, epoch)

        cnt += 1
        if val_loss < 1.:
            cnt = 0

        if trial.should_prune() or cnt == 10:
            raise optuna.exceptions.TrialPruned()
        
        if check_early(val_loss):
            break

    return val_loss

def load_dataset(path, n_input, batch_size, datatype):
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
            print(f'N: {N} is started.')
            study_name = f'JC_unet_{cond}_{datatype}_N_{N}'
            storage_name = "sqlite:///data/tune/{}.db".format(study_name)
            study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(max_resource=max_iter), load_if_exists=True)
            study.optimize(objective, n_trials=1000, gc_after_trial=True, n_jobs=2)

            best_trial = study.best_trial

            if (study.trials[-1].number - 1000) > best_trial.number:
                break

        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

if __name__ == '__main__':
    main()
