import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import gc
import jc_torch_library as lib
import os
from tqdm import tqdm
import optuna
import numpy as np

torch.manual_seed(0)
th_epoch = 1000

def validation_loss(dataloader, model, device):
    n = len(dataloader)
    running_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        model.eval()
        with tqdm(dataloader, unit='batch', ascii=True) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f'Validation')
                inputs, targets = data.to(device), target.to(device)
                loss = criterion(model(inputs), targets)
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
    
    model.train()
    return running_loss / n

def train(device, param, trainloader, valloader, N, cond, datatype):
    if datatype == 'sq':
        model = lib.DNN(N, param['n_units'], param['n_layers'], device, '', param['act'], load_param=False)
    else:
        model = lib.DNN(N-1, param['n_units'], param['n_layers'], device, '', param['act'], load_param=False)

    PATH = f'data/dnn/{cond}/{datatype}/'
    os.makedirs(PATH, exist_ok=True)
    if param['loss'] == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(beta=2.0)
    else:
        criterion = getattr(nn, param['loss'])()
    optimizer = optim.Adam(model.parameters(), lr=param['lr'], amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=param['t'], T_mult=param['t_mul'], eta_min=0)

    train_loss_list = []
    val_loss_list = []
    n = len(trainloader)
    early_stopping_loss = 1.
    early_stopping_epoch = 0
    early_stopping_train_loss = 0.
    early_stopping_val_loss = 0.
    cnt = 0

    for epoch in range(th_epoch):
        running_loss = 0.0
        with tqdm(trainloader, unit='batch', ascii=True) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f'Epoch {epoch+1}')
                inputs, targets = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            scheduler.step()
        train_loss = running_loss / n
        train_loss_list.append(train_loss)
        val_loss = validation_loss(valloader, model, device)
        val_loss_list.append(val_loss)
        cnt += 1

        if val_loss < early_stopping_loss:
            torch.save(model.state_dict(), f'{PATH}N_{N}_model.pth')
            early_stopping_train_loss = train_loss
            early_stopping_val_loss = val_loss
            early_stopping_epoch = epoch
            early_stopping_loss = val_loss
            cnt = 0
        
        if cnt == 10:
            break

    print('Final pretrained model >> [%d] train loss: %.5f, validation loss: %.5f'%(early_stopping_epoch+1, early_stopping_train_loss, early_stopping_val_loss))
    with open(f'{PATH}N_{N}_train_results.txt', mode='w') as f:
        f.write(f'{param}\n')
        f.write('Final pretrained model >>\n[%d] train loss: %.5f, validation loss: %.5f\n'%(early_stopping_epoch+1, early_stopping_train_loss, early_stopping_val_loss))
        f.write('Train losses:\n')
        f.write(f'{train_loss_list}\n')
        np.save(f'{PATH}N_{N}_train_loss_list.npy', train_loss_list)
        f.write('Validation losses:\n')
        f.write(f'{val_loss_list}')
        np.save(f'{PATH}N_{N}_val_loss_list.npy', val_loss_list)

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss_list, color='r', label='Train')
    ax1.set_ylabel('Train Loss')
    ax2 = ax1.twinx()
    ax2.plot(val_loss_list, color='b', label='Validation')
    ax2.set_ylabel('Validation Loss')
    fig.legend()
    plt.title('Loss')
    ax1.set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(f'{PATH}N_{N}_train_history.png')
    plt.close('all')

def load_dataset(path, n_input, batch_size, datatype):
    dataset = lib.JCDataset(path, n_input, datatype)
    num_train = int(len(dataset) * 0.8)
    trainset, valset = random_split(dataset, [num_train, len(dataset) - num_train])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    return trainloader, valloader

def main():
    cond = 'cond2'
    path = f'data/trainset_size_20000000_{cond}.npy'
    batch_size = 1024
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available.')
    datatype = 'sq_diff'

    for N in iter([12, 24, 48, 96]):
        study_name = f'JC_dnn_{cond}_{datatype}_N_{N}'
        storage_name = "sqlite:///data/tune/{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(), load_if_exists=True)
    
        params = study.best_trial.params

        trainloader, valloader = load_dataset(path, N, batch_size, datatype)
        print(f'N: {N}, condition: {cond}, datatype: {datatype}, {params}')
        train(device, params, trainloader, valloader, N, cond, datatype)
        print('Completed.')

        del trainloader, valloader
        gc.collect()

if __name__ == '__main__':
    main()
