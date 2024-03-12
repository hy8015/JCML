from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class JCDataset(Dataset):
    def __init__(self, path, n_input, datatype):
        data = np.load(path)
        x_data = data[:,3:3+n_input]
        self.y_data = data[:,:3]
        self.len = len(data)
        self.x_data = preprocessing(datatype, n_input, x_data)

    def __getitem__(self, index):
        x_data = torch.FloatTensor(self.x_data[index])
        y_data = torch.FloatTensor(self.y_data[index])
        
        return x_data, y_data

    def __len__(self):
        return self.len

def preprocessing(datatype, n_input, x_data):
    if datatype == 'diff_1':
        x_data -= x_data[:, 0].reshape((-1, 1))
        x_data = x_data[:,1:]
    elif datatype == 'diff_2':
        x_len = n_input
        x_data_ = np.zeros_like(x_data)

        for i in range(1, x_len):
            x_data_[:,i] = x_data[:, i] - x_data[:, i-1]
            
        x_data = x_data_[:,1:]
    elif datatype == 'sq':
        x_data = np.power(x_data, 2)
    elif datatype == 'sq_diff':
        x_data = np.power(x_data, 2)
        
        x_len = n_input
        x_data_ = np.zeros_like(x_data)

        for i in range(1, x_len):
            x_data_[:,i] = x_data[:, i] - x_data[:, i-1]
            
        x_data = x_data_[:,1:]
    
    return x_data

class EarlyStoppingCheck():
    def __init__(self) -> None:
        self.cnt = 0
        self.th = 1.
    def __call__(self, val_loss):
        if self.th > val_loss:
            self.th = val_loss
            self.cnt = 0
        else:
            self.cnt += 1
        
        if self.cnt == 10:
            return True
        else:
            return False

def DNN(N, n_units, n_layers, device, path, act, out_dim=3, load_param=True):
    layers = []

    in_features = N

    for _ in range(n_layers):
        out_features = n_units
        layers.append(nn.Linear(in_features, out_features))
        layers.append(getattr(nn, act)())
        in_features = out_features
    layers.append(nn.Linear(in_features, out_dim))

    model = nn.Sequential(*layers).to(device)
    if load_param:
        model.load_state_dict(torch.load(path, map_location=device))

    return model

class DenoisingUNet3(nn.Module):
    def __init__(self, N, n_units, device, act, alpha) -> None:
        super(DenoisingUNet3, self).__init__()
        self.fc1 = torch.nn.Linear(N, n_units[0], device=device)
        self.fc2 = torch.nn.Linear(n_units[0], n_units[1], device=device)
        self.fc3 = torch.nn.Linear(n_units[1], n_units[2], device=device)
        self.fc4 = torch.nn.Linear(n_units[2], N, device=device)
        self.act = getattr(nn, act)()
        self.dropout = torch.nn.Dropout(p = alpha)
        self.alpha = alpha

    def forward(self, x):
        out = self.act(self.fc1(x))
        out1 = out.clone()
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc2(out))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc3(out))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.fc4(out + out1)

        return out
    
class DenoisingUNet5(nn.Module):
    def __init__(self, N, n_units, device, act, alpha) -> None:
        super(DenoisingUNet5, self).__init__()
        self.fc1 = torch.nn.Linear(N, n_units[0], device=device)
        self.fc2 = torch.nn.Linear(n_units[0], n_units[1], device=device)
        self.fc3 = torch.nn.Linear(n_units[1], n_units[2], device=device)
        self.fc4 = torch.nn.Linear(n_units[2], n_units[3], device=device)
        self.fc5 = torch.nn.Linear(n_units[3], n_units[4], device=device)
        self.fc6 = torch.nn.Linear(n_units[4], N, device=device)
        self.act = getattr(nn, act)()
        self.dropout = torch.nn.Dropout(p = alpha)
        self.alpha = alpha

    def forward(self, x):
        out = self.act(self.fc1(x))
        out1 = out.clone()
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc2(out))
        out2 = out.clone()
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc3(out))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc4(out))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc5(out + out2))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.fc6(out + out1)

        return out

class DenoisingUNet7(nn.Module):
    def __init__(self, N, n_units, device, act, alpha) -> None:
        super(DenoisingUNet7, self).__init__()
        self.fc1 = torch.nn.Linear(N, n_units[0], device=device)
        self.fc2 = torch.nn.Linear(n_units[0], n_units[1], device=device)
        self.fc3 = torch.nn.Linear(n_units[1], n_units[2], device=device)
        self.fc4 = torch.nn.Linear(n_units[2], n_units[3], device=device)
        self.fc5 = torch.nn.Linear(n_units[3], n_units[4], device=device)
        self.fc6 = torch.nn.Linear(n_units[4], n_units[5], device=device)
        self.fc7 = torch.nn.Linear(n_units[5], n_units[6], device=device)
        self.fc8 = torch.nn.Linear(n_units[6], N, device=device)
        self.act = getattr(nn, act)()
        self.dropout = torch.nn.Dropout(p = alpha)
        self.alpha = alpha

    def forward(self, x):
        out = self.act(self.fc1(x))
        out1 = out.clone()
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc2(out))
        out2 = out.clone()
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc3(out))
        out3 = out.clone()
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc4(out))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc5(out))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc6(out + out3))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.act(self.fc7(out + out2))
        if not self.alpha == 0:
            out = self.dropout(out)
        out = self.fc8(out + out1)

        return out

def cal_units_dae(n_layers, nodes, multiple):
    n_units_t = []
    n_layers += 2
    for i in range(n_layers//2):
        if not i == 0:
            n_units_t.append(int(nodes * multiple))
            nodes = int(nodes * multiple)
        else:
            n_units_t.append(nodes)
    n_units = n_units_t
    for i in range(n_layers//2 - 2, -1, -1):
        n_units.append(n_units_t[i])

    return n_units

class UNet_DNN(nn.Module):
    def __init__(self, unet, dnn) -> None:
        super(UNet_DNN, self).__init__()
        self.unet = unet
        self.dnn = dnn

    def forward(self, x):
        out = self.unet(x)
        out = self.dnn(out[:, :-1])

        return out

def rmse_loss(x, y):
    results = torch.sqrt(torch.mean((x - y)**2, 1))

    return results
    # return results.mean(), results.var()

def find_diff_min(data):
    diff = torch.abs(data[:,1:] - data[:,0:-1])
    min = torch.min(diff, dim=1)

    return torch.mean(min.values).item()

def annihilation(d):
    '''
    This function's objective is to return a annihilation operator matrix like qutip's function.
    '''
    m = np.zeros((d,d))

    for i in range(1, d):
        m[i-1][i] = np.sqrt(i)

    return m

def eig_h(param, cutoff):
    '''
    This function is calculate eigenvalues of Hamiltonian with given parameters.
    '''
    a = np.kron(annihilation(cutoff), np.eye(2)) # a operator
    sm = np.kron(np.eye(cutoff), annihilation(2))    # sigma_minus operator

    x = np.zeros((3, cutoff*2, cutoff*2))

    x[0] = a.conj().T @ a
    x[1] = sm.conj().T @ sm
    x[2] = a.conj().T @ sm + a @ sm.conj().T

    # Let h_bar = 1
    H = param[0]*x[0] + param[1]*x[1] + param[2]*x[2]

    eigvals = torch.linalg.eigvalsh(H)

    return eigvals

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # y_true, y_pred = np.array(y_true[:,1:]), np.array(y_pred[:,1:])
    idx = np.where(y_true == 0)
    y_true, y_pred = np.delete(y_true, idx[1], axis=1), np.delete(y_pred, idx[1], axis=1)

    return np.mean(np.abs((y_true - y_pred) / y_true), axis=1) * 100

def dae_test(dataloader, model, device, eps, mean):
    n = len(dataloader)
    rmse_mean = 0.0
    rmse_var = 0.0
    mape_mean = 0.0

    cnt = 1

    with torch.no_grad():
        model.eval()
        with tqdm(dataloader, unit='batch', ascii=True) as tepoch:
            for data, _ in tepoch:
                tepoch.set_description('Test')
                inputs = data.to(device)
                if not eps == 0:
                    noise = torch.normal(0, mean/3, data.shape)
                    noise_inputs = data + eps * noise
                    noise_inputs = noise_inputs.to(device)
                else:
                    noise_inputs = inputs
                outputs = model(noise_inputs)

                results = rmse_loss(outputs, inputs)
                rmse_mean += results.mean().item()
                rmse_var += results.var().item()

                mape = mean_absolute_percentage_error(inputs.to('cpu'), outputs.to('cpu'))
                mape_mean += mape.mean()
                tepoch.set_postfix(loss=results.mean().item())

                cnt += 1
    
    model.train()

    return rmse_mean / n, rmse_var / n, mape_mean / n

# bases: |n, e>, |n+1, g>. n=0, 1, 2, ...
def eigs(w_c, w_a, g, n):
    delta = w_c - w_a
    ep = (n+1)*w_c - 0.5*delta + 0.5*np.sqrt(np.power(delta, 2) + (n+1)*np.power(2*g, 2))
    em = (n+1)*w_c - 0.5*delta - 0.5*np.sqrt(np.power(delta, 2) + (n+1)*np.power(2*g, 2))

    return ep, em

# n_limit: n's upper limit - 1
def eigs_multiple(w_c, w_a, g, n_limit):
    delta = w_c - w_a
    n = np.arange(0, n_limit)
    ep = (n+1)*w_c - 0.5*delta + 0.5*np.sqrt(np.power(delta, 2) + (n+1)*np.power(2*g, 2))
    em = (n+1)*w_c - 0.5*delta - 0.5*np.sqrt(np.power(delta, 2) + (n+1)*np.power(2*g, 2))

    results = np.concatenate((ep, em), axis=0)
    results.sort()

    return results

def dnn_test(dataloader, model, device, eps, mean, N):
    n = len(dataloader)

    rmse_error = {'running_loss': 0.0, 'w_c_loss': 0.0, 'w_a_loss': 0.0, 'g_loss': 0.0, 'e_loss': 0.0}
    rmse_error_var = {'running_loss': 0.0, 'w_c_loss': 0.0, 'w_a_loss': 0.0, 'g_loss': 0.0, 'e_loss': 0.0}
    mape = {'running_loss': 0.0, 'w_c_loss': 0.0, 'w_a_loss': 0.0, 'g_loss': 0.0, 'e_loss': 0.0}

    with torch.no_grad():
        model.eval()
        with tqdm(dataloader, unit='batch', ascii=True, position=0) as tepoch:
            for data, target in tepoch:
                tepoch.set_description('Test')
                inputs, targets = data.to(device), target.to(device)

                if not eps == 0:
                    noise = torch.normal(0, mean/3, data.shape)
                    noise_inputs = data + eps * noise
                    noise_inputs = noise_inputs.to(device)
                    outputs = model(noise_inputs)
                else:
                    outputs = model(inputs)

                w_c = rmse_loss(outputs[:,0].reshape((-1, 1)), targets[:,0].reshape((-1, 1)))
                rmse_error['w_c_loss'] += w_c.mean().item()
                rmse_error_var['w_c_loss'] += w_c.var().item()
                w_c_p = mean_absolute_percentage_error(targets[:,0].cpu().reshape((-1, 1)), outputs[:,0].cpu().reshape((-1, 1)))
                mape['w_c_loss'] += w_c_p.mean()

                w_a = rmse_loss(outputs[:,1].reshape((-1, 1)), targets[:,1].reshape((-1, 1)))
                rmse_error['w_a_loss'] += w_a.mean().item()
                rmse_error_var['w_a_loss'] += w_a.var().item()
                w_a_p = mean_absolute_percentage_error(targets[:,1].cpu().reshape((-1, 1)), outputs[:,1].cpu().reshape((-1, 1)))
                mape['w_a_loss'] += w_a_p.mean()

                g = rmse_loss(outputs[:,2].reshape((-1, 1)), targets[:,2].reshape((-1, 1)))
                rmse_error['g_loss'] += g.mean().item()
                rmse_error_var['g_loss'] += g.var().item()
                g_p = mean_absolute_percentage_error(targets[:,2].cpu().reshape((-1, 1)), outputs[:,2].cpu().reshape((-1, 1)))
                mape['g_loss'] += g_p.mean()

                total = rmse_loss(outputs, targets)
                rmse_error['running_loss'] += total.mean().item()
                rmse_error_var['running_loss'] += total.var().item()
                total_p = mean_absolute_percentage_error(targets.cpu(), outputs.cpu())
                mape['running_loss'] += total_p.mean()

                e_temp = []
                e_temp_p = []
                for i in tqdm(range(len(outputs)), desc='inner', ascii=True, position=1, leave=False):
                    origin = np.zeros(N)
                    pred = np.zeros(N)
                    origin = eigs_multiple(targets[i, 0].cpu().numpy(), targets[i, 1].cpu().numpy(), targets[i, 2].cpu().numpy(), N//2)
                    pred = eigs_multiple(outputs[i, 0].cpu().numpy(), outputs[i, 1].cpu().numpy(), outputs[i, 2].cpu().numpy(), N//2)
                    origin = torch.tensor(origin).view((1, -1))
                    pred = torch.tensor(pred).view((1, -1))
                    e_temp.append(rmse_loss(pred, origin).mean().item())
                    e_temp_p.append(mean_absolute_percentage_error(origin, pred).mean())
                rmse_error['e_loss'] += np.mean(e_temp)
                rmse_error_var['e_loss'] += np.var(e_temp)
                mape['e_loss'] += np.mean(e_temp_p)

                tepoch.set_postfix(loss=total.mean().item())
    
    model.train()

    for name, _ in rmse_error.items():
        rmse_error[name] /= n
        rmse_error_var[name] /= n
        mape[name] /= n

    return rmse_error, rmse_error_var, mape

def max_min(max, min, max_ref, min_ref):
    if max > max_ref:
        max_ref = max
    if min < min_ref:
        min_ref = min

    return max_ref, min_ref
