import torch
from torch.utils.data import DataLoader
import gc
import jc_torch_library as lib
import optuna

datatype = 'sq_diff'
mean = {'diff_2': 0.26, 'sq_diff': 0.28}    # cond2

def load_dataset(path, n_input, datatype, batch_size):
    dataset = lib.JCDataset(path, n_input, datatype)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return testloader

def print_values(dataloader, model, device, f, eps):
    iter_data = iter(dataloader)
    data = next(iter_data)
    inputs, targets = data[0], data[1]
    inputs = inputs[:5]
    targets = targets[:5]

    with torch.no_grad():
        model.eval()
        if not eps == 0:
            noise = torch.normal(0, mean[datatype]/3, inputs.shape)
            add_noise = eps * noise
            noise_inputs = inputs + add_noise
            outputs = model(noise_inputs.to(device))
        else:
            outputs = model(inputs.to(device))

        outputs = outputs.cpu()
        f.write(f'{"-"*60}\n')
        for i in range(5):
            eigvals_p = lib.eig_h(outputs[i], 5)
            eigvals_g = lib.eig_h(targets[i], 5)
            f.write(f' Ground Truth\tPredicted\n')
            f.write(f'w_c: {targets[i, 0]:.5f}\t{outputs[i, 0]:.5f}\n')
            f.write(f'w_a: {targets[i, 1]:.5f}\t{outputs[i, 1]:.5f}\n')
            f.write(f'g  : {targets[i, 2]:.5f}\t{outputs[i, 2]:.5f}\n')
            for j in range(10):
                f.write(f'[{j}]: {eigvals_g[j]:.5f}\t{eigvals_p[j]:.5f}\n')
            f.write(f'{"-"*60}\n')

def main():
    cond = 'cond2'
    batch_size = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available.')
    input_path = f'data/testset_size_400000_{cond}.npy'

    for N in iter([12, 24, 48, 96]):
        study_name = f'JC_dnn_{cond}_{datatype}_N_{N}'
        storage_name = "sqlite:///data/tune/{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(), load_if_exists=True)
        param = study.best_trial.params
        path = f'data/dnn/{cond}/{datatype}/N_{N}_model.pth'

        model = lib.DNN(N-1, param['n_units'], param['n_layers'], device, path, param['act'], load_param=True)
        for eps in iter([0, 0.2, 0.4, 0.6, 0.8]):
            torch.manual_seed(0)
            testloader = load_dataset(input_path, N, datatype, batch_size)
            print(f'N: {N}, datatype: {datatype}, eps: {eps}, condition: {cond}')
            rmse_error, rmse_error_var, mape = lib.dnn_test(testloader, model, device, eps, mean[datatype], N)
            with open(f'data/dnn/{cond}/{datatype}/N_{N}_test_results_eps_{eps}.txt', 'w') as f:
                f.write(f'RMSE:\n')
                f.write(f'w_c_loss: {rmse_error["w_c_loss"]} (var: {rmse_error_var["w_c_loss"]}),\
w_a_loss: {rmse_error["w_a_loss"]} (var: {rmse_error_var["w_a_loss"]}),\
g_loss: {rmse_error["g_loss"]} (var: {rmse_error_var["g_loss"]}),\
total loss: {rmse_error["running_loss"]} (var: {rmse_error_var["running_loss"]}),\
    e_loss: {rmse_error["e_loss"]} (var: {rmse_error["e_loss"]})\n')
                print('RMSE:')
                print(f'w_c_loss: {rmse_error["w_c_loss"]} (var: {rmse_error_var["w_c_loss"]}),\
w_a_loss: {rmse_error["w_a_loss"]} (var: {rmse_error_var["w_a_loss"]}),\
g_loss: {rmse_error["g_loss"]} (var: {rmse_error_var["g_loss"]}),\
total loss: {rmse_error["running_loss"]} (var: {rmse_error_var["running_loss"]}),\
    e_loss: {rmse_error["e_loss"]} (var: {rmse_error["e_loss"]})')
                f.write(f'MAPE:\n')
                f.write(f'w_c_loss: {mape["w_c_loss"]}%, w_a_loss: {mape["w_a_loss"]}%, g_loss: {mape["g_loss"]}%, total loss: {mape["running_loss"]}%, e_loss: {mape["e_loss"]}%\n')
                print('MAPE:')
                print(f'w_c_loss: {mape["w_c_loss"]}%, w_a_loss: {mape["w_a_loss"]}%, g_loss: {mape["g_loss"]}%, total loss: {mape["running_loss"]}%, e_loss: {mape["e_loss"]}%')
                print_values(testloader, model, device, f, eps)

            gc.collect()

if __name__ == '__main__':
    main()
