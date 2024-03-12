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
    inputs = data[0]
    inputs = inputs[:5]

    with torch.no_grad():
        model.eval()
        if not eps == 0:
            noise = torch.normal(0, mean[datatype]/3, inputs.shape)
            add_noise = eps * noise
            noise_inputs = inputs + add_noise
        else:
            noise_inputs = inputs
        outputs = model(noise_inputs.to(device))
        outputs = outputs.cpu()

        f.write(f'{"-"*60}\n')
        for i in range(5):
            f.write(f' Ground Truth\tPredicted\tNoise\n')
            for j in range(len(outputs[0])):
                if not eps == 0:
                    f.write(f'[{j}]: {inputs[i, j]:.5f}\t{outputs[i, j]:.5f}\t{add_noise[i, j]:.5f}\n')
                else:
                    f.write(f'[{j}]: {inputs[i, j]:.5f}\t{outputs[i, j]:.5f}\t{0.0}\n')
            f.write(f'{"-"*60}\n')

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available.')
    cond = 'cond2'
    input_path = f'data/testset_size_400000_{cond}.npy'
    batch_size = 1024

    for N in iter([12, 24, 48, 96]):
        path = f'data/unet/{cond}/{datatype}/N_{N}_model.pth'
        study_name = f'JC_unet_{cond}_{datatype}_N_{N}'
        storage_name = "sqlite:///data/tune/{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(), load_if_exists=True)
        params = study.best_trial.params

        n_units = lib.cal_units_dae(params['n_layers'], params['nodes'], params['multiple'])
        if params['n_layers'] == 3:
            model = lib.DenoisingUNet3(N, n_units, device, params['act'], params['alpha'])
        elif params['n_layers'] == 5:
            model = lib.DenoisingUNet5(N, n_units, device, params['act'], params['alpha'])
        elif params['n_layers'] == 7:
            model = lib.DenoisingUNet7(N, n_units, device, params['act'], params['alpha'])
        model.load_state_dict(torch.load(path, map_location=device))
        for eps in iter([0, 0.2, 0.4, 0.6, 0.8]):
            torch.manual_seed(0)
            if 'diff' in datatype:
                testloader = load_dataset(input_path, N+1, datatype, batch_size)
            else:
                testloader = load_dataset(input_path, N, datatype, batch_size)
            print(f'N: {N}, datatype: {datatype}, eps: {eps}, condition: {cond}')
            rmse_mean, rmse_var, mape_mean = lib.dae_test(testloader, model, device, eps, mean[datatype])

            with open(f'data/unet/{cond}/{datatype}/N_{N}_test_results_eps_{eps}.txt', 'w') as f:
                f.write(f'RMSE: {rmse_mean} (var: {rmse_var}), MAPE: {mape_mean}%\n')
                print(f'RMSE: {rmse_mean} (var: {rmse_var}), MAPE: {mape_mean}%')
                print_values(testloader, model, device, f, eps)

            gc.collect()

if __name__ == '__main__':
    main()
