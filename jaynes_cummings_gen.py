import torch
from torch.distributions import uniform
import numpy as np
from tqdm import tqdm
import jc_torch_library as lib

'''
For generating input data file only.
'''

def annihilation(d):
    '''
    This function's objective is to return a annihilation operator matrix like qutip's function.
    '''
    m = np.zeros((d,d))

    for i in range(1, d):
        m[i-1][i] = np.sqrt(i)

    return m

def eig_h(param, cutoff):
    # This function is calculate eigenvalues of Hamiltonian with given parameters.
    a = np.kron(annihilation(cutoff), np.eye(2)) # a operator
    sm = np.kron(np.eye(cutoff), annihilation(2))    # sigma_minus operator

    x = np.zeros((3, cutoff*2, cutoff*2))

    x[0] = a.conj().T @ a
    x[1] = sm.conj().T @ sm
    x[2] = a.conj().T @ sm + a @ sm.conj().T

    # Let h_bar = 1
    H = param[0]*x[0] + param[1]*x[1] + param[2]*x[2]

    eigvals = torch.linalg.eigvalsh(torch.tensor(H))

    return eigvals

def gen_value(min, max):
    while True:
        # val = tf.random.uniform([1], min, max, dtype=tf.float32).numpy()
        dist = uniform.Uniform(torch.tensor([min]), torch.tensor([max]))
        val = dist.sample().numpy()
        if val[0] > 0:
            return val[0]

def gen_input(n, size, index):
    # sequences of raw data: w_c, w_a, g, eig_vals of energy[n]
    # n: cutoff, num: numbers of output energy eigenvalues
    # This function is generate txt file including input datas.
    data = np.zeros((size,n+3))
    param = np.zeros((3,))

    print(f'progress:')

    for i in tqdm(range(size), ascii=True):
        flag = False    # False: e1 < e2

        param[0] = gen_value(0.1, 1)
        param[1] = gen_value(0.1, 1)
        w_p = (param[0] + param[1]) / 2
        if index==1:
            param[2] = gen_value(0.1, 1)
        elif index==2:
            param[2] = gen_value(0.001 * w_p, 0.1 * w_p)
        elif index==3:
            param[2] = gen_value(0.1 * w_p, 0.2 * w_p)
        data[i, :3] = [param[0], param[1], param[2]]

        _, e1 = eigs(param[0], param[1], param[2], 0)
        _, e2 = eigs(param[0], param[1], param[2], 1)
        if e1 > e2:
            flag = True

        th = ((np.power((np.power(param[0], 2) - np.power(param[2], 2)), 2) - np.power(param[0], 2) * np.power(param[0] - param[1], 2)) / (4 * np.power(param[0], 2) * np.power(param[2], 2)))
        if th < 0:
            th = 0

        if flag:
            temp = np.zeros((int(th+1) + n)*2)
            temp = lib.eigs_multiple(param[0], param[1], param[2], int(th+1) + n)
        else:
            temp = np.zeros(n*2)
            temp = lib.eigs_multiple(param[0], param[1], param[2], n)

        data[i, 3:] = temp[:n]

    np.save(f'data/testset_size_{size}_cond{index}.npy', data)

# bases: |n, e>, |n+1, g>. n=0, 1, 2, ...
def eigs(w_c, w_a, g, n):
    delta = w_c - w_a
    ep = (n+1)*w_c - 0.5*delta + 0.5*np.sqrt(np.power(delta, 2) + (n+1)*np.power(2*g, 2))
    em = (n+1)*w_c - 0.5*delta - 0.5*np.sqrt(np.power(delta, 2) + (n+1)*np.power(2*g, 2))

    return ep, em

if __name__ == '__main__':
    n = 97
    size = 400000
    condition = 2

    print(f'size: {size}, condition: {condition}')
    gen_input(n, size, condition)
