import argparse
import pandas as pd
import numpy as np
import scipy.optimize
import os

import eval_kc
import load
import morse

PARAMS_REF = {
    'CC': [3.205843, 7.511126, 1.235334, 1.528338E-5, 37.530428, 15.499947, 0.7954443, 3.681440, 25.714535E3],
    'CB': [3.303662, 10.54415, 2.926741, 16.719972, 0.3571734, 15.305254, 0.7001581, 3.097327, 30.162869E3],
    'CN': [3.253564, 8.825921, 1.059550, 18.344740, 21.913573, 15.000000, 0.7234983, 3.013117, 19.063095E3]
}

def save(save_fn, np_obj):
    print(f'saving {save_fn}:', np_obj)
    to_save = np.array([np_obj])
    with open(save_fn, 'ab') as f:
        np.savetxt(f, to_save)

def eval_kc_train(df, basis, *params_e0, potential_dir='potentials', ilp_filename='BNC.ILP.train', save_params=True):
    energy = []
    for _, row in df.iterrows():
        e = eval_kc.eval_energy_from_params(row['registry'], row['d'], row['a'], params_e0, ilp_filename, basis, potential_dir=potential_dir)
        energy.append(e)
    en = np.array(energy)
    if save_params:
        save('params.txt', params_e0)
        save('energy.txt', en)
    return en

def get_starting_params(basis, exclude_idxs=[], resume_from_dir=None):
    if resume_from_dir is not None:
        current_params = np.loadtxt(f'{resume_from_dir}/params.txt')
        starting_params_full = current_params[-1, :]
    elif os.path.isfile('params.txt'):
        current_params = np.loadtxt('params.txt')
        starting_params_full = current_params[-1, :]
    else:
        if basis == 'BN':
            starting_params_full = np.array(PARAMS_REF['CB'] + PARAMS_REF['CN'] + [0])
        elif basis == 'C2':
            starting_params_full = np.array(PARAMS_REF['CC'] + [0])
    mask = np.ones(len(starting_params_full), dtype=bool)
    print('mask', mask)
    print('exclude_idxs', exclude_idxs)
    mask[exclude_idxs] = False
    starting_params_train = starting_params_full[mask]
    return starting_params_full, starting_params_train, mask

def get_bounds(basis):
    base_bounds_min = [0, 0, 0, 0, 0, 14, 0, 2, 0]
    base_bounds_max = [np.inf, np.inf, np.inf, np.inf, np.inf, 16, 1, 4, np.inf]
    if basis == 'BN':
        bounds_min = base_bounds_min*2 + [-np.inf]
        bounds_max = base_bounds_max*2 + [np.inf]
    elif basis == 'C2':
        bounds_min = base_bounds_min + [-np.inf]
        bounds_max = base_bounds_max + [np.inf]
    return np.array(bounds_min), np.array(bounds_max)

def get_alpha(basis):
    base_alpha = [0, 0, 0, 0, 0, 1, 1, 1, 0]
    if basis == 'BN':
        alpha = base_alpha*2 + [0]
    elif basis == 'C2':
        alpha = base_alpha + [0]
    return np.array(alpha)*0.5

def get_exclude_idxs(basis, exclude):
    if basis == 'BN':
        pads = [0, 9]
    elif basis == 'C2':
        pads = [0]

    exclude_idxs = []
    for pad in pads:
        for idx in list(exclude):
            exclude_idxs.append(int(idx) + pad)
    return exclude_idxs

def fit(method, basis, shift, dirname='.', kT='inf', exclude='567', weight_type='unimodal'):
    '''
    method: 'dft-d3'
    basis: 'BN', 'C2'
    shift: 147.105
    dirname:
    kT: 'inf', 0.0045
    exclude: indexs of parameters to exclude from training
    weight_type:
        'unimodal': data points near equilibrium get larger weights
        'bimodal': data points near equilibrium and tail get larger weights
    '''
    d = load.load(method, basis, shift)
    ydata = d['energy']
    print(d)

    workdir = os.getcwd()
    os.makedirs('fit', exist_ok=True)
    os.makedirs(f'fit/{dirname}', exist_ok=True)
    os.chdir(f'fit/{dirname}')
    bounds_min, bounds_max = get_bounds(basis)

    if kT == 'inf':
        kT_str = 'inf'
        weights = np.nan
        sigma = None
    else:
        # set weights such that points near the minimum (lower energy) have more weights, e.g. kT=0.002, 0.004, 0.010
        kT = float(kT)
        kT_str = f'{kT:.4f}'
        if weight_type == 'unimodal':
            weights = np.exp(-(ydata-ydata.min())/kT)
        elif weight_type == 'bimodal':
            mins = d.groupby('registry')['energy'].transform(min)
            ws = []
            for name, g in d.groupby('registry'):
                mask = g['d'] >= 5.0
                g_min = g['energy'].min()
                g_max = g.loc[mask, 'energy'].max()
                w1 = np.exp(-(g.loc[~mask, 'energy'] - g_min)/kT)
                w2 = np.exp(-(g.loc[mask, 'energy'] - g_min)/kT) + np.exp(-(g_max - g.loc[mask, 'energy'])/kT)
                w = np.concatenate([w1, w2])
                ws.append(w)
            weights = np.concatenate(ws)
        sigma = 1/np.sqrt(weights)

        # # check weights
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # print(d)
        # d['weight'] = weights
        # f = sns.FacetGrid(data=d, col='registry')
        # f.map(plt.plot, 'd', 'weight')
        # plt.show()

    exclude_idxs = get_exclude_idxs(basis, exclude)
    print(exclude_idxs)
    if exclude == '567':
        starting_params_full, starting_params_train, mask = get_starting_params(basis, exclude_idxs=exclude_idxs, resume_from_dir=None)
    elif exclude == '7':
        starting_params_full, starting_params_train, mask = get_starting_params(basis, exclude_idxs=exclude_idxs, resume_from_dir=f'{workdir}/fit/{basis}_{method}_{shift:.3f}_{kT_str}_567')

    print(starting_params_full)
    print(starting_params_train)

    def eval_kc_train_wrap(df, *params_e0):
        print('params_e0', params_e0)
        reconstructed_params = np.zeros(len(starting_params_full))
        reconstructed_params[mask] = params_e0
        reconstructed_params[exclude_idxs] = starting_params_full[exclude_idxs]
        print('reconstructed_params', reconstructed_params)
        return eval_kc_train(df, basis, *reconstructed_params, potential_dir=f'{workdir}/potentials', ilp_filename=f'BNC.ILP.train.{dirname}')

    scipy.optimize.curve_fit(eval_kc_train_wrap, d, ydata,
        p0=starting_params_train, method='trf', sigma=sigma, bounds=(bounds_min[mask], bounds_max[mask])
        )
    os.chdir(workdir)

def find_shift(method, basis):
    '''
    find tails of qmc and KC so we can align their tails and help with training.
    Needs to be run once. Not part of the training.
    '''
    d = load.load(method, basis, shift=0)
    starting_params_full, _, _ = get_starting_params(basis)
    en = eval_kc_train(d, basis, *starting_params_full, ilp_filename=f'BNC.ILP.{method}_{basis}.find_shift', save_params=False)

    k = d.copy()
    k['energy'] = en
    einf_qmc = d.groupby('registry').apply(morse.get_e_inf).reset_index()['e_inf'].max()
    einf_kc = k.groupby('registry').apply(morse.get_e_inf).reset_index()['e_inf'].max()
    middle_shift = round(-einf_qmc + einf_kc, 3)
    return middle_shift

if __name__ == '__main__':
    # find_shift('dft-d3', 'BN')
    # fit('dft-d3', 'BN', 157.803, dirname='BN_d3_test', kT='0.0040', exclude='567')
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--shift', type=float)
    parser.add_argument('--dirname', type=str)
    parser.add_argument('--basis', type=str)
    parser.add_argument('--kT', type=float)
    parser.add_argument('--exclude', type=str)
    args = parser.parse_args()
    fit(method=args.method, basis=args.basis, shift=args.shift, dirname=args.dirname, kT=args.kT, exclude=args.exclude)
