import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.linalg as la
import pandas as pd
import seaborn as sns
import os
import re

import load
import eval_kc

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def get_dirname(method, basis, shift, kT, exclude):
    kT_str = 'inf' if kT == 'inf' else f'{kT:.4f}'
    dirname = f'fit/{basis}_{method}_{shift:.3f}_{kT_str}_{exclude}'
    return dirname


def plot_error(ax, x, y, yerr=None, color='C0', label=None):
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt='none', ms=0, mew=1, elinewidth=1, capsize=1.5, zorder=1, ecolor=color)
        ax.plot(x, y, 'o', ms=2, mew=1, mec='white', mfc='white', zorder=2)
    ax.plot(x, y, 'o', ms=2, mew=1, mec=color, mfc=mpl.colors.to_rgba(color, 0.7), zorder=3, label=label)

def get_stacking(registry):
    stacking_map = {
        0.0: 'AB',
        0.16667: 'SP',
        0.33333: 'BA',
        0.5: 'Mid',
        0.66667: 'AA'
    }
    return stacking_map[registry]

def gen_kc(dirname, registries, distances, basis):
    params_all = np.loadtxt(f'{dirname}/params.txt')
    l = []
    for iteration in [0, params_all.shape[0] - 1]:
        params = params_all[iteration, :]
        for registry in registries:
            for distance in distances:
                print(f'evaluating kc for dir: {dirname} registry: {registry} distance: {distance}')
                energy = eval_kc.eval_energy_from_params(registry, distance, 2.46, params, 'BNC.ILP.plot', basis)
                l.append({
                    'iteration': iteration,
                    'registry': registry,
                    'distance': distance,
                    'energy': energy
                    })
    d = pd.DataFrame(l)
    d.to_csv(f'{dirname}/kc.csv', index=False)

def plot_energy(d, dirname):
    p = np.loadtxt(f'{dirname}/params.txt')
    registries = d['registry'].unique()

    if not os.path.isfile(f'{dirname}/kc.csv'):
        distances = np.arange(2.6, 9.0+0.05, 0.05)
        gen_kc(dirname, registries, distances, basis)
    k = pd.read_csv(f'{dirname}/kc.csv')

    emax = d.loc[d['d'] > 4, 'energy'].max() + 0.001
    emin = d['energy'].min() - 0.001

    ncols = len(registries)
    fig, axs = plt.subplots(nrows=1, ncols=ncols, sharey=True, figsize=(6.5, 2.5))
    for i, registry in enumerate(registries):
        ax = axs[i]
        dd = d.loc[d['registry'] == registry, :]
        plot_error(ax, dd['d'], dd['energy'], yerr=dd['energy_err'], color='C0', label=method)
        for iteration in [p.shape[0] - 1, 0]:
            mask = (k['registry'] == registry) & (k['iteration'] == iteration)
            kk = k.loc[mask, :]
            kc_label = 'ILP-Ouyang' if iteration == 0 else f'ILP-{method}'
            color = 'C1' if iteration == 0 else 'C0'
            ax.plot(kk['distance'], kk['energy'], label=kc_label, zorder=0, color=color)

        stacking = get_stacking(registry)
        ax.set_title(stacking)
        ax.set_xlabel('$d~(\\textrm{{\\AA}})$')

        ax.set_ylim(emin, emax)
    axs[0].set_ylabel('$E$ (eV/atom)')
    axs[-1].legend(bbox_to_anchor=(1, 0.8), frameon=False)

    fig.tight_layout()
    dirname_tail = os.path.split(dirname)[-1]
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{dirname_tail}_energy.pdf', bbox_inches='tight')
    plt.close()

def plot_rmse(d, dirname):
    k = np.loadtxt(f'{dirname}/energy.txt')
    l = []
    for iteration in range(k.shape[0]):
        kc_energy = k[iteration, :]
        rms = la.norm(d['energy'] - kc_energy) / np.sqrt(d.shape[0])
        l.append(rms)
    last_rms = l[-1]
    plt.plot(l)
    plt.xlabel('iteration')
    plt.ylabel('rms (eV/atom)')
    plt.ylim(0, 0.005)
    dirname_tail = os.path.split(dirname)[-1]
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{dirname_tail}_rmse.pdf', bbox_inches='tight')
    plt.close()
    return rms

def get_rms_r2(ydata, ypred, kT):
    if kT == 'inf':
        weights = np.ones(ydata.shape)
    else:
        weights = np.exp(-(ydata-ydata.min())/kT)

    residuals = ydata - ypred
    ss_res = np.sum(residuals**2)
    # rms = (ss_res / len(ydata))**0.5

    weighted_sum = np.sum(weights * np.square(residuals))
    sum_of_weights = np.sum(weights)
    rms = np.sqrt(weighted_sum / sum_of_weights)

    ss_res = np.sum(weights * np.square(residuals))
    ss_tot = np.sum(weights * np.square(ydata - np.average(ydata, weights=weights)))
    r2 = 1 - (ss_res / ss_tot)
    # ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return r2, rms

l = []
for dirname in os.listdir('fit'):
    match = re.search(f'(.+)_(.+)_(.+)_(.+)_(.+)', dirname)
    if not match:
        continue
    basis = match.group(1)
    method = match.group(2)
    shift = float(match.group(3))
    kT = float(match.group(4))
    exclude = match.group(5)
    d = load.load(method, basis, shift)
    dirname = get_dirname(method, basis, shift, kT, exclude)
    print(dirname)
    plot_energy(d, dirname)
    rms = plot_rmse(d, dirname)
    l.append({
        'basis': basis,
        'method': method,
        'shift': shift,
        'kT': kT,
        'exclude': exclude,
        'rms': rms
        })
df = pd.DataFrame(l)
print(df)
