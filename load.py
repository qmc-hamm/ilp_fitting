import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load(method, basis, shift=0):
    if method in ['dft-d3', 'dft-d2', 'qmc']:
        d = pd.read_csv('data/all.csv')
        mask = (d['method'] == method.upper()) & (d['basis'] == basis.lower())
        d = d.loc[mask, :]
        d['d'] = d['distance']
        d['a'] = 2.46
    else:
        raise RuntimeError()
    d['energy'] += shift
    return d

if __name__ == '__main__':
    d = load('qmc', 'BN')
    # d = load('dft-d3', 'BN')
    print(d)
    exit()
    print(get_starting_e_inf(d))
    print(d['energy'].min())
    f = sns.FacetGrid(data=d, col='registry')
    f.map(plt.errorbar, 'd', 'energy', 'energy_err')
    plt.show()
