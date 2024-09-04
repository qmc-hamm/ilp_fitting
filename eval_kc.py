'''
Evaluate KC potential on a given structure
'''
from ase.calculators.lammpsrun import LAMMPS
import os
import shutil
import numpy as np

import geom

def get_lammps_options_tersoff(ilp_filename, potential_dir):
    parameters = {
        'pair_style': 'hybrid/overlay rebo tersoff ilp/graphene/hbn 16.0 coul/shield 16.0',
        'pair_coeff': [
            f'* * rebo {potential_dir}/CH.rebo NULL C NULL',
            f'* * tersoff {potential_dir}/BNC.tersoff B NULL N',
            f'* * ilp/graphene/hbn {potential_dir}/{ilp_filename} B C N',
            '1 1 coul/shield 0.70',
            '1 3 coul/shield 0.695',
            '3 3 coul/shield 0.69'
            ],
        'atom_style': 'full',
        'specorder': ['B', 'C', 'N'],
        }
    files =  [f'{potential_dir}/{fn}' for fn in ['CH.rebo', 'BNC.tersoff', ilp_filename]]
    return parameters, files

def get_lammps_options(ilp_filename, potential_dir):
    parameters = {
        'pair_style': 'hybrid/overlay rebo extep ilp/graphene/hbn 16.0 1',
        'pair_coeff': [
            f'* * rebo {potential_dir}/CH.rebo NULL C NULL',
            f'* * extep {potential_dir}/BN.extep B NULL N',
            f'* * ilp/graphene/hbn {potential_dir}/{ilp_filename} B C N',
            ],
        'atom_style': 'full',
        'specorder': ['B', 'C', 'N'],
        }
    files =  [f'{potential_dir}/{fn}' for fn in ['CH.rebo', 'BN.extep', ilp_filename]]
    return parameters, files


def eval_energy(registry, distance, a, ilp_filename, tmp_dir='lmp_tmp', basis='BN', potential_dir='potentials'):
    '''
    Finds the energy for a given geometry (defined by `distance` and `registry`) and KC paramters
    '''
    options, files = get_lammps_options(ilp_filename, potential_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    atoms = geom.create_geom(registry, distance, a, basis=basis)
    atoms.calc = LAMMPS(files=files, keep_tmp_files=True, tmp_dir=tmp_dir, lammps_options='', **options)
    e = atoms.get_potential_energy()/len(atoms)
    return e

def write_interlayer_potential(ilp_filename, params_dict, potential_dir, column_space=16):
    headers = ['beta(A)', 'alpha', 'delta(A)', 'epsilon(meV)', 'C(meV)', 'd', 'sR', 'reff(A)', 'C6(meV*A^6)']
    headers_str = ''.join(f'{header:>{column_space}}' for header in headers)
    params_str_dict = {}
    for pair in ['CC', 'BB', 'NN', 'HH', 'CB', 'CN', 'BN', 'CH', 'BH', 'NH']:
        params_str_dict[pair] = ''.join(f"{f'{param:.8f}':>{column_space}}" for param in params_dict[pair])
    potential_str = f"""# Interlayer Potential (ILP) for graphene/hBN trained from QMC data
#
#
#
# ---------------------------------- Repulsion Potential ------------------------------+++++++++++++++++++++++ Vdw Potential +++++++++++++++++++++++++************
#   {headers_str}    S    rcut
C C {params_str_dict['CC']}   1.0   2.0
B B {params_str_dict['BB']}   1.0   2.0
N N {params_str_dict['NN']}   1.0   2.0
H H {params_str_dict['HH']}   1.0   1.2
C B {params_str_dict['CB']}   1.0   2.0
C N {params_str_dict['CN']}   1.0   2.0
B N {params_str_dict['BN']}   1.0   2.0
C H {params_str_dict['CH']}   1.0   1.5
B H {params_str_dict['BH']}   1.0   1.5
N H {params_str_dict['NH']}   1.0   1.5
# Equivalent pairs
B C {params_str_dict['CB']}   1.0   2.0
N C {params_str_dict['CN']}   1.0   2.0
N B {params_str_dict['BN']}   1.0   2.0
H C {params_str_dict['CH']}   1.0   2.2
H B {params_str_dict['BH']}   1.0   2.2
H N {params_str_dict['NH']}   1.0   2.2
"""
    with open(f'{potential_dir}/{ilp_filename}', 'w') as f:
        f.write(potential_str)

def write_interlayer_potential_bnc(ilp_filename, params, potential_dir, basis):
    '''
    starting parameters from W. Ouyang, D. Mandelli, M. Urbakh and O. Hod, Nano Lett. 18, 6009-6016 (2018)
    '''
    params_dict = {
        'BB': [3.143737, 9.825139, 1.936405, 2.7848400, 14.495957, 15.199263, 0.7834022, 3.682950, 49.498013E3],
        'NN': [3.443196, 7.084490, 1.747349, 2.9139991, 46.508553, 15.020370, 0.8008370, 3.551843, 14.810151E3],
        'HH': [3.974540, 6.53799, 1.080633, 0.6700556, 0.8333833, 15.022371, 0.7490632, 2.767223, 1.6159581E3],
        'BN': [3.295257, 7.224311, 2.872667, 1.3715032, 0.4347152, 14.594578, 0.8044028, 3.765728, 24.669996E3],
        'CH': [2.642950, 12.91410, 1.020257, 0.9750012, 25.340996, 15.222927, 0.8115998, 3.887324, 5.6874617E3],
        'BH': [2.718657, 9.214551, 3.273063, 14.015714, 14.760509, 15.084752, 0.7768383, 3.640866, 7.9642467E3],
        'NH': [2.753464, 8.226713, 3.106390, 0.8073613, 0.3944229, 15.033188, 0.7451414, 2.733583, 3.8461530E3]
    }
    if basis == 'BN':
        params_dict['CC'] = [3.205843, 7.511126, 1.235334, 1.528338E-5, 37.530428, 15.499947, 0.7954443, 3.681440, 25.714535E3]
        params_dict['CB'] = params[0:9]
        params_dict['CN'] = params[9:18]
    elif basis == 'C2':
        params_dict['CC'] = params
        params_dict['CB'] = [3.303662, 10.54415, 2.926741, 16.719972, 0.3571734, 15.305254, 0.7001581, 3.097327, 30.162869E3]
        params_dict['CN'] = [3.253564, 8.825921, 1.059550, 18.344740, 21.913573, 15.000000, 0.7234983, 3.013117, 19.063095E3]
    write_interlayer_potential(ilp_filename, params_dict, potential_dir)

def eval_energy_from_params(registry, distance, a, params_e0, ilp_filename, basis, potential_dir='potentials'):
    params = params_e0[:-1]
    e0 = params_e0[-1]
    write_interlayer_potential_bnc(ilp_filename, params, potential_dir, basis)
    energy = eval_energy(registry, distance, a, ilp_filename, potential_dir=potential_dir, basis=basis) + e0
    return energy

if __name__ == '__main__':
    e = eval_energy(registry=0.0, distance=3.2, a=2.46, ilp_filename='BNC.ILP.tmp')
    print(e)
