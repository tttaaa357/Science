'''Создание и сохранение в словаре матриц трансформаций'''
import astropy.units as u
import numpy as np
from snewpy.flavor_transformation import EarthMatter
from snewpy.neutrino import MixingParameters, MassHierarchy
from astropy.coordinates import AltAz
from tqdm.auto import tqdm, trange

matrices = {}
energy = np.linspace(1,100,800)<<u.MeV
time = np.linspace(0,2,800)<<u.s
for angle in trange(-90, 1, 1):
    for mh in MassHierarchy:
        altaz_nh = AltAz(alt=angle*u.deg, az=0*u.deg)
        flavor_transform = EarthMatter(AltAz=altaz_nh, mixing_params= MixingParameters(mh))
        M = flavor_transform.P_ff(time, energy)
        matrices[f'M_{angle:d}_{mh.name}'] = M.array
np.savez('flavor_matrix.npz', # запись в файл всех матриц
         **matrices,
         T=time.to_value('s'), 
         E=energy.to_value('MeV'))

