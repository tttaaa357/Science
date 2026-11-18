from snewpy.models import ccsn
import astropy.units as u
import numpy as np
from snewpy.flavor import FlavorMatrix, ThreeFlavor
'''Класс для чтения матриц и преобразования потоков'''
class MatrixLoader:
    def __init__(self, filepath='flavor_matrix.npz'):
        self.filepath = filepath
        self.data = None
        self.time = None
        self.energy = None
        self._load_data()

    def _load_data(self):
        self.data = np.load(self.filepath)
        self.time = self.data['T'] << u.s
        self.energy = self.data['E'] << u.MeV

    def get_matrix(self, angle, hierarchy): # получение матрицы
        array = self.data[f'M_{angle}_{hierarchy}']
        return FlavorMatrix(array=array, flavor=ThreeFlavor)

    def get_time(self): # получение временных бинов
        return self.time

    def get_energy(self): # получение энергетических бинов
        return self.energy
        
    def apply_to_flux(self, matrix, model=ccsn.Bollig_2016(progenitor_mass=27<<u.Msun), distance=1): #преобразование потока
        M = matrix
        flux = model.get_flux(self.time, self.energy, distance=distance*u.kpc)
        transformed_flux = (M @ flux)
        return transformed_flux
    
