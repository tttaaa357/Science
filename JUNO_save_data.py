''' Создание и сохранение в отдельных файлах числа событий для разных углов'''
import astropy.units as u
import numpy as np
from snewpy.models import ccsn
from snewpy.flavor_transformation import EarthMatter, AdiabaticMSW
from snewpy.neutrino import MixingParameters, MassHierarchy, Flavor
from astropy.coordinates import AltAz
from snewpy.rate_calculator import RateCalculator, Detector, DetectionChannel, SmearingMatrix
from tqdm.auto import tqdm
# подавляем все предупреждения 
import warnings
warnings.filterwarnings('ignore')

model = ccsn.Bollig_2016(progenitor_mass=27<<u.Msun)
#define the sampling points
energy = np.linspace(1,100,800)<<u.MeV
time = np.linspace(0,2,800)<<u.s
rc = RateCalculator()

#создание своего детектора
xsec_ibd=rc.load_xsec('ibd',Flavor.NU_E_BAR) #сечение для ibd электронное антинейтрино из SNOwGLoBES
e_true  = np.arange(0,50,0.2)<<u.MeV #энергия нейтрино в пучке
e_smear = np.arange(0,50,0.2)<<u.MeV #измеренная энергия, которую увидит детектор(энергия позитрона)/бины для интегрирования по энергии

#определим видимую энергию для IBD как функцию 
def E_vis_ibd(E_nu:u.Quantity):
    return E_nu - 784*u.keV

#SmearingMatrix.from_Gaussian — это конструктор на основе простого гауссовского распределения. 
#матрица вероятности того, что событие с энергией E_true будет зарегистрировано как E_smear.
m_smear_ibd=SmearingMatrix.from_Gaussian(e_true,e_smear,
                                         mean=E_vis_ibd, #среднее значение, пик Гаусса
                                         sigma=lambda e: 0.05*e #стандартное отклонение, ширина Гаусса, 5% разрешение
                                        )
#создание детектора с каналом IBD
det = Detector('JUNO', mass=20<<u.kt,
               channels={
                   'ibd':DetectionChannel(flavor=Flavor.NU_E_BAR,
                                          xsec=xsec_ibd, 
                                          efficiency=1., 
                                          smearing = m_smear_ibd, 
                                          weight=0.1429) #массовая доля свободных протонов в сцинтилляторе
                                         })
def calc_rates(angles= [-90,90] <<u.deg,
               distance=1*u.kpc,
               mixing_params=MixingParameters(MassHierarchy.NORMAL)
              ):
    results = []
    for alt in tqdm(angles):
        if alt<0:
            altaz = AltAz(alt=alt, az=0*u.deg) 
            flavor_transform = EarthMatter(AltAz=altaz, mixing_params=mixing_params) # Создание преобразования ароматов с учетом Земли
        else:
            flavor_transform = AdiabaticMSW(mixing_params=mixing_params)

        flux = model.get_flux(time, energy, distance=distance, flavor_xform=flavor_transform) #поток нейтрино после осциляций
        ibd_rate = det.run(flux)['ibd'] #скорость событий в детекторе(число событий в секунду) автоматическое интегрирование по энергии в det
        time_bins = np.linspace(0,1,600)<<u.s
        r = ibd_rate.integrate('time', time_bins) #число событий во временном бине
        results += [r]
    return results

angles = np.arange(-90,0,5)
rates = calc_rates(angles=angles<<u.deg)

for angle, rate in zip(angles, rates):
    filename = f'ibd_Juno_rate.{angle:d}.npz'
    print(f"saving rate for {angle=} to {filename=}")
    rate.save(filename)
