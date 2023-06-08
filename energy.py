'''
@author: Rupa Kurinchi-Vendhan
The following code offers a method for generating a kinetic energy spectrum, in a manner similar to generating a power spectrum.
For an official implementation of how to create a plot using turbulent flow statistics as in the paper, refer to this repository:
https://github.com/b-fg/Energy_spectra/blob/master/ek.py.

Modify file directories and other parameters as necessary.
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

Energy_Spectrum = {'Ground Truth':  {'x':[], 'y':[]}, 'LR Input': {'x':[], 'y':[]}, 'WiREDiff': {'x':[], 'y':[]}, 'SR3': {'x':[], 'y':[]}, 'SRCNN': {'x':[], 'y':[]}, 'Bicubic': {'x':[], 'y':[]}}

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16) # controls default text size
plt.rc('axes', titlesize=14) # fontsize of the title
plt.rc('axes', labelsize=14) # fontsize of the x and y labels
plt.rc('xtick', labelsize=14) # fontsize of the x tick labels
plt.rc('ytick', labelsize=14) # fontsize of the y tick labels
plt.rc('legend', fontsize=14) # fontsize of the legend


def energy_spectrum(data_path, min, max):
    data = np.load(data_path)
    npix = data.shape[0]
    fourier_image = np.fft.fftn(data)
    fourier_amplitudes = np.abs(fourier_image)**2
    fourier_amplitudes = np.fft.fftshift(fourier_amplitudes)

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins

def compare_output_helper(data_type, component, timestep, i):
    gt_HR = "data/wind_test//HR/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    gt_LR = "data/wind_test/LR/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    wirediff = "wirediff_output/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    sr3 = "sr3_output/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cub = "bicubic/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cnn = "cnn/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    
    min, max = np.min(np.load(gt_HR)), np.max(np.load(gt_HR))

    if os.path.isfile(cub) and os.path.isfile(cnn) and os.path.isfile(wirediff):        
        HR_kvals2, HR_ek = energy_spectrum(gt_HR, min, max)
        Energy_Spectrum['Ground Truth']['x'].append(HR_kvals2)
        Energy_Spectrum['Ground Truth']['y'].append(HR_ek)

        LR_kvals2, LR_ek = energy_spectrum(gt_LR, min, max)
        Energy_Spectrum['LR Input']['x'].append(LR_kvals2)
        Energy_Spectrum['LR Input']['y'].append(LR_ek)

        sr3_kvals2, sr3_EK = energy_spectrum(sr3, min, max)

        Energy_Spectrum['sr3']['x'].append(sr3_kvals2)
        Energy_Spectrum['sr3']['y'].append(sr3_EK)

        wirediff_kvals2, wirediff_EK = energy_spectrum(wirediff, min, max)

        Energy_Spectrum['PhIREGAN']['x'].append(wirediff_kvals2)
        Energy_Spectrum['PhIREGAN']['y'].append(wirediff_EK)

        cnn_kvals2, cnn_EK = energy_spectrum(cnn, min, max)

        Energy_Spectrum['SR CNN']['x'].append(cnn_kvals2)
        Energy_Spectrum['SR CNN']['y'].append(cnn_EK)

        cub_kvals2, cub_EK = energy_spectrum(cub, min, max)

        Energy_Spectrum['Bicubic']['x'].append(cub_kvals2)
        Energy_Spectrum['Bicubic']['y'].append(cub_EK)

def plot_energy_spectra():
    colors = {'Ground Truth': 'black', 'LR Input': 'pink', 'WiREDiff': 'tab:green', 'SR3': 'tab:orange', 'SR CNN': 'tab:red', 'Bicubic': 'tab:purple'}
    for model in Energy_Spectrum:
        k = np.flip(np.mean(Energy_Spectrum[model]['x'], axis=0))
        E = np.mean(Energy_Spectrum[model]['y'], axis=0) / 10000
        plt.loglog(k, E, color=colors[model], label=model)
    plt.xlabel("k (wavenumber)")
    plt.ylabel("Kinetic Energy")
    plt.tight_layout()
    plt.title("Energy Spectrum")
    plt.legend()
    plt.savefig("wind_spectrum.png", dpi=1000, transparent=True, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    test_wind_timesteps = [2889]
    data_type = 'wind'
    component = None
    for comp in ['ua', 'va']:
            for timestep in test_wind_timesteps:
                for i in range(256):
                    compare_output_helper(data_type, comp, timestep, i)
    plot_energy_spectra()