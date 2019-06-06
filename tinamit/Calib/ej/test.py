import os
import numpy as np
from xarray import Dataset
from tinamit.Análisis.Valids import _plot_poly
from tinamit.cositas import cargar_json
from matplotlib import pyplot
from tinamit.Calib.ej.cor_patrón import ori_valid, ori_calib

var = 'mds_Watertable depth Tinamit'
m='abc'
if m == 'abc':
    n_sim = (0, 145)
    sim= "D:\Thesis\pythonProject\localuse\Dt\Calib\simular\\new\\fscabc\\aic_old\\"
    aic = np.load("D:\Thesis\pythonProject\localuse\Dt\Calib\cali_res\\abc\\May-11\\aic.npy").tolist()
    aic_rev = np.load("D:\Thesis\pythonProject\localuse\Dt\Calib\cali_res\\abc\\May-11\\aic_rev.npy").tolist()
else:
    n_sim = (0, 495)
    sim = "D:\Thesis\pythonProject\localuse\Dt\Calib\simular\\new\\old_dream\dream\\aic\\"
    aic = np.load("D:\Thesis\pythonProject\localuse\Dt\Calib\cali_res\\dream\\May-07\\aic.npy").tolist()
    aic_rev = np.load("D:\Thesis\pythonProject\localuse\Dt\Calib\cali_res\\dream\\May-07\\aic_rev.npy").tolist()

val1, val2= np.sort(aic['prob'])[-10:-3], np.sort(aic_rev['prob'])[-10:-3]
shp = np.zeros([20, 41, 18])

ind= np.argsort(aic_rev['prob'])[-10:-3]
val = np.sort(aic_rev['prob'])[-10:-3]
ind_poly  = np.asarray([p-1 for p in ori_calib[1]])

for i, v in enumerate(ind):
    for p in ori_calib[1]:
        pyplot.ioff()
        pyplot.plot(ori_calib[1][p],  label=f'obs_{p}')
        pyplot.plot(Dataset.from_dict(cargar_json(os.path.join(sim, f'{v+n_sim[0]}')))[var].values[:,p-1], label=f'sim_{v}-{p}')
        _plot_poly(f'{p}', f'sim{v}', "D:\Thesis\pythonProject\localuse\Dt\Calib\plot\\test\\dream\\aic_rev\\")

    for t in range(41):
        shp[i, t, :] = np.take(Dataset.from_dict(cargar_json(os.path.join(sim, f'{v+n_sim[0]}')))[var].values[t, :], ind_poly)
print(shp)