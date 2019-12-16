import os

import numpy as np
from xarray import Dataset

from tinamit.Análisis.Calibs import gen_gof, aplastar, patro_proces
from tinamit.Calib.ej.cor_patrón import ori_valid, ori_calib
from tinamit.cositas import cargar_json

path = "C:\\Users\\umroot\\Desktop\map\sim_obs_check\\"
vr = 'mds_Watertable depth Tinamit'

valid_calib = 'valid'

if valid_calib == 'calib':
    calib_poly = list(ori_valid[1])
    calib_res = np.asarray([Dataset.from_dict(cargar_json(os.path.join(path, 'N1')))[vr].values[:, j - 1] for j in calib_poly]) #19*41

    obs = np.asarray([v for i, v in ori_valid[1].items()]).astype(float) #19*41
    mu_obs, sg_obs, norm_obs = aplastar(calib_poly, obs.T)

    eval = np.load(path+'calib_eval.npy').tolist()
    normeval =np.load(path+'norm_calib_eval.npy').tolist()

    sim = calib_res
    normsim = ((sim.T - mu_obs) / sg_obs)

else:
    valid_poly = list(ori_calib[1])
    valid_res = np.asarray([Dataset.from_dict(cargar_json(os.path.join(path, 'N1')))[vr].values[:, j - 1] for j in valid_poly])

    obs = np.asarray([v for i, v in ori_calib[1].items()]).astype(float)
    mu_obs, sg_obs, norm_obs = aplastar(valid_poly, obs.T)

    eval = patro_proces('patrón', valid_poly, obs, obj_func='aic')
    normeval = patro_proces('patrón', valid_poly, norm_obs, obj_func='aic')

    sim = valid_res
    normsim = ((sim.T - mu_obs) / sg_obs)


gen_gof('patrón', sim, eval, obj_func='aic', obs = obs.T)