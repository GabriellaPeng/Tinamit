import numpy as np
from tinamit.Análisis.Calibs import _conv_xr
from tinamit.Calib.ej.obs_patrón import read_obs_csv

path = "D:\Gaby\\" + "\Dt\Calib\\"
calib = "D:\Gaby\\" +  "calib.csv"
valid = "D:\Gaby\\" + 'valid.csv'

vr = 'mds_Watertable depth Tinamit'

ori_calib = read_obs_csv(calib)
ori_valid = read_obs_csv(valid)

warmup_period = None
c_poly = np.asarray([p for p in _conv_xr(ori_calib, vr, warmup_period)['x0'].values])
v_poly = np.asarray([p for p in _conv_xr(ori_valid, vr, warmup_period)['x0'].values])