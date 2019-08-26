import numpy as np
from tinamit.Análisis.Calibs import _conv_xr
from tinamit.Calib.ej.obs_patrón import read_obs_csv

path = "C:\\Users\\umroot\\OneDrive - Concordia University - Canada\\gaby\\pp2_data\\calib\\"
calib = path +  "calib.csv"
valid = path + 'valid.csv'

vr = 'mds_Watertable depth Tinamit'

ori_calib = read_obs_csv(calib)
ori_valid = read_obs_csv(valid)

c_poly = np.asarray([p for p in _conv_xr(ori_calib, vr, 2)['x0'].values])
v_poly = np.asarray([p for p in _conv_xr(ori_valid, vr, 2)['x0'].values])

