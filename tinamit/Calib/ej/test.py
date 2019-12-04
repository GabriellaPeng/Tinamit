import numpy as np
# vr = 'mds_Watertable depth Tinamit'
# mtd = ['mle', 'fscabc', 'demcz', 'dream']
# data  = {m:{}for m in mtd}
#
# for m in mtd:
#     for obj_func in ['aic_rev', 'rmse_rev', 'nse_rev']:
#         if 'aic' in obj_func:
#             obj_func = 'aic_rev_patrón'
#         else:
#             obj_func = f'{obj_func}_multidim'
#
#         path  = "D:\Gaby\Tinamit\Dt\Calib\\real_run\\" + m+f'\\valid_{obj_func}_like.npy'
#         data[m][obj_func] = np.load(path).tolist()[vr]
#

# def aic(a, b):
#     print('aic', a, b)
#
# def bic(a, b):
#     print('bic', a, b)
#
# fc = {'aic': aic, 'bic': bic}

# fc['bic'](1, 2)
# vars()['aic'](1, 2)
from tinamit.Análisis.Valids import proc_prob_data
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid, vr
from tinamit.Calib.ej.ej_calib.calib_análisis import plot_top_sim_obs
from tinamit.Calib.ej.sens_análisis import criteria_stat, hist_conv

# f_simul= np.load("D:\Gaby\Tinamit\Dt\Mor\\f_simul\\aug\\f_simul_550.npy").tolist()
# all_beh_dt = np.load("D:\Gaby\Tinamit\Dt\Mor\\f_simul\\aug\\New folder\\all_beh_dt.npy").tolist()
# poly18, poly19 = list(ori_calib[1]), list(ori_valid[1])
# print()

criteria_stat(625, "D:\Gaby\Tinamit\Dt\Mor\\f_simul\\aug\\f_simul_", "D:\Gaby\Tinamit\Dt\Mor\simular\\625_mor\\")

# poly = np.sort(np.concatenate((list(ori_calib[1]), list(ori_valid[1]))))
# gof_stat = np.load("D:\Gaby\Tinamit\Dt\Mor\\gof_stat.npy").tolist()
#
# gof_type=['aic', 'bic', 'mic', 'srm', 'press', 'fpe']
#
# coverage = {gof: np.zeros(len(poly)) for gof in gof_type}
# linear = {gof: np.zeros(len(poly)) for gof in gof_type}
# rmse = {gof: {'mean': np.zeros(len(poly)), 'std': np.zeros(len(poly))} for gof in gof_type}
# for gof in gof_type:
#     for i, p in enumerate(poly):
#         coverage[gof][i] = np.average(gof_stat[gof]['converage'][:, i])
#         linear[gof][i] = np.count_nonzero(gof_stat[gof]['linear'][:, i])/625
#         rmse[gof]['mean'][i] =  np.average(gof_stat[gof]['rmse'][:, i])
#         rmse[gof]['std'][i] = np.std(gof_stat[gof]['rmse'][:, i])


path = "D:\Gaby\Tinamit\Dt\Calib\\real_run\\"
# proc_sim = np.load(path + 'fscabc\\valid_aic.npy').tolist()
# obs_norm = np.load(path + 'valid_obs_norm.npy').tolist()
# percentile = np.load(path + 'ci.npy').tolist()
# npoly = np.load(path + 'npoly.npy').tolist()
#
# mask = proc_prob_data(proc_sim['prob'], 'aic', top_N=100)[1]
# plot_top_sim_obs(proc_sim["all_sim"][mask], obs_norm[vr], npoly, path, proc_sim, percentile,
#                  l_poly=[36, 71, 175])

gof_type=['aic', 'bic', 'mic', 'srm', 'press', 'fpe']
data = np.load("D:\Gaby\Tinamit\Dt\Mor\\gof_stat.npy").tolist()
hist_conv(gof_type, data, save="D:\Gaby\Tinamit\Dt\Mor\map\hist\\")
print()


