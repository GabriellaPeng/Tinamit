import operator
import numpy as np

from tinamit.Análisis.Valids import _plot_poly
from tinamit.Calib.ej.ej_calib.calib_análisis import detect_21, point_based, barlas, plot_top_sim, prep_cluster_dt, coa, load_path  # , sim_eq_obs
from tinamit.Calib.ej.info_paráms import _soil_canal
from tinamit.Calib.ej.sens_análisis import clustering
import matplotlib.pyplot as plt

def load_results(cls, type, method):
    all_tests, vr, calib_ind, trend_multi, trend_barlas, d_trend, agreement, calib_dream, gard = load_path(cls, type, method)

    kap, icc = coa(vr, calib_ind, agreement)
    aic, aic_21 = detect_21(calib_dream, all_tests, vr, 'aic_21', calib_ind, operator.gt, 5)
    rmse_21 = detect_21(calib_dream, all_tests, vr, 'rmse_21', calib_ind, operator.lt, 0.5)
    nse_21 = detect_21(calib_dream, all_tests, vr, 'nse_21', calib_ind, operator.gt, 0.65)

    AIC = point_based('AIC', all_tests, 20, calib_ind)
    NSE = point_based('NSE', all_tests, 20, calib_ind)
    RMSE = point_based('RMSE', all_tests, 20, calib_ind)

    t_n, t_p = barlas(gard + f'{type}7-{method}-trend.npy', calib_ind, 't_sim', trend=True)
    corr_n, corr_p = barlas(gard + f'{type}7-{method}-Rk.npy', calib_ind, 'corr_sim')

    diff = np.load(gard + f'{type}7-{method}-autocorr_variables.npy').tolist()
    diff_n = set(n for p in diff['pass_diff'] for n in diff['pass_diff'][p] if int(n[1:]) in calib_ind)  # 63

    mean_n, mean_p = barlas(gard + f'{type}7-{method}-Mean.npy', calib_ind)
    amp_n, amp_p = barlas(gard + f'{type}7-{method}-Std.npy', calib_ind)
    phase_n, phase_p = barlas(gard + f'{type}7-{method}-phase.npy', calib_ind)
    u_n, u_p = barlas(gard + f'{type}7-{method}-multi_behavior_tests.npy', calib_ind, 'multi_behavior_tests', vr=vr)

    print(plot_top(trend_multi, trend_barlas, calib_ind, all_tests, gard, type, method, calib_dream))

    return {'21': [aic_21, rmse_21, nse_21], 'point_based': [AIC, NSE, RMSE],
            'barlas': [t_n, t_p, d_trend, corr_n, corr_p, phase_n, phase_p],
            'coa': [kap, icc]}

def plot_top(trend_multi, trend_barlas, calib_ind, all_tests, gard, type, method, calib_dream):
    trend_multi = trend_multi
    trend_barlas = trend_barlas
    save_plot = f"D:\Gaby\Tinamit\Dt\Calib\plot\\{type}\\"
    mismatch = []
    for obj in ['aic_21', 'kappa', 'rmse_21', 'nse_21', 'multi_behavior_tests', 'AIC']:
        mismatch.append(plot_top_sim(obj, trend_multi, trend_barlas, save_plot, calib_ind, all_tests, gard, type, method, calib_dream))
    return mismatch

def prep_cluster(sim_eq_obs):
    vr = 'mds_Watertable depth Tinamit'
    sim_eq_obs = np.load(sim_eq_obs).tolist()
    polys = np.asarray(list(sim_eq_obs['mds_Watertable depth Tinamit']['kappa']))

    aic_21 = prep_cluster_dt('aic_21', vr, sim_eq_obs)
    d_cls = clustering(aic_21, n_cls=2, valid=True)['d_km']
    aic_21 = {0: polys[d_cls[0]], 1:polys[d_cls[1]]}

    rmse_21 = prep_cluster_dt('rmse_21', vr, sim_eq_obs)
    d_cls = clustering(rmse_21, n_cls=2, valid=True)['d_km']
    rmse_21 = {0: polys[d_cls[0]], 1: polys[d_cls[1]]}

    nse_21 = prep_cluster_dt('nse_21', vr, sim_eq_obs)
    d_cls = clustering(nse_21, n_cls=2, valid=True)['d_km']
    nse_21 = {0: polys[d_cls[0]], 1: polys[d_cls[1]]}

    kappa = prep_cluster_dt('kappa', vr, sim_eq_obs)
    d_cls = clustering(kappa, n_cls=2, valid=polys)['d_km']
    kappa = {0: polys[d_cls[0]], 1: polys[d_cls[1]]}
    return aic_21, rmse_21, nse_21, kappa

# aic_21, rmse_21, nse_21, kappa = prep_cluster(sim_eq_obs)


# top_nse_21 = max(v for p in nse_21 if p != 'n' for n, v in nse_21[p].items())
# top_aic_21 = max(v for p in aic_21 if p != 'n' for n, v in aic_21[p].items())  # n_564
# top_rmse_21 = min(v for p in rmse_21 if p != 'n' for n, v in rmse_21[p].items())

# AIC_barlas = []
# rmse_21_barlas = []
# aic_barlas = []
# aic_kapp = []
# all = []
# for n in phase_n:  # 41
#     if n in AIC['n']:  # 60
#         aic_barlas.append(n)
# for n in kapp_21['n']:  #
#     if n in AIC['n']:
#         aic_kapp.append(n)
# for n in kapp_21['n']:
#     if n in phase_n and n in AIC['n']:
#         all.append(n)
#
# only_b = sorted([i for i in [i for i in phase_n if int(i[1:]) not in AIC['n']] if i not in kapp_21['n']])
# only_k = sorted([i for i in [i for i in kapp_21['n'] if i not in AIC['n']] if i not in phase_n])
# only_aic = sorted([i for i in [i for i in AIC['n'] if i not in phase_n] if i not in kapp_21['n']])
#
# both_b_k = sorted([i for i in phase_n if i in kapp_21['n']])
# both_b_aic = sorted([i for i in u_n if i in AIC['n']])
# both_aic_k = sorted([i for i in kapp_21['n'] if i in AIC['n']])

