import operator

from matplotlib import pyplot
import numpy as np
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
# calib_dream = np.load("D:\Gaby\Tinamit\Dt\Calib\\real_run\\revserse_res\\calib_reverse-dream.npy").tolist()
# gard = "D:\Gaby\Tinamit\Dt\Calib\\real_run\\cluster\\"
from tinamit.Análisis.Valids import _plot_poly, _label
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid
from tinamit.Calib.ej.info_paráms import _soil_canal


def load_path(cls, type, method):
    gard_point = "D:\Gaby\Tinamit\Dt\Calib\\real_run\\nse\\"
    calib_point = np.load(gard_point + f'calib-{method}_nse.npy').tolist()
    point_ind = np.argsort(calib_point['prob'])[-20:]
    valid_point = np.load(gard_point + f'{method}-nse.npy').tolist()

    gard = f"D:\Gaby\Tinamit\Dt\Calib\\real_run\\{cls}\\"
    # n = 7
    # calib_abc = np.load(gard+f'calib_cluster.npy').tolist()
    calib_dream = np.load(gard + f'calib-{method}.npy').tolist()
    all_tests = np.load(gard + f"{type}21-{method}-aic.npy").tolist()  #
    calib_ind = np.argsort(calib_dream['prob'])[-20:]
    vr = 'mds_Watertable depth Tinamit'
    # sim_eq_obs = "D:\Thesis\pythonProject\localuse\Dt\Calib\cali_res\\reverse\\t_sim_all\\all\\all_sim_eq_obs.npy"
    d_trend = np.load(gard + f'{type}7-{method}-detrend.npy').tolist()
    trend_multi = np.load(gard + f'{type}21-{method}-trend.npy').tolist()
    trend_barlas = np.load(gard + f'{type}7-{method}-trend.npy').tolist()
    # agreement = np.load(gard + f'rev_agreemt-fscabc-coeffienct of agreement.npy').tolist()
    agreement = np.load(gard + f'agreemt-{method}-coeffienct of agreement.npy').tolist()
    return all_tests, vr, calib_ind, trend_multi, trend_barlas, d_trend, agreement, calib_dream, gard


def detect_21(calib_dream, all_tests, vr, obj_func, calib_ind, relate, threshold):
    obj = {}
    if obj_func == 'aic_21':
        aic = {i: calib_dream['prob'][i] for i in calib_ind}
    for p, vec in all_tests[vr][f'{obj_func}'].items():
        if obj_func == 'aic_21':
            if p not in obj:
                obj[p] = []
            # obj[p].extend([(key, value) for (key, value) in sorted(vec.items(), key=lambda x: x[1])][-threshold:])
            obj[p].extend([(key, value) for (key, value) in
                           sorted({n: v for n, v in vec.items() if n in [f'n{i}' for i in np.sort(calib_ind)]
                                   and v - aic[int(n[1:])] > 2}.items(), key=lambda x: x[1])])

        else:
            for n, v in vec.items():
                if obj_func == 'kappa':
                    if all(i == threshold for i in list(v.values())):
                        if p not in obj:
                            obj[p] = []
                        obj[p].append(int(n[1:]))
                elif relate(v, threshold):
                    if p not in obj:
                        obj[p] = []
                    obj[p].append((n, v))
    obj['n'] = []
    if obj_func == 'kappa':
        for i in np.asarray(list(set([n for p in obj for n in obj[p]]))):  # 69n, all=1
            if i in calib_ind:
                obj['n'].append(i)
    else:
        for i in [int(n[0][1:]) for p in obj for n in obj[p]]:
            if i in calib_ind:
                obj['n'].append(i)
    obj['n'] = set(obj['n'])
    if obj_func == 'aic_21':
        return aic, obj
    else:
        return obj


def coa(vr, calib_ind, agreement):
    kap = {}
    icc1 = {}
    kappa = agreement[vr]['kappa']
    icc = agreement[vr]['icc']
    for p in kappa:
        for n, v in kappa[p].items():
            if v['ka_slope'] == 1 and int(n[1:]) in calib_ind:
                if p not in kap:
                    kap[p] = []
                kap[p].append(n)
    kap['n'] = set([n for p in kap for n in kap[p]])
    kap['slope'] = all(v['ka_slope'] for p in kappa if p != 'n' for n, v in kappa[p].items() if
                       int(n[1:]) in calib_ind and v['ka_slope'] == 1)

    for p in icc:
        for n in icc[p]:
            if icc[p][n] > 0.75 and int(n[1:]) in calib_ind:
                if p not in icc1:
                    icc1[p] = []
                icc1[p].append(n)
    icc1['n'] = set([n for p in icc1 for n in icc1[p]])
    icc1['all>0'] = all(v for p in icc if p != 'n' for n, v in icc[p].items() if int(n[1:]) in calib_ind and v > 0.4)
    return kap, icc1


def point_based(obj_fun, all_tests, top_n, calib_ind):
    vr = 'mds_Watertable depth Tinamit'
    obj = {'n': {}}
    if obj_fun == 'RMSE':
        for i in np.argsort(all_tests[vr][obj_fun])[top_n:]:
            if i in calib_ind:
                obj['n'].update({i: all_tests[vr][obj_fun][i]})
    else:
        for i in np.argsort(all_tests[vr][obj_fun])[-top_n:]:
            if i in calib_ind:
                obj['n'].update({i: all_tests[vr][obj_fun][i]})
    return obj


def barlas(path, calib_ind, key=False, trend=False, vr=False, plot=False):
    out = np.load(path).tolist()
    if trend:
        n = [n for n in out[key] if int(n[1:]) in calib_ind]
        p = set(p for n1 in n for p in out[key][n1])
        pout = out[key]
    elif key == 'corr_sim':
        p = list(set(p for p in out[key] for n in out[key][p] if int(n[1:]) in calib_ind))  # 9
        n = set(n for p in out[key] for n in out[key][p] if int(n[1:]) in calib_ind)  # 63
        pout = out[key]
    elif key == 'multi_behavior_tests':
        n = set(n for p in out[vr][key] for n in out[vr][key][p] if n[-1] != '|' and int(n[1:]) in calib_ind)  # 10
        p = list(set(p for p in out[vr][key] for n in out[vr][key][p] if n[-1] != '|' and int(n[1:]) in calib_ind))
        pout = out[vr][key]
    else:
        n = set(n for p in out for n in out[p] if n[-1] != '|' and int(n[1:]) in calib_ind)  # 10
        p = list(set(p for p in out for n in out[p] if n[-1] != '|' and int(n[1:]) in calib_ind))
        pout = out
    if plot:
        p_n = [(pp, int(nn[1:])) for pp in pout for nn in list(n) if nn in pout[pp]]
        return p_n
    else:
        return n, p


# sub_4_plot = [only_b, only_aic, both_b_aic, only_k, both_b_k, both_aic_k, all]
def plot_venn(top_val, save_path, set_labels=('Multi-step tests', 'AIC', 'KAPPA')):
    # top_val = [11, 51, 4, 47, 25, 4, 1]
    loc = ('100', '010', '110', '001', '101', '011', '111')
    new_sub = {}
    for i, val in enumerate(loc):
        new_sub[val] = top_val[i]

    pyplot.ioff()
    v = venn3(subsets=new_sub, set_labels=set_labels)
    # v.get_label_by_id(loc[0]).set_text('best is n123')
    # v.get_label_by_id(loc[1]).set_text('best is n564')
    # v.get_label_by_id(loc[2]).set_text(f'overlapped 5')
    # # v.get_label_by_id(loc[3]).set_text(' ')
    # v.get_label_by_id(loc[4]).set_text('overlapped 25 simulations')
    # v.get_label_by_id(loc[5]).set_text('overlapped 5(include n564)')
    # v.get_label_by_id(loc[6]).set_text('478')
    pyplot.annotate('69 simulations are equally the best',
                    xy=v.get_label_by_id(loc[3]).get_position() - np.array([0, 0.05]),
                    ha='center', xytext=(0.1, 0.0001), textcoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='gray'))

    pyplot.annotate('485-th run', xy=v.get_label_by_id(loc[6]).get_position() - np.array([0, 0.05]),
                    ha='center', xytext=(0.8, 0.2), textcoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='gray'))

    pyplot.annotate('485-th run', xy=v.get_label_by_id(loc[6]).get_position() - np.array([0, 0.05]),
                    ha='center', xytext=(0.8, 0.2), textcoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='gray'))

    pyplot.savefig(save_path)


def plot_top_sim(obj_func, trend_multi, trend_barlas, save_plot, calib_ind, all_tests, gard, type, method, calib_dream):
    vr = 'mds_Watertable depth Tinamit'
    if obj_func == 'multi_behavior_tests':
        p_n = barlas(gard + f'{type}7-{method}-phase.npy', calib_ind, plot=True)
        obj_func = 'Muti-behaviour tests'
        trend = trend_barlas
    elif obj_func == 'AIC':
        AIC = point_based('AIC', all_tests, 20)
        p_n = [(p, n) for n in AIC['n'] for p in trend_multi['t_sim'][f'n{n}']]
        trend = trend_multi
    else:
        p_n = []
        if obj_func == 'aic_21':
            aic, obj_21 = detect_21(calib_dream, all_tests, vr, 'aic_21', calib_ind, operator.gt, 5)
        elif obj_func == 'kappa':
            obj_21 = detect_21(all_tests, vr, 'kappa', calib_ind, operator.eq, 1)
            p_n = [(p, n) for p in obj_21 if not isinstance(p, str) for n in obj_21['n'] if n in obj_21[p]]
        elif obj_func == 'rmse_21':
            obj_21 = detect_21(all_tests, vr, 'rmse_21', calib_ind, operator.lt, 0.4)
        elif obj_func == 'nse_21':
            obj_21 = detect_21(all_tests, vr, 'nse_21', calib_ind, operator.gt, 0.7)
        if not len(p_n):
            p_n = [(p, n) for p in obj_21 if not isinstance(p, str) for n in list(obj_21['n']) if
                   f'n{n}' in [i[0] for i in obj_21[p]]]
        obj_func = obj_func[:3]
        trend = trend_multi
    mismatch_sim = {obj_func: []}
    for pn in p_n:
        obs_yred = trend['t_obs'][pn[0]]['y_pred']
        if f'n{pn[1]}' in trend['t_sim']:
            sim_yred = trend['t_sim'][f'n{pn[1]}'][pn[0]]['y_pred']
        else:
            mismatch_sim[obj_func].append(pn)
        pyplot.ioff()
        pyplot.plot(obs_yred, 'g--', label=f"obs_poly{pn[0]}")
        pyplot.plot(sim_yred, 'r-.', label=f"sim_{pn[1]}")

        pyplot.legend()
        pyplot.title(f'{obj_func}-poly{pn[0]}-sim{pn[1]} Vs Obs{pn[0]}')
        pyplot.savefig(save_plot + f'{obj_func}_poly{pn[0]}_sim{pn[1]}')
        pyplot.close('all')
    return mismatch_sim


def prep_cluster_dt(obj_func, vr, sim_eq_obs):
    obj = sim_eq_obs[vr][obj_func]
    cluster = np.empty([len(obj), len(list(obj.values())[0])])
    for p in obj:
        if obj_func == 'kappa':
            cluster[list(obj).index(p), :] = np.asarray([list(v.values())[0] for v in list(obj[p].values())])
        else:
            cluster[list(obj).index(p), :] = np.asarray(list(obj[p].values()))
    return cluster


def path_4_plot(plot_type, mtd, obj_func, p_m=None):
    vr = 'mds_Watertable depth Tinamit'

    save_plot = "D:\Gaby\Tinamit\Dt\Calib\plot\\"
    res_path = "D:\Gaby\Tinamit\Dt\Calib\\real_run\\"

    obs_param = np.load(res_path + 'obss.npy').tolist()

    c_obs_dt = ori_calib[1]
    v_obs_dt = ori_valid[1]

    if plot_type == 'prm_prb':
        calib_res = {}
        for m in mtd:
            calib_res[m] = np.load(res_path + f'{m}\\{m}_{obj_func}.npy').tolist()

        if 'rev' in obj_func:
            poly = np.asarray(list(v_obs_dt))
        else:
            poly = np.asarray(list(c_obs_dt))

        return poly, calib_res, obj_func, obs_param, save_plot + plot_type + f'\\{obj_func}'

    elif plot_type == 'valid':
        prob = {}
        for m in mtd:
            if p_m == 'patrón':
                prob[m] = np.load(res_path + f'{m}\\valid_{obj_func}_patrón.npy').tolist()[vr][
                    obj_func[:obj_func.index('_')] if '_' in obj_func else obj_func]
            else:
                prob[m] = np.load(res_path + f'{m}\\valid_{obj_func}_multidim.npy').tolist()[vr][
                    obj_func[:obj_func.index('_')] if '_' in obj_func else obj_func]

        if 'rev' in obj_func:
            poly = np.asarray(list(c_obs_dt))
        else:
            poly = np.asarray(list(v_obs_dt))

        return poly, prob, obj_func, obs_param, save_plot + f'valid\\{obj_func}'


def combine_calib_res(res, method, obj_func=None):
    '''
    :param res: [res1, res2]
    :param method: ['abc', 'mle']
    :return:
    '''
    for pm in ['POH Kharif Tinamit', 'POH rabi Tinamit', 'Capacity per tubewell']:
        for m in method:
            if 'sampled_prm' in res[m]:
                res[m]['sampled_prm'][pm] = np.asarray([j.tolist() for j in res[m]['sampled_prm'][pm]])
            res[m][pm] = np.asarray([j.tolist() for j in res[m][pm]])
    for m in method:
        res[m].update({p: np.asarray(res[m][p]) for p, v in res[m].items() if isinstance(v, list)})

    d_param = {m: {} for m in method}

    for m in method:
        d_param[m] = {p: v for p, v in res[m].items() if len(v) == len(res[m]['buenas']) and p != 'buenas'}

    if 'aic' in obj_func:
        prob = {m: np.negative(np.take(res[m]['prob'], res[m]['buenas'])) for m in method}
    else:
        prob = {m: np.take(res[m]['prob'], res[m]['buenas']) for m in method}
    return d_param, prob

def clr_marker(color=False, marker=False):
    if color:
        return {'fscabc': 'b', 'dream': 'orange', 'mle': 'r', 'demcz': 'g'}
    elif marker:
        return ['o', 'v', "x", "*"]

def plot_prm_prb(obj_func, mtd):
    '''

    :param obj_func: if _rev in obj_func then poly needs to be v_poly
    :param poly:
    :param mtd:
    :return:
    '''

    poly, calib_res, obj_func, obs_param, save_plot = path_4_plot('prm_prb', mtd, obj_func)

    d_param, prob = combine_calib_res(calib_res, mtd, obj_func)

    s_cnl_msk = _soil_canal(poly)
    prb = [i for m, l in prob.items() for i in l]

    x = [i + 1 for i in range(len(s_cnl_msk))]
    xlabels = [i[0] + i[i.index(',') + 2] for i in s_cnl_msk]

    for p in obs_param:
        y = [d_param[m][p] for m in d_param if not isinstance(obs_param[p], np.ndarray) and p != 'opt']
        if len(y):
            mds_dist = {}
            for i, m in enumerate(prob):
                plt.ioff()
            #     plt.scatter(prob[m], y[i], alpha=0.5, marker='o', label=f'{m}', c=f'{clr_marker(color=True)[m]}')
                mds_dist[m] = np.abs(np.nanmean(y[i]) - obs_param[p])
            # plt.hlines(obs_param[p], xmin=min(prb), xmax=max(prb), colors='g', label='Previously calibrated value')
            # _label(f"{obj_func[:obj_func.index('_')].upper() if '_' in obj_func else obj_func}", f"{p}")
            # _plot_poly(p, '', save_plot)

            for i, m in enumerate(prob):
                plt.plot(i+1, mds_dist[m], marker='o', color=f'{clr_marker(color=True)[m]}', label=m.upper())
            plt.xticks([i+1 for i in range(len(prob))], [i for i in prob], rotation=70, fontsize=6)
            _label('Calibration approaches', f"Distance of the mean value of calibrating {p[:p.index('-')] if '-' in p else p} to the previous optimum value",
                   f"Differences of mean {p[:p.index('_')] if '_' in p else p} to the optimum to calibration approaches", fontsize=6)
            _plot_poly(p, '_mds_dist', save_plot)

        elif p != 'opt':
            bf_dist = {m: [[] for j in range(len(s_cnl_msk))] for i, m in enumerate(prob)}

            for c, l_p in s_cnl_msk.items():
                for pl in poly:
                    if pl in l_p:
                        for m in prob:
                            plt.ioff()
                            # plt.scatter(prob[m], d_param[m][p].T[pl - 1], alpha=0.5, marker='o', label=f'{m}', c=colors[m])
                            if p.startswith('K'):
                                bf_dist[m][list(s_cnl_msk).index(c)].append(np.abs(np.nanmean(d_param[m][p].T[pl - 1]) - obs_param['opt'][p]))
                            else:
                                bf_dist[m][list(s_cnl_msk).index(c)].append(np.abs(np.nanmean(d_param[m][p].T[pl - 1]) - obs_param[p][pl - 1]))

                        # _label(f"{obj_func[:obj_func.index('_')].upper() if '_' in obj_func else obj_func}", f"{p.capitalize()}", f"Poly{pl}, Condition:{c}")
                        # plt.hlines(obs_param[p][pl - 1], xmin=min(prb), xmax=max(prb), colors='g',
                        #            label='Previously calibrated value')
                        # if p.startswith('K'):
                        #     plt.hlines(obs_param['opt'][p], xmin=min(prb), xmax=max(prb), colors='r',
                        #                linestyles="dashed", label='Previously optimized value')
                        # _plot_poly(p, f'{pl}', save_plot)
            a=[]
            for i, m in enumerate(prob):
                for xe, ye in zip(x, bf_dist[m]):
                    plt.ioff()
                    a.append(plt.scatter([xe] * len(ye), ye, alpha=0.3, marker=f'{list(clr_marker(marker=True))[i]}',
                                label=y, color=f'{clr_marker(color=True)[m]}'))

            plt.xticks(x, xlabels, rotation=70, fontsize=10)
            _label('Soil Class & Canal Position', f"Distance of the mean calibrated{p[:p.index('-')] if '-' in p else p} "
            f"to old optimum value", f" Differences of mean {p[:p.index('-')] if '-' in p else p} to the optimum to physical conditions")
            labl = [m for m in prob]
            ind = np.searchsorted([i for i, m1 in enumerate(prob) for m in [p.get_label() for p in a] if m1 == m],
                                  [i for i in range(len(labl))], side='right')
            plt.legend([a[i - 1] for i in ind], labl)
            _plot_poly(p, '_bf_dist', save_plot)


def prb_cls(obj_func, mtd, p_m):
    poly, prob, obj_func, obs_param, save_plot = path_4_plot('valid', mtd, obj_func, p_m)
    s_cnl_msk = _soil_canal(poly)
    objfc = [obj_func[:obj_func.index('_')].upper() if '_' in obj_func else obj_func.upper()]

    x = [i + 1 for i in range(len(s_cnl_msk))]

    def _plot():
        plt.xticks(x, [i[0] + i[i.index(',') + 2] for i in s_cnl_msk], rotation=70, fontsize=10)
        _label('Soil Class & Canal Position', f"{objfc[0].upper()}", f"{objfc[0]} to Validation Polygons")

    m_y = {m: [[] for j in range(len(s_cnl_msk))] for i, m in enumerate(prob)}

    for c, l_p in s_cnl_msk.items():
        for i, pl in enumerate(poly):
            if pl in l_p:
                for j, m in enumerate(prob):
                    m_y[m][list(s_cnl_msk).index(c)].append(prob[m][i])

    a = []
    for i, y in enumerate(m_y):
        for xe, ye in zip(x, m_y[y]):
            plt.ioff()
            a.append(
                plt.scatter([xe] * len(ye), ye, alpha=0.3, marker=f'{list(clr_marker(marker=True))[i]}', label=y,
                            color=f'{clr_marker(color=True)[y]}'))

    _plot()
    labl = [m.upper() for m in prob]
    ind = np.searchsorted([i for i, m1 in enumerate(prob) for m in [p.get_label() for p in a] if m1 == m],
                          [i for i in range(len(labl))], side='right')
    plt.legend([a[i - 1] for i in ind], labl)
    _plot_poly('valid', f'', save_plot)

    for i, m in enumerate(mtd):
        plt.plot(x, [np.mean(np.asarray(v)) for v in m_y[m]], label=m.upper(), color=f'{clr_marker(color=True)[m]}',
                 marker=f'{list(clr_marker(marker=True))[i]}', linestyle='dashed')
    _plot()
    _plot_poly('valid', f'_mean', save_plot)

    for i, m in enumerate(mtd):
        plt.plot(x, [np.nanmedian(np.asarray(v)) for v in m_y[m]], label=m.upper(), color=f'{clr_marker(color=True)[m]}',
                 marker=f'{list(clr_marker(marker=True))[i]}', linestyle='dashed')
    _plot()
    _plot_poly('valid', f'_median', save_plot)


aic_mtd = ['fscabc', 'mle', 'dream']
aic_rev_mtd = ['fscabc', 'dream', 'demcz', 'mle']

nse_rev_mtd = ['fscabc', 'dream']

obj_func = 'nse'
p_m = 'multidim'  # multidim, patrón

# plot_prm_prb(obj_func, mtd=['fscabc'])
prb_cls(obj_func, aic_mtd, p_m) # if obj_fc == aic--> v_poly// rev==c_poly
