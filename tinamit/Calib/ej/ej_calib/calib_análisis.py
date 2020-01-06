import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.gridspec import GridSpec
import seaborn as sns

from matplotlib.patches import Polygon
from tinamit.Calib.ej.info_paráms import _soil_canal

from xarray import Dataset

from tinamit.Análisis.Sens.behavior import superposition, simple_shape
from tinamit.Calib.ej.cor_patrón import ori_valid, ori_calib

from tinamit.cositas import cargar_json

def plot_save(p, name, save_plot):
    handles, labels = plt.gca().get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list)
    plt.savefig(save_plot + f'{name}_{p}')
    plt.close('all')
    plt.clf()


def _label(xlabel, ylabel, title=None, fontsize=None):
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    if title is not None:
        plt.title(title, fontsize=fontsize)


def plot_ci(x_data, y_data, label_poly, save_plot, fig, ax,  ind_ax=None):
    if isinstance(label_poly, list):
        label_poly = label_poly[0]

    x_data = np.insert(x_data, 0, 0)
    y_data = np.insert(y_data, 0, 0)
    y_data[np.isnan(y_data)] = 1

    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g-.', label="CI over Percentiles")
    ax.plot(x_data, y_data, 'r.-', label=f"{label_poly} CI")
    ax.set_xticks(x_data[1:])

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    # plt.xticks(np.arange(0, 1.1, 0.1), [np.round(i, 2) for i in np.arange(0, 1.1, 0.1)])
    # for i in x_data:
    #     ax.axvline(i, color='grey', alpha=0.4, linestyle='--')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    if ind_ax is not None and ind_ax == 0:
        ax.set_yticklabels([np.round(i, 2) for i in np.arange(0, 1.1, 0.1)], size=15)
    else:
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    if ind_ax is None:
        fig.savefig(save_plot + f'ci_{label_poly}')
        plt.close('all')
        plt.clf()


def _save_dist(ax, fig, type_density, dpi=500, save_plot=False):
    if 'parameter' in type_density:
        v = type_density['parameter']
    if 'polygon' in type_density:
        v = f"polygon {type_density['polygon']}"
    elif 'obj_func' in type_density:
        v = type_density['obj_func'].upper()

    ax.set_title(f'Shaded density plot of {v}',fontsize=8)

    if save_plot:
        fig.savefig(save_plot + f'{v}', dpi=dpi)


def _set_ax_marker(ax, xlabel, ylabel, title, xlim=None, ylim=None):

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    elif ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_title(title)


def plot_top_sim_obs(sim_norm, obs_norm, npoly, save_plot, proc_sim, percentails=None, l_poly=None):
    if l_poly is not None:
        ncol = len(l_poly)
        plt.ioff()
        if percentails is not None: #todo: move to new project
            for pcentl in percentails: # for pcentl, array in percentails.items()
                pcentl = np.asarray(pcentl)
                fig1 = plt.figure(figsize=(4 * 2 * ncol, 4), constrained_layout=False)
                gs1 = GridSpec(ncols=len(l_poly), nrows=1, wspace=0.0, hspace=0.0)

                fig2 = plt.figure(figsize=(4 * 2 * ncol, 4), constrained_layout=False)
                gs2 = GridSpec(ncols=len(l_poly), nrows=1, wspace=0.0, hspace=0.0)
                for i in range(ncol):
                    vars()[f'ax1' + str(i)] = fig1.add_subplot(gs1[i])
                    vars()[f'ax2' + str(i)] = fig2.add_subplot(gs2[i])

                for i in range(ncol):
                    ind_p = npoly.index(l_poly[i])
                    p_sim = {'weighted_sim': proc_sim['weighted_sim'][:, ind_p]}
                    # p_sim = {n: val[:, ind_p] for n, val in proc_sim.items() if val != 'prob'}
                    param_uncertainty_bounds(sim_norm[:, :, ind_p], obs_norm[:, ind_p], l_poly[i], p_sim, save_plot,
                                             fig1, ax=vars()[f'ax1' + str(i)], ind_ax=i)

                    plot_ci(np.arange(0.05, 1.05, 0.05), np.sort(pcentl[:, ind_p]), f'Poly-{l_poly[i]}', save_plot, fig=fig2,
                            ax=vars()[f'ax2' + str(i)], ind_ax=i)

                fig1.savefig(save_plot + f'bounds_{l_poly}')
                fig2.savefig(save_plot + f'ci_{l_poly}')

        # else:
        #     fig = plt.figure(figsize=( 4 * 2 * ncol, 4), constrained_layout=False)
        #     gs = GridSpec(ncols=len(l_poly), nrows=2, wspace=0.0, hspace=0.0)
        #     percentile = np.divide(np.arange(1, obs_norm.shape[0] + 1), obs_norm.shape[0])
        #     for j in range(2):
        #         for i in range(len(l_poly)):
        #             vars()[f'ax' + str(j) + str(i)] = fig.add_subplot(gs[j, i])
        #
        #     for j in range(2):
        #         for i in range(len(l_poly)):
        #             ind_p = np.where(npoly == l_poly[i])[0][0]
        #             if j == 0:
        #                 p_sim = {n: val[:, ind_p] for n, val in proc_sim.items()}
        #                 param_uncertainty_bounds(sim_norm[:, :, ind_p], obs_norm[:, ind_p], l_poly[i], p_sim, save_plot, fig, ax=vars()[f'ax'+str(j)+str(i)], ind_ax=i)
        #
        #             elif j ==1:
        #                 plot_ci(percentile, np.sort(pcentl[:, ind_p]), f'Poly-{l_poly[i]}', save_plot,  fig=fig, ax=vars()[f'ax' + str(j) + str(i)],  ind_ax=i)

    else:
        for ind_p, p in enumerate(npoly):
            p_sim = {n: val[:, ind_p] for n, val in proc_sim.items()}
            plt.ioff()
            fig1, ax1 = plt.subplots(figsize=(16, 9))
            param_uncertainty_bounds(sim_norm[:, :, ind_p], obs_norm[:, ind_p], p, p_sim, save_plot, fig1, ax1, None)

            # if pcentl is not None:
            #     percentile = np.divide(np.arange(1, obs_norm.shape[0] + 1), obs_norm.shape[0])
            #     plt.ioff()
            #     fig2, ax2 = plt.subplots(figsize=(11, 4))
            #     plot_ci(percentile, np.sort(pcentl[:, ind_p]), f'Poly-{l_poly[ind_p]}', save_plot, fig2, ax2)

        # if pcentl is not None:
        #     plt.figure(figsize=(15, 5))
        #     plt.plot(obs_norm[:, j], 'r-', label=f'obs_{p}', linewidth=4) #obs:t, p
        #     for i in range(len(sim_norm)):
        #         plt.plot(sim_norm[i, :, j], 'b--', linewidth=0.2)

            # plt.xticks(range(len(percentile)), [f"{i}\n{round(t * 100, 1)}%" for i, t in enumerate(pcentl[:, j])],
            #            fontsize=5)
            # for xc in range(len(percentile)):
            #     plt.axvline(x=xc, linewidth=0.2, linestyle=':')
            #_label("Season\nConfidential Interval", "Water Table Depth", 8)
            # plt.savefig(save_plot + f't_ci_{p}', dpi=250)



def path_4_plot(res_path, save_plot, plot_type, mtd, obj_func, poly_type=19, proc_sim=False, theil=False, trd_agree=False):
    '''

    :param res_path:
    :param save_plot:
    :param plot_type:
    :param mtd:
    :param obj_func:
    :param p_m:
    :param poly_type: calib=19poly='rev'; valid=18poly='ori'
    :return:
    '''
    vr = 'mds_Watertable depth Tinamit'

    if os.name == 'posix':
        obs_param = np.load(res_path + 'obs_prm.npy').tolist()
    else:
        obs_param = np.load(res_path + 'obss.npy').tolist()

    c_obs_dt = ori_calib[1]
    v_obs_dt = ori_valid[1]

    if plot_type == 'prm_prb':
        calib_res = {}
        for m in mtd:
            calib_res[m] = np.load(res_path + f'{m}\\oct\\{m}_{obj_func}.npy').tolist()

        if poly_type == 19:
            poly = np.asarray(list(v_obs_dt)) #19
        else:
            poly = np.asarray(list(c_obs_dt)) #18

        return poly, calib_res, obj_func, obs_param, save_plot + plot_type + f'\\{obj_func}'

    elif plot_type == 'valid':
        theil_stat = {m : {} for m in mtd}
        prob = {}

        if poly_type == 19:
            poly = np.asarray(list(v_obs_dt))
        else:
            poly = np.asarray(list(c_obs_dt))

        trend_agree = { }
        for m in mtd:
            if obj_func in ['aic', 'mic']:
                if trd_agree:
                    # trend_agree[m] = np.load(res_path + f'{m}\\valid_{obj_func}_patrón_wt.npy').tolist()[vr]['patrón'][obj_func]['weighted_sim']['trend_agreement']
                    trend_agree[m] = np.load(res_path + f'{m}\\valid_{obj_func}_patrón.npy').tolist()[vr]['patrón'][obj_func]['weighted_sim']['trend_agreement']

                else:
                    path = np.load(res_path + f'{m}\\nov\\valid_{obj_func}.npy').tolist()[vr]
                    prob[m] = path['patrón'][obj_func]['likes']

                    if proc_sim:
                        prob[m].update({'proc_sim': path['proc_sim']})
            else:
                if trd_agree:
                    trend_agree[m] = np.load(res_path + f"{m}\\valid_{obj_func}_multidim.npy").tolist()[vr]['multidim'][obj_func]['weighted_sim']['trend_agreement']

                else:
                    path = np.load(res_path + f'{m}\\nov\\valid_{obj_func}.npy').tolist()[vr]
                    prob[m] = path['multidim'][obj_func]['likes']

                    if proc_sim:
                        prob[m].update({'proc_sim': path['proc_sim']})
            # for ts, d_u in path['Theil'].items():
            #     theil[m][ts] = [ ]
            #     for p in range(len(poly)):
            #         theil[m][ts].append([i[p] for i in list(d_u.values())])
            if theil:
                theil_stat[m] = path['Theil']

        if theil:
            return theil_stat, poly, save_plot+f'valid\\{obj_func}\\theil\\'
        elif trd_agree:
            return trend_agree
        else:
            return poly, prob, obj_func, obs_param, save_plot + f'valid\\{obj_func}'


def combine_calib_res(res, method, obj_func, prob_type='top', proc_sim=False):
    '''
    :param res: [res1, res2]
    :param method: ['fscabc', 'mle']
    :return:
    '''
    if 'chains' in res[method[0]]:
        d_param = {m: {} for m in method}

        for m in method:
            d_param[m] = res[m]['parameters']

        if prob_type == 'top': #TODO: KK
            if obj_func in ['aic']:
                prob = {m: np.negative(np.take(res[m]['prob'], res[m]['buenas'])) if m!= 'dream' else np.take(res[m]['prob'], res[m]['buenas']) for m in method }
            elif obj_func in ['mic', 'nse', 'rmse']:
                prob = {m: np.take(res[m]['prob'], res[m]['buenas']) for m in method}

        elif prob_type == 'all':
            if obj_func in ['aic']: #TODO: KK
                prob = {m: np.negative(res[m]['prob']) if m!= 'dream' else res[m]['prob'] for m in method}
            else:
                prob = {m: res[m]['prob'] for m in method}
        return d_param, prob

    else:
        if proc_sim:
            return {m: {lv: res[m][lv]['likes'] if 'likes' in res[m][lv] else res[m][lv] if lv!='top_sim' else None for lv in res[m]} for m in res}
        else:
            return {m: {lv: res[m][lv]['likes'] for lv in res[m] if lv != 'top_sim'} for m in res}

def plot_heatmap_4_methods(list_method, list_obj_fc):
    x_labels = list_method
    y_label = [i.upper() for i in list_obj_fc]
    data = np.zeros([len(x_labels), len(y_label)])
    for i in range(len(x_labels)):
        data[:, i] = i

    rbg = {c: mcolors.to_rgba(v) for c, v in {'fscabc': '#0000FF', 'dream': '#FFA500', 'mle': '#FF0000', 'demcz': '#008000'}.items()}
    cm = mcolors.LinearSegmentedColormap.from_list('colors', [rbg[y] for y in x_labels])
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cm)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_label)))
    # ... and label them with the respective list entries
    ax.set_xticklabels([i.upper() for i in x_labels])
    ax.set_yticklabels(y_label)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

    for i in range(len(y_label)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, x_labels[j].upper()+'\n'+y_label[i].upper(),
                           ha="center", va="center", color="w")

    fig.tight_layout()
    plt.savefig("C:\\Users\\umroot\\Desktop\\linear\\as", dpi=500)


def clr_marker(mtd_clr=False, mtd_mkr=False, obj_fc_clr=False, obj_fc_mkr=False, wt_mu_m=False):
    if mtd_clr:
        return {'fscabc': 'b', 'dream': 'orange', 'mle': 'r', 'demcz': 'g'}
    elif mtd_mkr:
        return {'fscabc':'o', 'dream':'v', 'mle':'x', 'demcz':'*'}
    elif obj_fc_clr:
        return {'aic': 'b', 'nse': 'orange', 'rmse': 'g'}
    elif obj_fc_mkr:
        return {'aic': 'o', 'nse': 'v', 'rmse': 'x'}
    elif wt_mu_m:
        return  {'weighted_sim' : 'orange', 'mean_sim': 'b', 'median_sim': 'g'}


def plot_prm_prb(obj_func, mtd, res_path, save_plot):
    '''

    :param obj_func: if _rev in obj_func then poly needs to be v_poly
    :param poly:
    :param mtd:
    :return:
    '''

    poly, calib_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'prm_prb', mtd, obj_func, poly_type=19)

    d_param, prob = combine_calib_res(calib_res, mtd, obj_func)

    s_cnl_msk = _soil_canal(poly)

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
                plt.plot(i + 1, mds_dist[m], marker='o', color=f'{clr_marker(mtd_clr=True)[m]}', label=m.upper())
            plt.xticks([i+1 for i in range(len(prob))], [i for i in prob], rotation=70, fontsize=6)
            _label('Calibration approaches', f"Distance of the mean value of calibrating {p[:p.index('-')] if '-' in p else p} to the previous optimum value",
                   f"Differences of mean {p[:p.index('_')] if '_' in p else p} to the optimum to calibration approaches", fontsize=6)
            plot_save(p, '_mds_dist', save_plot)

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
                    a.append(plt.scatter([xe] * len(ye), ye, alpha=0.3, marker=f'{list(clr_marker(mtd_mkr=True))[i]}',
                                         label=y, color=f'{clr_marker(mtd_clr=True)[m]}'))

            plt.xticks(x, xlabels, rotation=70, fontsize=10)
            _label('Soil Class & Canal Position', f"Distance of the mean calibrated{p[:p.index('-')] if '-' in p else p} "
            f"to old optimum value", f" Differences of mean {p[:p.index('-')] if '-' in p else p} to the optimum to physical conditions")
            labl = [m for m in prob]
            ind = np.searchsorted([i for i, m1 in enumerate(prob) for m in [p.get_label() for p in a] if m1 == m],
                                  [i for i in range(len(labl))], side='right')
            plt.legend([a[i - 1] for i in ind], labl)
            plot_save(p, '_bf_dist', save_plot)


def prb_cls(obj_func, mtd, p_m, res_path, save_plot):

    poly, prob, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'valid', mtd, obj_func, p_m,
                                                             poly_type=18)

    s_cnl_msk = _soil_canal(poly)
    objfc = [obj_func[:obj_func.index('_')-1].upper() if '_' in obj_func else obj_func.upper()]

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
                plt.scatter([xe] * len(ye), ye, alpha=0.3, marker=f'{list(clr_marker(mtd_mkr=True))[i]}', label=y,
                            color=f'{clr_marker(mtd_clr=True)[y]}'))

    _plot()
    labl = [m.upper() for m in prob]
    ind = np.searchsorted([i for i, m1 in enumerate(prob) for m in [p.get_label() for p in a] if m1 == m],
                          [i for i in range(len(labl))], side='right')
    plt.legend([a[i - 1] for i in ind], labl)
    plot_save('valid', f'', save_plot)

    for i, m in enumerate(mtd):
        plt.plot(x, [np.mean(np.asarray(v)) for v in m_y[m]], label=m.upper(), color=f'{clr_marker(mtd_clr=True)[m]}',
                 marker=f'{list(clr_marker(mtd_mkr=True))[i]}', linestyle='dashed')
    _plot()
    plot_save('valid', f'_mean', save_plot)

    for i, m in enumerate(mtd):
        plt.plot(x, [np.nanmedian(np.asarray(v)) for v in m_y[m]], label=m.upper(), color=f'{clr_marker(mtd_clr=True)[m]}', marker= f'{clr_marker(mtd_mkr=True)[m]}', linestyle= 'dashed')
    _plot()
    plot_save('valid', f'_median', save_plot)


def param_uncertainty_bounds(sim_res, observations, poly, proc_sim, save_plot, fig, ax, ind_ax=None):
    q5,q25,q75,q95=[],[],[],[]
    for t in range(len(observations)):
        q5.append(np.percentile(sim_res[:, t],2.5))
        q95.append(np.percentile(sim_res[:, t],97.5))
    ax.plot(q5,color='lightblue',linestyle='solid')
    ax.plot(q95,color='lightblue',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='lightblue',zorder=0,
                    linewidth=0,label='5th-95th percentile parameter uncertainty', alpha=0.4)
    for n, array in proc_sim.items():
        ax.plot(array,color=f'{clr_marker(wt_mu_m=True)[n]}',label=f'{n.capitalize()[:n.index("_")]} Simulation', linestyle='dashed')

    ax.plot(observations, 'r-', label=f'Poly{poly} observation')
    ax.set_ylim(-6, 9)
    ax.set_yticks(np.arange(-6, 9, 2))
    yrange = [str(i) for i in np.arange(-4, 9, 2)]
    yrange.insert(0, '')
    if ind_ax is not None and ind_ax==0:
        ax.set_yticklabels(yrange,size=15)
    else:
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_xticks(np.arange(len(observations)))
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_xticklabels([f'{i}-season' for i in np.arange(len(observations))], rotation=45, size=6)
    # ax.legend()


def param_distribution(res_path, save_plot, methods, obj_func):
    '''
    15 parameters distribution based on the diff chain & 9 parameters distribution over validation polygons
    :param res_path:
    :param save_plot:
    :param methods:
    :return:
    '''
    poly, calib_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'prm_prb', methods, obj_func, poly_type=19)
    def _ind_4_obj_func(m_prob):
        if obj_func in ['nse', 'mic']:
            ind = np.argmax(m_prob)
        elif obj_func in ['aic', 'rmse']:
            ind = np.argmin(m_prob)
        return ind

    calib_res, prob = combine_calib_res(calib_res, methods, obj_func, prob_type='all')
    s_cnl_msk = _soil_canal(poly)
    j=0
    width = 10
    height = 5
    fig1, ax1 = plt.subplots(3, figsize=(width, 3 * height))
    for prm in list(calib_res.values())[0]:
        if '-' in prm:
            for cp, lst_p in s_cnl_msk.items():
                plt.ioff()
                fig, ax = plt.subplots(len(lst_p), figsize=(width, len(lst_p) * height), squeeze=False)
                for i in range(len(lst_p)):
                    p = lst_p[i]
                    obs_val = obs_param[prm][p-1]
                    for m in methods:
                        ind = _ind_4_obj_func(prob[m])

                        opt_sim = np.nanmean(np.take(calib_res[m][prm][ind], np.asarray([i-1 for i in poly])))

                        if i == 0:
                            label = f'{m}'
                        else:
                            label = None

                        sns.distplot(calib_res[m][prm][:, p - 1], hist=False, kde=True,
                                     color=clr_marker(mtd_clr=True)[m], ax=ax[i, 0], label=label)
                        l_obs = ax[i, 0].axvline(obs_val, ls="--", color='k')
                        l_opt = ax[i, 0].axvline(opt_sim, ls=":", color=clr_marker(mtd_clr=True)[m])

                    _save_dist(ax[i, 0], fig, type_density={'polygon': p}, save_plot=False)

                # fig.text(0.01, 0.98, '--- Previously calibrated value', color='k', weight='roman', size='small')
                # fig.text(0.01, 0.98 - 0.02, '...optimum calibrated value', color='b', weight='roman', size='small')
                fig.text(0.005, 0.5, f'{cp} Probability density', ha='center', va='center', rotation='vertical')
                fig.text(0.5, 0.005, f'{prm[:prm.index("-")-1]}', ha='center', va='center')
                fig.legend((l_obs, l_opt), ('Previously calibrated value', 'Optimum calibrated value'),
                           'upper left')
                fig.tight_layout()
                fig.savefig(save_plot + f'{cp[0]+cp[cp.index(",")+2]}_{prm[:prm.index("-")-1]}')
                plt.clf()

        else:
            obs_val = obs_param[prm]
            plt.ioff()
            for m in methods:
                ind = _ind_4_obj_func(prob[m])
                opt_sim = calib_res[m][prm][ind]
                if j == 0:
                    label = f'{m}'
                else:
                    label = None
                sns.distplot(calib_res[m][prm], hist=False, kde=True,
                             color=clr_marker(mtd_clr=True)[m], ax=ax1[j], label=label)
                l_obs = ax1[j].axvline(obs_val, ls="--", color='k')
                l_opt = ax1[j].axvline(opt_sim, ls=":", color=clr_marker(mtd_clr=True)[m])

            _save_dist(ax1[j], fig1, type_density={'parameter': prm}, save_plot=False)
            j += 1
            if j == 3:
                # fig1.text(0.01, 0.98, '--- Previously calibrated value', color='k', weight='roman', size='small')
                # fig1.text(0.01, 0.98 - 0.02, '.... Optimum calibrated value', color='b', weight='roman', size='small')
                fig1.text(0.005, 0.5, f'Probability density', ha='center', va='center', rotation='vertical')
                fig1.text(0.5, 0.005, f'SD Parameter Range', ha='center', va='center')
                fig1.legend((l_obs, l_opt), ('Previously calibrated value', 'Optimum calibrated value'),
                           'upper left')

                fig1.tight_layout()
                fig1.savefig(save_plot + f'sdm_{[prm[:3] if "Tinamit" in prm else prm][0]}')
                plt.close(fig1)


def likes_distribution(res_path, methods, save_plot, obj_func):
    '''
    likes distribution of calibration results
    :param res_path:
    :param methods:
    :param save_plot:
    :return:
    '''
    calib_res = path_4_plot(res_path, save_plot, 'prm_prb', methods, obj_func, poly_type=19)[1]

    d_likes = combine_calib_res(calib_res, methods, obj_func, prob_type='all')[1]

    opt_val = {m: np.nanmax(d_likes[m]) if obj_func in ['nse', 'mic'] else np.nanmin(d_likes[m]) for m in methods}

    fig, ax = plt.subplots(figsize=(10, 8))

    ll = [ ]
    for m in d_likes:
        val = d_likes[m]
        sns.distplot(val, hist=False, kde=True, color=clr_marker(mtd_clr=True)[m], ax=ax, label=f'{m}')
        # ax.axvline(np.nanmean(val), ls="--", color='k')
        # ax.axvline(np.nanmedian(val), ls=":", color='b')
        l = ax.axvline(opt_val[m], ls=':',  color=clr_marker(mtd_clr=True)[m])
        ll.append(l)
    fig.legend((ll), ('Optimum calibrated value', '','',''))
    # fig.text(0.8, 0.9, '... Optimum calibrated value', color='k', weight='roman', size='small')

    _save_dist(ax, fig, save_plot=save_plot+'prm_prb\\like_dist\\',
               type_density={'obj_func': obj_func[:obj_func.index('_')] if 'rev' in obj_func else obj_func},
               dpi=500)

    plt.clf()


def compare_vsim_to_vobs(methods, res_path, save_plot, obj_func):

    poly, valid_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'valid', methods, obj_func, poly_type=18, proc_sim=True)

    valid_res = combine_calib_res(valid_res, methods, obj_func, proc_sim=True)

    sim_type = ['weighted_sim']

    for i, p in enumerate(poly):
        plt.ioff()
        fig, axs = plt.subplots(1, 3, figsize=(10*3, 5))
        for ind, t_sim in enumerate(sim_type):
            axs[ind].plot(valid_res[methods[0]]['proc_sim']['obs_norm'][:, i], label='Observation')
            for m, d_data in valid_res.items():
                axs[ind].plot(valid_res[m]['proc_sim'][t_sim][:,i], color=clr_marker(mtd_clr=True)[m],
                                                label=f'{m}')

            _set_ax_marker(axs[ind], 'Time\nSeason', 'Water table depth', f"{t_sim[:t_sim.index('_')].capitalize()+' Simulation'}")

        fig.suptitle(f'Polygon {p}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.legend()
        fig.savefig(save_plot+f'\\vsim_vobs\\'+f'{p}') #, dpi=500)
        plt.close(fig)

def boxplot_like_loc(methods, res_path, save_plot, obj_func, prob_type='top'):

    vpoly, valid_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'valid', methods, obj_func,
                                                                  poly_type = 18, proc_sim = False)

    calib_res = path_4_plot(res_path, save_plot, 'prm_prb', methods, obj_func, poly_type=19)[1]

    calib_prob = combine_calib_res(calib_res, methods, obj_func, prob_type=prob_type)[1]

    opt_val = {m: np.argmax(calib_prob[m]) if  obj_func in ['nse', 'mic'] else np.argmin(calib_prob[m]) for m in methods} #TODO: KK

    s_cnl_msk = _soil_canal(vpoly)

    sl_cl = {m: [valid_res[m]['weighted_res']] for m in methods}

    xlim = [0, 18]
    xpos = np.linspace(xlim[0] + 0.5, xlim[1] + 0.5, len(vpoly), endpoint=False)
    xpol = [i for j in list(s_cnl_msk.values()) for i in j]

    l_xpol = [np.arange(xpos[xpol.index(l_p[0])], xpos[xpol.index(l_p[0])] + len(l_p) * 0.5, 0.5) for j, l_p in
     s_cnl_msk.items()]

    xlabels = [ ]
    for j in [[[i[0] + i[i.index(',') + 2] for i in s_cnl_msk][i]] * len(v) for i, v in enumerate(l_xpol)]:
        xlabels.extend(j)

    # whiskers = [ ]
    for m in methods:
        data = sl_cl[m][0]
        vsim_data = valid_res[m]['weighted_res']

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.canvas.set_window_title(f'{m} Boxplot')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = ax1.boxplot(data, notch=0, sym='.', positions=np.asarray(xpos), whis=[5, 95])
        # whiskers.append(min([i[0] for i in [item.get_ydata() for item in bp['whiskers']]]))

        # Add a horizontal grid to the plot
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_axisbelow(True)
        # _set_ax_marker(ax1, 'Canal position', f'{obj_func.upper()}', #f'{obj_func[:obj_func.index("_")].upper()}_AIC'
        #                f' {m.upper()} comparison of {obj_func.upper()} Across All Validation Canal Positions')

        #fill the boxes with desired colors
        box_colors = ['#F5B7B1','#00BCD4', '#e3931b', '#A5D6A7', '#B39DDB', '#F48FB1', '#C8E851', '#039BE5', '#3949AB']
        rbg = [mcolors.to_rgba(c) for c in box_colors]
        num_boxes = len(data)
        medians = np.empty(num_boxes)

        color = []
        for i, l_pos in enumerate(l_xpol):
            for j in range(len(l_pos)):
                color.append(rbg[i])

        for i, v in enumerate(xpos):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])

            box_coords = np.column_stack([boxX, boxY])
            ax1.add_patch(Polygon(box_coords, facecolor=color[i]))

            # Now draw the median lines back over what we just filled in
            med = bp['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                l_median = ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]

            # add mean
            l_mean = ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*')

        for tick, label in zip(range(num_boxes), ax1.get_xticks()):
            ax1.text(label, .95, vpoly[tick], transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='small', color=color[tick])

        l_opt = ax1.axhline(np.nanmean(vsim_data[opt_val[m]]), ls="dashed", color='r') #the optimal run in calibration

        # Set the axes ranges and axes labels
        if obj_func == 'mic':
            ylim = [0.4, np.nanmax(vsim_data)+0.1]
            ax1.set_ylim(ylim[0], ylim[1])
            ax1.set_yticks(np.arange(0.4, ylim[1], 0.06))
        else:
            ax1.set_ylim(-300, 0)
            ax1.set_yticks(np.arange(-300, 0, 50))

        ax1.set_xlim(xlim[0], xlim[1])
        # ax1.set_xticks([np.average(i) for i in l_xpol])
        ax1.set_xticklabels([j for i in l_xpol for j in i])
        ax1.set_xticklabels(xlabels, rotation=45, fontsize=14)

        # add a basic legend
        # pos = [i for i in np.flip(np.arange(0.01, 0.7, 0.05))][:len([i[0] + i[i.index(',') + 2] for i in s_cnl_msk])] #pos=0.035 is right
        # for i, v in enumerate([i[0] + i[i.index(',') + 2] for i in s_cnl_msk]):
        #     fig.text(0.97, pos[i], f'{v}',
        #              backgroundcolor=rbg[i], color='white', weight='roman', size='small', bbox={'pad': 3.5, 'ec':rbg[i], 'fc':rbg[i]})

        # fig.legend((l_median[0], l_mean[0], l_opt), ('Median', 'Mean', 'Optimum calibrated value'),
        #            bbox_to_anchor=(0.9, 0.1), facecolor='#CACFD2', fontsize='x-small')

        fig.savefig(save_plot + f'\\boxplot\\' + f'{prob_type}_{m}', dpi=500, bbox_inches='tight')
        plt.close(fig)


def plot_theil(methods, res_path, save_plot, obj_func):

    theil, poly, save_plot = path_4_plot(res_path, save_plot, 'valid', methods, obj_func, poly_type=18, proc_sim=False, theil=True)

    s_scl_msk = _soil_canal(poly)

    # tl = {m: {} for m in methods}
    # for m in methods:
    #     for ts, d_u in theil[m].items():
    #         tl[m][ts] = []
    #         for p in range(len(poly)):
    #             tl[m][ts].append([i[p] for i in list(d_u.values())])
    i=0
    for name, l_poly in s_scl_msk.items():
        N = len(l_poly)
        name = name[0]+name[name.index(',')+2]
        for ts in theil[methods[0]]:
            xticks = [f'Poly{p}' for p in l_poly]

            plt.ioff()
            with sns.axes_style("white"):
                sns.set_style("ticks")
                sns.set_context("talk")

                # plot details
                bar_width = 0.15
                epsilon = .005
                line_width = 1
                opacity = 0.7

                def _plt_bar(bar_pos, m, ind):
                    if ind == 0:
                        slabel = f'{m} Us'
                        clabel = f'{m} Uc'
                    else:
                        slabel = None
                        clabel = None

                    poly_ind = [np.where(poly == p)[0][0] for p in l_poly]
                    u_data = {u: np.take(theil[m][ts][u], poly_ind) for u in theil[m][ts]}

                    plt.bar(bar_pos, u_data['Um'], bar_width,
                                              color=clr_marker(mtd_clr=True)[m],
                                              label=f'{m} Um')
                    plt.bar(bar_pos, u_data['Us'], bar_width - epsilon,
                                              bottom=u_data['Um'],
                                              alpha=opacity,
                                              color='white',
                                              edgecolor=clr_marker(mtd_clr=True)[m],
                                              linewidth=line_width,
                                              hatch='//',
                                              label=slabel)
                    plt.bar(bar_pos, u_data['Uc'], bar_width - epsilon,
                                               bottom=u_data['Um'] + u_data['Us'],
                                               alpha=opacity,
                                               color='white',
                                               edgecolor=clr_marker(mtd_clr=True)[m],
                                               linewidth=line_width,
                                               hatch='0',
                                               label=clabel)

                bar_position = {m: np.arange(N)+i*bar_width for i, m in enumerate(methods)}

                for m in methods:
                    _plt_bar(bar_position[m], m, ind=methods.index(m))
                i+=1
                pos_tick = []
                for i in range(N):
                    pos_tick.append(np.average([v[i] for m, v in bar_position.items()]))

                plt.xticks(pos_tick, xticks, fontsize=12)
                plt.ylabel('Errors')
                sns.despine()

                if i == 2:
                    lgd = plt.legend(bbox_to_anchor=(1.1, 1.1), fontsize=10)
                    plt.savefig(save_plot+f'{ts}_{name}', dpi=500, bbox_extra_artists=(lgd), bbox_inches='tight')
                else:
                    plt.savefig(save_plot + f'{ts}_{name}', dpi=500)
                plt.clf()


def plot_gof_convergence(gof, methods, res_path, save_plot):
    def scale(x, out_range=(0, 1)):
        domain = np.min(x), np.max(x)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range)

    gof_coverg = {m:{} for m in methods}
    for m in methods:
        gof_coverg[m] = np.load(res_path+f'{m}\\{m}_{gof}_PrmProb.npy').tolist()['likes']

    vals = [i for m, v in gof_coverg.items() for i in v]
    if gof == 'rmse':
        mini, maxi = -3.0, 0.0
    elif gof == 'nse':
        mini, maxi = -15.0, 1.0
    elif gof == 'mic':
        mini, maxi = np.min(vals), np.max(vals)+0.1
    elif gof=='aic':
        mini, maxi = np.min(vals), np.max(vals)+10

    lines = []
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.set_ylim(mini, maxi)

    for m, v in gof_coverg.items():
        x = np.arange(0, len(v))
        lines += ax.plot(x, v, '-', color=clr_marker(mtd_clr=True)[m])
        # line = ax.plot(x, v, '-', color=clr_marker(mtd_clr=True)[m])
        # ax.legend(line, [m.upper()], loc='upper left', frameon=False)
    ax.set_xlabel('Number of Runs', fontname="Cambria", fontsize=20)
    ax.set_ylabel(f'{gof.upper()}', fontname="Cambria", fontsize=20)
    ax.legend(lines, [m.upper() for m in methods], ncol=4, loc='upper left', frameon=False)
    fig.savefig(save_plot + f'ConvrgeP_{gof}', dpi=500)
    plt.clf()
    plt.close(fig)

    # ax.legend(lines, [m.upper() for m in methods], ncol=4, loc='upper left', frameon=False)

def input_data(obj_func, algorithm):
    tipo_proc = ['patrón' if obj_func in ['aic', 'mic'] else 'multidim' if obj_func in ['nse', 'rmse'] else ''][0]

    d_val =  {'dream': {'aic': (0, 10) , 'mic': (10, 20)},
                 'mle': {'aic': (40, 50), 'mic': (50, 60)},
                 'fscabc': {'aic': (20, 30), 'mic': (30, 40)},
                 'demcz': {'aic': (0, 10), 'mic': (0, 10)}}

    n_sim = d_val[algorithm][obj_func]

    sim_path = f"D:\Gaby\Dt\Calib\simular\dec\\{algorithm}\\{obj_func}\\"

    return tipo_proc, n_sim, sim_path


def cal_gof(tipo_proc, sim, eval, obs=None, name=None, show_gof=True, algrithm=None, plot=False, stat=None):
    if tipo_proc == 'patrón':
        likes = np.zeros([len(eval)])
        poly = list(eval.keys())

        if sim.shape[1] != len(poly):
            sim = sim.T

        for i, (p, best_behav) in enumerate(eval.items()):
            if show_gof:
                if best_behav[:3] == 'spp':
                    if 'spp_oscil_aten' in best_behav:
                        behaviour = best_behav[[x.start() for x in re.finditer("\_", best_behav)][2] + 1:]
                    elif 'spp_oscil' in best_behav:
                        behaviour = best_behav[[x.start() for x in re.finditer("\_", best_behav)][1] + 1:]
                else:
                    behaviour = best_behav

                behaviours = [['linear', behaviour] if stat is not None else [behaviour]][0]
                shape = superposition(np.arange(1, sim.shape[0] + 1), sim[:, poly.index(p)], gof_type=[obj_func],
                                      behaviours=behaviours)[0]
                likes[i] = shape[best_behav]['gof'][obj_func]

                if stat:
                    j = stat['run']
                    data =  stat['data']
                    data['sim_slope'][j, i] = shape['linear']['bp_params']['slope']
                    data['sim_mu'][j, i] = np.average(sim[:, i])

            if plot:
                import matplotlib.pyplot as plt
                # y = predict(np.arange(1, 42, 1), shape[best_behav]['bp_params'], best_behav)
                # plt.plot(y)
                plt.plot(obs[:, i])
                plt.plot(sim[:, i], label='Sim')
                plt.ylim(0, 7)
                plt.title(f"AIC={round(likes[i], 3)}\nobs_beh={best_behav}")
                plt.legend()
                plt.savefig(f"C:\\Users\\umroot\Desktop\map\sim_obs_check\\plots\\{algrithm}\\{name}_{p}")
                plt.close()

def plotss(polys, run, sim_path):
    # sc = _soil_canal(calib_poly)
    # d_c = {c: [] for c in ['H', 'M', 'T']}
    # for n, l_p in sc.items():
    #     d_c[n[n.find(',') + 2]].extend(l_p)

    sim = np.asarray([Dataset.from_dict(cargar_json(os.path.join(sim_path, f'{run}')))[vr].values[:, j - 1] for j in list(polys)]) #19*41
    # mu_obs, sg_obs, norm_obs = aplastar(calib_poly, obs.T)

    eval = np.load(path+'calib_eval.npy').tolist()
    # normeval =np.load(path+'norm_calib_eval.npy').tolist()

    # eval = patro_proces('patrón', valid_poly, obs, obj_func='aic')
    # normeval = patro_proces('patrón', valid_poly, norm_obs, obj_func='aic')

    # normsim = ((sim.T - mu_obs) / sg_obs)

    return sim, eval

def linear_slope(polys, data): #41*19
    len_p = len(polys)
    if data.shape[1] != len_p:
        data = data.T

    d_sign = np.zeros(len_p)
    for p in range(len_p):
        d_sign[p] = simple_shape(np.arange(1, len(data)+1), data[:, p], tipo_egr='linear')['bp_params']['slope']

    return d_sign


algorithms = [ 'demcz', 'dream', 'fscabc', 'mle']

if os.name == 'posix':
    res_path, save_plot = '/Users/gabriellapeng/Downloads/', '/Users/gabriellapeng/Downloads/'
    spotpy_csv = '/Users/gabriellapeng/Downloads/calib/fscabc_aic_rev'

else:
    # path = r'"C:\\Users\\umroot\\OneDrive - Concordia University - Canada\\gaby\pp2_data\\calib\\"'
    # res_path, save_plot = path + "npy_res\\", path + "plot\\"
    # path = "D:\Gaby\Dt\Calib\\"
    # res_path, save_plot = path + "real_run\\", path + "plot\\"
    path = "C:\\Users\\umroot\\Desktop\map\sim_obs_check\\"
    vr = 'mds_Watertable depth Tinamit'

# plot_heatmap_4_methods(mtd, ['mic','aic','rmse', 'nse'])

# trend_agree = { }
# for obj_func in [ 'mic','aic','rmse', 'nse']: #,'rmse', 'nse']: #'mic','aic','rmse', 'nse'
    # plot_gof_convergence(obj_func, mtd, res_path, save_plot)
    # boxplot_like_loc(mtd, res_path, save_plot, obj_func)

    # plot_theil(mtd, res_path, save_plot, obj_func)

    # param_distribution(res_path, save_plot, mtd, obj_func)

    # likes_distribution(res_path, mtd, save_plot, obj_func)

    # compare_vsim_to_vobs(mtd, res_path, save_plot, obj_func)

    # plot_prm_prb(obj_func, mtd, res_path, save_plot)

    # trend_agre = path_4_plot(res_path, save_plot, 'valid', mtd, obj_func, poly_type=19, trd_agree=True)
    # trend_agree.update({obj_func: trend_agre})

plot=False
stat=True
load_data = True

valid_calib = 'calib'


obj_func = ['mic', 'aic']

if valid_calib == 'calib':
    polys = ori_valid[1]
else:
    polys = ori_calib[1]


if not load_data:
    obs = np.asarray([v for i, v in polys.items()]).astype(float) #19*41
    obs_slope = linear_slope(polys, obs)
    obs_mu =np.asarray([np.nanmean(v) for i, v in enumerate(obs)])

else:
    dict_stat = {m:{oj: { } for oj in obj_func}for m in algorithms}

for m in algorithms:
    for gof in obj_func:
        if load_data:
            dict_stat[m][gof] = np.load(path+f'\\data_analysis\\{m[:2]}_{gof}.npy').tolist()

        else:
            tipo_proc, n_sim, sim_path = input_data(gof, m)

            if stat:
                d_stat = {j: np.zeros([n_sim[1]-n_sim[0], len(polys)]) for j in ['sim_mu', 'sim_slope']}
                stat = {'run': i, 'data': d_stat}

            for i, v in enumerate(np.arange(n_sim[0], n_sim[1])):
                print(f'process {i}-th run')
                sim, eval = plotss(polys, v, sim_path)
                cal_gof(tipo_proc, sim, eval,obs = obs.T, name=f'{gof}_{i}', algrithm=m, plot=plot, stat=stat)

            if stat:
                sim_slope = d_stat['sim_slope']
                sim_mu = d_stat['sim_mu']

                d_stat = {'mu(S-O)': np.average(np.asarray([v-obs_mu for i, v in enumerate(sim_mu)]), axis=0)}
                d_stat.update({'Slope(S&O)':np.asarray([len(np.where(np.sign(sim_slope[:, i2])==np.sign(v2))[0])/sim_slope.shape[0] for i2, v2 in enumerate(obs_slope)])})
                np.save(path + f'{m[:2]}_{gof}', d_stat)

print()