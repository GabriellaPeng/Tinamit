import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.patches import Polygon
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid
from tinamit.Calib.ej.info_paráms import _soil_canal


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


def _label(xlabel, ylabel, title=None, fontsize=None):
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    if title is not None:
        plt.title(title, fontsize=fontsize)


def plot_ci(x_data, y_data, var, save_plot):
    plt.ioff()
    plt.figure(figsize=(11, 4))
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g-.', label="CI over Percentiles")
    plt.plot(x_data, y_data, 'r.-', label=f"{var} CI")
    plt.xticks(x_data, [f'{round(t, 2)}' for t in x_data], fontsize=5.5)
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{t}%" for t in np.arange(0, 101, 10)])
    plot_save(var, 'ci', save_plot)

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


def plot_top_sim_obs(sim_norm, obs_norm, npoly, save_plot, proc_sim, pcentl=False, ci=False):
    for j, p in enumerate(npoly):
        plt.ioff()

        p_sim = {n: val[:, j] for n, val in proc_sim.items()}
        param_uncertainty_bounds(sim_norm[:, :, j], obs_norm[:, j], p, p_sim, save_plot)

        if pcentl:
            plt.figure(figsize=(15, 5))
            plt.plot(obs_norm[:, j], 'r-', label=f'obs_{p}', linewidth=4) #obs:t, p
            for i in range(len(sim_norm)):
                plt.plot(sim_norm[i, :, j], 'b--', linewidth=0.2)
            percentile = np.divide(np.arange(1, obs_norm.shape[0] + 1), obs_norm.shape[0])
            plt.xticks(range(len(percentile)), [f"{i}\n{round(t * 100, 1)}%" for i, t in enumerate(pcentl[:, j])],
                       fontsize=5)
            for xc in range(len(percentile)):
                plt.axvline(x=xc, linewidth=0.2, linestyle=':')
            _label("Season\nConfidential Interval", "Water Table Depth", 8)
            plot_save(p, 't_ci', save_plot)

            if ci:
                plot_ci(percentile, np.sort(pcentl[:, j]), f'Poly-{p}', save_plot)


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
            calib_res[m] = np.load(res_path + f'{m}\\{m}_{obj_func}.npy').tolist()

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
            if 'aic' in obj_func:
                if trd_agree:
                    trend_agree[m] = np.load(res_path + f'{m}\\valid_{obj_func}_patrón_wt.npy').tolist()[vr]['patrón']['aic']['weighted_sim']['trend_agreement']
                else:
                    path = np.load(res_path + f'{m}\\valid_{obj_func}_patrón.npy').tolist()[vr]
                    prob[m] = path['patrón']['aic']

                    if proc_sim:
                        prob[m].update({'proc_sim': path['proc_sim']})
            else:
                if trd_agree:
                    trend_agree[m] = np.load(res_path + f"{m}\\valid_{obj_func}_multidim_wt.npy").tolist()[vr]['multidim'][obj_func[:obj_func.index('_')]]['weighted_sim']['trend_agreement']

                else:
                    path = np.load(res_path + f'{m}\\valid_{obj_func}_multidim.npy').tolist()[vr]
                    prob[m] = path['multidim'][obj_func[:obj_func.index('_')] if '_' in obj_func else obj_func]

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
    :param method: ['abc', 'mle']
    :return:
    '''
    if 'chains' in res[method[0]]:
        d_param = {m: {} for m in method}

        for m in method:
            d_param[m] = res[m]['parameters']

        if prob_type == 'top':
            if 'aic' in obj_func:
                prob = {m: np.negative(np.take(res[m]['prob'], res[m]['buenas'])) if m!= 'dream' else np.take(res[m]['prob'], res[m]['buenas']) for m in method }
            else:
                prob = {m: np.take(res[m]['prob'], res[m]['buenas']) for m in method}

        elif prob_type == 'all':
            if 'aic' in obj_func:
                prob = {m: np.negative(res[m]['prob']) if m!= 'dream' else res[m]['prob'] for m in method}
            else:
                prob = {m: res[m]['prob'] for m in method}
        return d_param, prob

    else:
        if proc_sim:
            return {m: {lv: res[m][lv]['likes'] if 'likes' in res[m][lv] else res[m][lv] if lv!='top_sim' else None for lv in res[m]} for m in res}
        else:
            return {m: {lv: res[m][lv]['likes'] for lv in res[m] if lv != 'top_sim'} for m in res}


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


def param_uncertainty_bounds(sim_res, observations, poly, proc_sim, save_plot):
    plt.ioff()
    fig, ax= plt.subplots(figsize=(16,9))
    q5,q25,q75,q95=[],[],[],[]
    for t in range(len(observations)):
        q5.append(np.percentile(sim_res[:, t],2.5))
        q95.append(np.percentile(sim_res[:, t],97.5))
    ax.plot(q5,color='dimgrey',linestyle='solid')
    ax.plot(q95,color='dimgrey',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='5th-95th percentile parameter uncertainty')
    ax.plot(observations,'r-',label='Obs')
    for n, array in proc_sim.items():
        ax.plot(array,color=f'{clr_marker(wt_mu_m=True)[n]}',label=f'{n}', linestyle='dashed')
    ax.set_xticks(np.arange(len(observations)))
    ax.set_xticklabels([f'{i}-season' for i in np.arange(len(observations))], rotation=45, size=6)
    ax.legend()
    fig.savefig(save_plot+f'{poly}')
    plt.clf()


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
        if 'nse' in obj_func:
            ind = np.argmax(m_prob)
        elif 'rmse' in obj_func or 'aic' in obj_func:
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
                fig.savefig(save_plot + '\\param_dist\\' + f'{cp[0]+cp[cp.index(",")+2]}_{prm[:prm.index("-")-1]}')
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
                fig1.savefig(save_plot + '\\param_dist\\' + f'sdm_{[prm[:3] if "Tinamit" in prm else prm][0]}')
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

    opt_val = {m: np.nanmax(d_likes[m]) if 'nse' in obj_func else np.nanmin(d_likes[m]) for m in methods}

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

    sim_type = ['weighted_sim', 'mean_sim', 'median_sim']

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
        fig.savefig(save_plot+'\\vsim_vobs\\'+f'{p}') #, dpi=500)
        plt.close(fig)


def boxplot_like_loc(methods, res_path, save_plot, obj_func, prob_type='top'):

    vpoly, valid_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'valid', methods, obj_func,
                                                                  poly_type = 18, proc_sim = False)

    calib_res = path_4_plot(res_path, save_plot, 'prm_prb', methods, obj_func, poly_type=19)[1]

    calib_prob = combine_calib_res(calib_res, methods, obj_func, prob_type=prob_type)[1]

    opt_val = {m: np.argmax(calib_prob[m]) if 'nse' in obj_func else np.argmin(calib_prob[m]) for m in methods}

    s_cnl_msk = _soil_canal(vpoly)

    sl_cl = {m: [valid_res[m]['top_sim'][:, i] for i, p in enumerate(vpoly)] for m in methods}

    xlim = [0, 18]
    xpos = np.linspace(xlim[0] + 0.5, xlim[1] + 0.5, len(vpoly), endpoint=False)
    xpol = [i for j in list(s_cnl_msk.values()) for i in j]

    l_xpol = [np.arange(xpos[xpol.index(l_p[0])], xpos[xpol.index(l_p[0])] + len(l_p) * 0.5, 0.5) for j, l_p in
     s_cnl_msk.items()]

    xlabels = [i[0] + i[i.index(',') + 2] for i in s_cnl_msk]

    for m in methods:
        data = sl_cl[m]
        vsim_data = valid_res[m]['top_sim'] #TODO change it to the all sim!

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.canvas.set_window_title(f'{m} Boxplot')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = ax1.boxplot(data, notch=0, sym='.', positions=np.asarray(xpos), whis=[5, 95])

        # Add a horizontal grid to the plot
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_axisbelow(True)
        _set_ax_marker(ax1, 'Canal position', f'AIC', #f'{obj_func[:obj_func.index("_")].upper()}_AIC'
                       f' {m.upper()} comparison of AIC Across All Validation Canal Positions')

        #fill the boxes with desired colors
        box_colors = ['#F5B7B1','#00BCD4', '#FFE082', '#A5D6A7', '#B39DDB', '#F48FB1', '#F48FB1', '#039BE5', '#3949AB']
        rbg = [mcolors.to_rgba(c) for c in box_colors]
        num_boxes = len(data)
        medians = np.empty(num_boxes)

        color = []
        for i, l_pos in enumerate(l_xpol):
            for j in range(len(l_pos)):
                color.append(rbg[i])

        ylim = [np.nanmin(vsim_data), np.nanmax(vsim_data)]

        for i, v in enumerate(xpos):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])

            box_coords = np.column_stack([boxX, boxY])
            matplotlib.use('Agg')
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
                     horizontalalignment='center', size='x-small', color=color[tick])
        if 'aic' in obj_func:
            l_opt = ax1.axhline(np.nanmin(calib_prob[m]), ls="dashed", color='r')
        else:
            l_opt = ax1.axhline(np.nanmean(vsim_data[opt_val[m]]), ls="dashed", color='r')

        # Set the axes ranges and axes labels
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])

        ax1.set_xticks([np.average(i) for i in l_xpol])
        ax1.set_xticklabels(xlabels, rotation=45, fontsize=8)

        # add a basic legend
        pos = [i for i in np.flip(np.arange(0.01, 0.6, 0.03))][:len(xlabels)]
        for i, v in enumerate(xlabels):
            fig.text(0.97, pos[i], f'{v}',
                     backgroundcolor=rbg[i], color='white', weight='roman', size='x-small')
        # fig.text(0.925, 0.13, '*', color='white', backgroundcolor='black', weight='roman', size='large')
        # fig.text(0.94, 0.128, ' Average Value', color='black', weight='roman', size=5)
        # fig.text(0.92, 0.09, '---', color='r', weight='roman', size='large')
        # fig.text(0.94, 0.092, 'Calibrated value', color='black', weight='roman', size=5
        fig.legend((l_median[0],l_mean[0], l_opt), ('Median', 'Mean', 'Optimum calibrated value'), 'upper right', facecolor = '#CACFD2')

        fig.savefig(save_plot+f'\\boxplot\\'+f'{prob_type}_{m}', dpi=500)
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
                pos_tick = []
                for i in range(N):
                    pos_tick.append(np.average([v[i] for m, v in bar_position.items()]))

                plt.xticks(pos_tick, xticks, rotation=45, fontsize=6)
                plt.legend(bbox_to_anchor=(0.98, 1.1), fontsize=5)
                plt.ylabel('Errors')
                sns.despine()
                plt.savefig(save_plot+f'{ts}_{name}', dpi=500)
                plt.clf()


mtd = ['fscabc', 'dream', 'demcz', 'mle']

# if os.name == 'posix':
#     res_path, save_plot = '/Users/gabriellapeng/Downloads/', '/Users/gabriellapeng/Downloads/'
#     spotpy_csv = '/Users/gabriellapeng/Downloads/calib/fscabc_aic_rev'
#
# else:
    # path = r'"C:\\Users\\umroot\\OneDrive - Concordia University - Canada\\gaby\pp2_data\\calib\\"'
    # res_path, save_plot = path + "npy_res\\", path + "plot\\"

trend_agree = { }
# for obj_func in ['aic_rev', 'rmse_rev', 'nse_rev']: #'aic_rev', 'rmse_rev', 'nse_rev'
#     # boxplot_like_loc(mtd, res_path, save_plot, obj_func)
#     #
#     # plot_theil(mtd, res_path, save_plot, obj_func)
#
#     # param_distribution(res_path, save_plot, mtd, obj_func)
#
#     # likes_distribution(res_path, mtd, save_plot, obj_func)
#
#     # compare_vsim_to_vobs(mtd, res_path, save_plot, obj_func)
#     trend_agre = path_4_plot(res_path, save_plot, 'valid', mtd, obj_func, poly_type=19, trd_agree=True)
#     trend_agree.update({obj_func: trend_agre})
#
# print()