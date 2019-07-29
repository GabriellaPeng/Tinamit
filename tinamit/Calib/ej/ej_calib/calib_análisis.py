import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.patches import Polygon
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid
from tinamit.Calib.ej.info_paráms import _soil_canal, calib_líms_paráms, calib_mapa_paráms


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
    plt.close()


def _save_dist(ax, fig, d_data, save_plot, type_density, dpi=500):
    if 'parameter' in type_density:
        v = type_density['parameter']
        if '-' in v:
            v = v[:v.index("-") - 1]

        val = [d_prm[v] for m, d_prm in d_data.items()][0]

    elif 'obj_func' in type_density:
        v = type_density['obj_func']

        val = [d for m, d in d_data.items()][0]

    ax.set_xlabel(f'{v} density')

    ax.set_xlim(np.nanmin(val), np.nanmax(val))

    ax.set_title(f'Shaded density plot of {v}')
    fig.savefig(save_plot, f'{v}.png', dpi=dpi)
    plt.close(fig)


def _set_ax_marker(ax, xlabel, ylabel, title, xlim=None, ylim=None):

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    elif ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_title(title)


def plot_top_sim_obs(sim_norm, obs_norm, npoly, pcentl, save_plot, proc_sim, ci=False):
    for j, p in enumerate(npoly):
        plt.ioff()
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

        proc_sim = {n: val[:, j] for n, val in proc_sim.items()}
        param_uncertainty_bounds(sim_norm[:, :, j], obs_norm[:, j], p, proc_sim, save_plot)


def path_4_plot(res_path, save_plot, plot_type, mtd, obj_func, p_m=None, poly_type='rev'):
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

        if poly_type == 'rev':
            poly = np.asarray(list(v_obs_dt)) #19
        else:
            poly = np.asarray(list(c_obs_dt)) #18

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

        if poly_type == 'rev':
            poly = np.asarray(list(v_obs_dt))
        else:
            poly = np.asarray(list(c_obs_dt))

        return poly, prob, obj_func, obs_param, save_plot + f'valid\\{obj_func}'


def combine_calib_res(res, method, obj_func, prob_type='top'):
    '''
    :param res: [res1, res2]
    :param method: ['abc', 'mle']
    :return:
    '''
    if 'chains' in res[method[0]]:
        for pm in ['POH Kharif Tinamit', 'POH rabi Tinamit', 'Capacity per tubewell']:
            for m in method:
                if 'sampled_prm' in res[m]:
                    res[m]['sampled_prm'][pm] = np.asarray([j.tolist() for j in res[m]['sampled_prm'][pm]])
                res[m][pm] = np.asarray([j.tolist() for j in res[m][pm]])

        for m in method:
            res[m].update({p: np.asarray(res[m][p]) for p, v in res[m].items() if isinstance(v, list)})

        d_param = {m: {} for m in method}

        for m in method:
            d_param[m] = res[m]['parameters']

        if prob_type == 'top':
            if 'aic' in obj_func:
                prob = {m: np.negative(np.take(res[m]['prob'], res[m]['buenas'])) for m in method}
            else:
                prob = {m: np.take(res[m]['prob'], res[m]['buenas']) for m in method}

        elif prob_type == 'all':
            if 'aic' in obj_func:
                prob = {m: np.negative(res[m]['prob']) for m in method}
            else:
                prob = {m: res[m]['prob'] for m in method}
        return d_param, prob

    else:
        valid = {m:{} for m in method}
        for m in res:
            valid[m]['prob_sim']  = res[m]['pro_sim']
            if 'patrón' in valid[m]:
                valid[m]['patrón']  = res[m]['patrón']
        return valid


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

    poly, calib_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'prm_prb', mtd, obj_func, poly_type='rev')

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
                                                             poly_type='ori')

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
    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for t in range(len(observations)):
        q5.append(np.percentile(sim_res[:, t],2.5))
        q95.append(np.percentile(sim_res[:, t],97.5))
    ax.plot(q5,color='dimgrey',linestyle='solid', label='5-th percentile')
    ax.plot(q95,color='dimgrey',linestyle='solid', label='95-th percentile')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='Parameter uncertainty')
    ax.plot(observations,'r.',label='Obs')
    for n, array in proc_sim.items():
        ax.plot(array,color=f'{clr_marker(wt_mu_m=True)[n]}',label=f'{n}', linestyle='dashed')
    ax.legend()
    fig.savefig(save_plot, f'{poly}.png',dpi=500)
    plt.close(fig)


def param_distribution(res_path, save_plot, methods):
    '''
    15 parameters distribution based on the diff chain & 9 parameters distribution over validation polygons
    :param res_path:
    :param save_plot:
    :param methods:
    :return:
    '''

    líms_paráms_final = gen_problema(líms_paráms=calib_líms_paráms, mapa_paráms=calib_mapa_paráms, ficticia=False)[1]

    poly, calib_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'prm_prb', methods, 'aic_rev', poly_type='ori')

    fig = plt.figure(figsize=(16*3, 16))

    for m in methods:
        for par, val in calib_res[m]['sampled_prm'].items():
            ind = list(calib_res[m]['sampled_prm'].keys()).index(par)

            plt.subplot(len(calib_res[m]['sampled_prm']), 2, 2 * ind - 1)

            for i in range(int(max(calib_res[m]['chains']))):
                index = np.where(calib_res[m]['chains'] == i)
                plt.plot(val[index], '.')
            plt.ylabel(par)
            plt.ylim(líms_paráms_final[par][0], líms_paráms_final[par][1])

            plt.subplot(len(líms_paráms_final), 2, 2 * ind + 2)
            normed_value = 1
            hist, bins = np.histogram(val, bins=40, density=True)
            widths = np.diff(bins)
            hist *= normed_value
            plt.bar(bins[:-1], hist, widths)

            plt.axvline(np.nanmean(val), label='Mean', linestyles="dashed")
            plt.axvline(np.nanmedian(val), label='Median', linestyles="dashed")

            plt.ylabel(par)
            plt.xlim(líms_paráms_final[par][0], líms_paráms_final[par][1])

            if ind + 1 == (ind + 1) * 2:
                plt.xlabel('Parameter range')

        fig.savefig(save_plot, f'{m}_param_dist.png', dpi=500)
        plt.close(fig)


    d_param = combine_calib_res(calib_res, methods, obj_func, prob_type='all')[0]

    for prm in list(d_param.values())[0]:
        if len(prm.shape) == 2:
            fig, ax = plt.subplots(len(poly), figsize=(4, len(poly)*2))
            for i, p in enumerate(poly):
                for m in methods:
                    val = calib_res[m]['parameters'][prm][p - 1]
                    sns.distplot(val, hist=False, kde=True,
                                 color=clr_marker(mtd_clr=True)[m], ax=ax[i], label=f'{m}')
                    ax[i].axvline(np.nanmean(val), label='Mean', linestyles="dashed")
                    ax[i].axvline(np.nanmedian(val), label='Median', linestyles="dashed")

                ax.set_ylabel('Probability density')
            _save_dist(ax, fig, d_param, save_plot, type_density={'parameter':prm}, dpi=500)

        else:
            fig, ax = plt.subplots(3, figsize=(4, 3*2))
            i=0
            for m in methods:
                val = calib_res[m]['parameters'][prm]
                sns.distplot(val, hist=False, kde=True,
                             color=clr_marker(mtd_clr=True)[m], ax=ax[i], label=f'{m}')
                ax[i].axvline(np.nanmean(val), label='Mean', linestyles="dashed")
                ax[i].axvline(np.nanmedian(val), label='Median', linestyles="dashed")

            _save_dist(ax, fig, d_param, save_plot, type_density={'parameter':prm}, dpi=500)
            i += 1


def likes_distribution(res_path, methods, save_plot):
    '''
    likes distribution of calibration results
    :param res_path:
    :param methods:
    :param save_plot:
    :return:
    '''
    poly, calib_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'prm_prb', methods, 'aic_rev', poly_type='ori')

    d_likes = combine_calib_res(calib_res, methods, obj_func, prob_type='all')[1]

    fig, ax = plt.subplots(figsize=(4, 2))

    for m in d_likes:
        val = d_likes[m]
        sns.distplot(val, hist=False, kde=True, color=clr_marker(mtd_clr=True)[m], ax=ax, label=f'{m}')
        ax.axvline(np.nanmean(val), label='Mean', linestyles="dashed")
        ax.axvline(np.nanmedian(val), label='Median', linestyles="dashed")

    _save_dist(ax, fig, d_likes, save_plot,
               type_density={'obj_func': obj_func[:obj_func.index('_') - 1] if 'rev' in obj_func else obj_func},
               dpi=500)


def compare_vsim_to_vobs(methods, res_path, save_plot):

    poly, valid_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'valid', methods, 'aic_rev', poly_type='ori')

    valid = combine_calib_res(valid_res, methods, 'aic_rev')

    sim_type = ['weighted_sim', 'mean_sim', 'median_sim']

    for i, p in enumerate(poly):
        plt.ioff()
        fig, axs = plt.subplots(1, 3, figsize=(4*3, 2))
        for t_sim in sim_type:
            axs[sim_type.index(t_sim)].plot(valid[methods[0]]['prob_sim']['obs_norm'][:, i], label='Observation')
            for m, d_data in valid.items():
                axs[sim_type.index(t_sim)].plot(valid[m]['prob_sim'][t_sim][:, i], color=clr_marker(mtd_clr=True)[m],
                                                label=f'{m}')

            _set_ax_marker(axs[sim_type.index(t_sim)], 'Time\nSeason', 'Water table depth', f'{t_sim.capitalize()}',
                           xlim=False, ylim=False)

        fig.suptitle(f'Polygon {p}')
        fig.savefig(save_plot, f'Polygon {p}.png', dpi=500)
        plt.close(fig)


def boxplot_like_loc(methods, res_path, save_plot):

    poly, valid_res, obj_func, obs_param, save_plot = path_4_plot(res_path, save_plot, 'valid', methods, 'aic_rev',
                                                                  poly_type='ori')
    valid = combine_calib_res(valid_res, methods, 'aic_rev')

    calib_res = path_4_plot(res_path, save_plot, 'prm_prb', methods, 'aic_rev', poly_type='ori')[1]
    prob = combine_calib_res(calib_res, methods, 'aic_rev')[1]

    s_cnl_msk = _soil_canal(poly)

    sl_cl = {m: [valid[m]['patrón']['all_sim'][:, p] for i, p in enumerate(poly)] for m in methods}

    xlim = [0, 18]
    a = np.linspace(xlim[0] + 0.5, xlim[1] + 0.5, len(poly), endpoint=False)
    b = [np.arange(i, i + 0.5 * len(l_p), 0.5) for i in a for j, l_p in s_cnl_msk.items()]
    c = [j for i in b for j in i]

    xlabels = [i[0] + i[i.index(',') + 2] for i in s_cnl_msk]

    for m in methods:
        data = sl_cl[m]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.canvas.set_window_title(f'{m} Boxplot')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        ylim = [np.nanmin(valid[m]['patrón']['all_sim']), np.nanmax(valid[m]['patrón']['all_sim'])]

        bp = ax1.boxplot(data, notch=0, sym='.', positions=np.asarray(c), whis=[5, 95])

        # Add a horizontal grid to the plot
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_axisbelow(True)
        _set_ax_marker(ax1, 'Canal position', 'AIC',
                       f' {m.capitalize()} comparison of AIC Across All Validation Canal Positions')

        #fill the boxes with desired colors
        box_colors = ['#F5B7B1','#00BCD4', '#FFE082', '#A5D6A7', '#B39DDB', '#F48FB1', '#F48FB1', '#039BE5', '#3949AB']
        rbg = [mcolors.to_rgba(c) for c in box_colors]
        num_boxes = len(data)
        medians = np.empty(num_boxes)

        color = [rbg[i_clr] for i_clr, j in enumerate(b) for v in c if v in j]

        for i, v in enumerate(c):
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
                ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]

            # add mean
            ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*')


        for tick, label in zip(range(num_boxes), ax1.get_xticks()):
            ax1.text(label, .95, poly[tick], transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='x-small', color=color[tick])

        ax1.axvline(np.nanmax(prob[m]), label='Optimum calibrated value', linestyles="dashed", color='r')
        # Set the axes ranges and axes labels
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])

        ax1.set_xticks([np.average(i) for i in b])
        ax1.set_xticklabels(xlabels, rotation=45, fontsize=8)

        # add a basic legend
        pos = [i for i in np.flip(np.arange(0.01, 0.6, 0.03))][:len(xlabels)]
        for i, v in enumerate(xlabels):
            fig.text(0.9, pos[i], f'{v}',
                     backgroundcolor=rbg[i], color='white', weight='roman', size='x-small')

        fig.text(0.90, 0.015, '*', color='white', backgroundcolor='black', weight='roman', size='large')
        fig.text(0.915, 0.013, ' Average Value', color='black', weight='roman', size='x-small')
        fig.text(0.90, 0.012, '---', color='r', weight='roman', size='large')
        fig.text(0.915, 0.010, 'Calibrated value', color='black', weight='roman', size='x-small')

        fig.savefig(save_plot, f'Boxplot {m}.png', dpi=500)
        plt.close(fig)


aic_mtd = ['fscabc', 'mle', 'dream']
aic_rev_mtd = ['fscabc', 'dream', 'demcz', 'mle']

nse_rev_mtd = ['fscabc', 'dream']

if os.name == 'posix':
    res_path, save_plot = '/Users/gabriellapeng/Downloads/', '/Users/gabriellapeng/Downloads/'
    spotpy_csv = '/Users/gabriellapeng/Downloads/calib/fscabc_aic_rev'

else:
    res_path, save_plot = "D:\Gaby\Tinamit\Dt\Calib\\real_run\\", "D:\Gaby\Tinamit\Dt\Calib\plot\\"

obj_func = 'nse'
p_m = 'multidim'  # multidim, patrón

# plot_prm_prb(obj_func, ['fscabc'], res_path, save_plot)
# prb_cls(obj_func, aic_mtd, p_m, res_path, save_plot) # if obj_fc == aic--> v_poly// rev==c_poly


