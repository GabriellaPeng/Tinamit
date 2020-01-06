import numpy as np
import minepy

import scipy.optimize as optim
import scipy.stats as estad
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

from tinamit.Análisis.Calibs import aplastar, patro_proces, gen_gof
from tinamit.Análisis.Sens.behavior import predict, theil_inequal, compute_rmse
from tinamit.Calib.ej.ej_calib.calib_análisis import plot_save


def validar_resultados(obs, npoly, matrs_simul, tipo_proc=None, obj_func=None, normalize=False):
    """

    patt_vec['bp_params']
    ----------
    obs : pd.DataFrame
    matrs_simul : dict[str, np.ndarray]

    Returns
    -------

    """
    l_vars = list(matrs_simul)
    egr = {vr: {} for vr in l_vars}

    if tipo_proc is None:
        tol=0.65
        mu_obs = obs.mean()
        sg_obs = obs.std()
        mu_obs_matr = mu_obs.to_array().values  # array[3] for ['y'] should be --> array[1]
        sg_obs_matr = sg_obs.to_array().values  # [3]
        sims_norm = {vr: (matrs_simul[vr] - mu_obs_matr[í]) / sg_obs_matr[í] for í, vr in
                     enumerate(l_vars)}  # {'y': 50*21}
        obs_data = {vr: ((obs - mu_obs) / sg_obs)[vr].values for vr in l_vars}  # {'y': 21}
        todas_sims = np.concatenate([x for x in sims_norm.values()], axis=1)  # 50*63 (63=21*3), 100*26*6,
        todas_obs = np.concatenate([x for x in obs_data.values()])  # [63]
        sims_norm_T = {vr: np.array([sims_norm[vr][d, :].T for d in range(len(sims_norm[vr]))]) for vr in l_vars}

        egr = {
            'total': _valid(obs=todas_obs, sim=todas_sims, comport=False),
            'vars': {vr: _valid(obs=obs_data[vr], sim=sims_norm[vr]) for vr in l_vars},
        }

        egr['éxito'] = all(v >= tol for ll, v in egr['total'].items() if 'ens' in ll) and \
                       all(v >= tol for vr in l_vars for ll, v in egr['vars'][vr].items() if 'ens' in ll)

        for vr in l_vars:
            egr[vr] = {'nse': gen_gof('multidim', sim=sims_norm_T[vr], eval=obs_data[vr], obj_func='NSE')}
            egr['éxito_nse'] = all(v >= tol for vr in l_vars for v in egr[vr]['NSE'])

            return egr

    else:
        if normalize:
            mu_obs_matr, sg_obs_matr, obs_data = aplastar(npoly, obs) # obs_norm: dict {'y': 6*21}
            obs_data = {vr: obs_data.T for vr in l_vars}  # [nparray[39*18]]
            for vr in l_vars:
                matrs_simul[vr].update({'all_res': (res - mu_obs_matr) / sg_obs_matr for type, res in matrs_simul[vr].items() if type == 'all_res'})
        else:
            obs_data = {vr: obs for vr in l_vars}

        for vr in l_vars:
            matrs_simul = matrs_simul[vr]
            matrs_simul = generate_weighted_simulations(matrs_simul["all_res"], matrs_simul["prob"], matrs_simul['buenas'])

            egr[vr] = {tipo_proc: {obj_func: { }}}
            egr[vr][tipo_proc][obj_func].update(matrs_simul)

            top_wt_res = matrs_simul['top_res']

            if tipo_proc == 'patrón':
                best_behaviors, obs_linear, obs_shps = patro_proces('patrón', npoly, obs_data[vr], valid=True,
                                                                    obj_func=obj_func)
            for wt_sim in ['weighted_sim', 'top_weighted_sim']:
                weighted_sim = matrs_simul[wt_sim]

                ideal_shape = (matrs_simul['all_res'].shape[1], len(npoly))
                weighted_sim = shape_consist_check(ideal_shape, weighted_sim)
                obs_data[vr] = shape_consist_check(ideal_shape, obs_data[vr])

                if tipo_proc == 'multidim':
                    likes = gen_gof('multidim', sim=weighted_sim, eval=obs_data[vr], valid=True, obj_func=obj_func)
                    top_likes = np.zeros([len(top_wt_res), len(npoly)])

                    obs_linear = patro_proces(tipo_proc, npoly, obs_data[vr], valid=True)[1]
                    sim_linear = patro_proces(tipo_proc, npoly, weighted_sim, valid=True)[1]

                    if not ('likes' in egr[vr][tipo_proc][obj_func]):
                        egr[vr][tipo_proc][obj_func].update({'likes': {}, 'trend_agreement': {}})

                        for i, sim_v in enumerate(top_wt_res):
                            top_likes[i, :] = gen_gof('multidim', sim=sim_v, eval=obs_data[vr], valid=True,
                                                      obj_func=obj_func)
                        egr[vr][tipo_proc][obj_func]['likes'].update({'top_res': top_likes})


                    egr[vr][tipo_proc][obj_func]['likes'].update({wt_sim: likes})
                    egr[vr][tipo_proc][obj_func]['trend_agreement'].update({wt_sim: coeff_agreement(obs_linear,
                                                    sim_linear, None, None, npoly, weighted_sim, obs_data[vr])})

                elif tipo_proc == 'patrón':
                    if not ('Theil' in egr[vr][tipo_proc][obj_func]):
                        egr[vr][tipo_proc][obj_func].update({'Theil': { }, 'likes': { }, 'trend_agreement': {}})

                        top_likes = np.zeros([len(top_wt_res), len(npoly)])
                        for i, sim_v in enumerate(top_wt_res):
                            print(f"Calculating GOF for the {i} run")
                            top_likes[i, :] = gen_gof('patrón', sim=sim_v, eval=best_behaviors, valid=False,
                                                      obj_func=obj_func, valid_like=True)
                        egr[vr][tipo_proc][obj_func]['likes'].update({'top_res': top_likes})

                    egr[vr][tipo_proc][obj_func]['Theil'].update({wt_sim: {i: np.zeros([len(npoly)]) for i in ['Um', 'Us', 'Uc', 'mse']}})

                    for i, p in enumerate(npoly):
                        mse, Um, Us, Uc = theil_inequal(weighted_sim[:, i], obs_data[vr][:, i])
                        egr[vr][tipo_proc][obj_func]['Theil'][wt_sim]['mse'][i] = mse
                        egr[vr][tipo_proc][obj_func]['Theil'][wt_sim]['Um'][i] = Um
                        egr[vr][tipo_proc][obj_func]['Theil'][wt_sim]['Us'][i] = Us
                        egr[vr][tipo_proc][obj_func]['Theil'][wt_sim]['Uc'][i] = Uc

                     #obs_data = 41*18

                    likes, sim_linear, sim_shps = gen_gof('patrón', sim=weighted_sim, eval=best_behaviors, valid=True,
                                                          obj_func=obj_func)  # 19*41

                    egr[vr][tipo_proc][obj_func]['likes'].update({wt_sim: likes})

                    egr[vr][tipo_proc][obj_func]['trend_agreement'].update({wt_sim: coeff_agreement(
                    obs_linear, sim_linear, obs_shps, sim_shps, npoly, weighted_sim, obs_data[vr], best_behaviors)})

            return egr


def _valid(obs, sim, comport=True, tipo_proc=None):  # obs63, sim 50*63//100*21*6
    if tipo_proc is None:
        ic_teor, ic_sim = _anlz_ic(obs, sim)
        egr = {
            'rcem': _rcem(obs, np.nanmean(sim, axis=0)),
            'ens': _ens(obs, np.nanmean(sim, axis=0)),
            'rcem_ic': _rcem(obs=ic_teor, sim=ic_sim),
            'ens_ic': _ens(obs=ic_teor, sim=ic_sim),
        }
    else:
        egr = {
            'rcem': np.array([_rcem(obs, sim[i, :]) for i in range(len(sim))]),
            'ens': np.array([_ens(obs, sim[i, :]) for i in range(len(sim))])
        }
    if comport:
        egr.update({
            'forma': _anlz_forma(obs=obs, sim=sim),
            'tendencia': _anlz_tendencia(obs=obs, sim=sim)
        })

    return egr


def _rcem(obs, sim):
    return np.sqrt(np.nanmean((obs - sim) ** 2))


def _ens(obs, sim):
    return 1 - np.nansum((obs - sim) ** 2) / np.nansum((obs - np.nanmean(obs)) ** 2)


def _anlz_ic(obs, sim):
    n_sims = sim.shape[0]  # 50
    prcnts_act = np.array(
        [np.nanmean(np.less(obs, np.percentile(sim, i / n_sims * 100))) for i in range(n_sims)]
    )  # sim 100*21*6
    prcnts_teor = np.arange(0, 1, 1 / n_sims)
    return prcnts_act, prcnts_teor


formas_potenciales = {
    'lin': {'f': lambda x, a, b: x * a + b},
    'expon': {'f': lambda x, a, b, c: a * c ** x + b},
    'logíst': {'f': lambda x, a, b, c, d: a / (1 + np.exp(-b * (x - c))) - a + d},
    'log': {'f': lambda x, a, b: a * np.log(x) + b},
    # 'gamma': {'f': lambda x, a, b, c, d: a * estad.gamma.pdf(x=x/c, a=b, scale=1) + d}
}


def _anlz_forma(obs, sim):
    dic_forma = _aprox_forma(obs)
    forma_obs = dic_forma['forma']
    ajusto_sims = np.array([_ajustar(s, forma=forma_obs)[0] for s in sim])
    return np.nanmean(ajusto_sims)


def _aprox_forma(vec):
    ajuste = {'forma': None, 'ens': -np.inf, 'paráms': {}}

    for frm in formas_potenciales:
        ajst, prms = _ajustar(vec, frm)
        if ajst > ajuste['ens']:
            ajuste['forma'] = frm
            ajuste['ens'] = ajst
            ajuste['paráms'] = prms

    return ajuste


def _ajustar(vec, forma):
    datos_x = np.arange(len(vec))
    f = formas_potenciales[forma]['f']
    try:
        prms = optim.curve_fit(f=f, xdata=datos_x, ydata=vec)[0]
        ajst = _ens(vec, f(datos_x, *prms))
    except RuntimeError:
        prms = None
        ajst = -np.inf
    return ajst, prms


def _anlz_tendencia(obs, sim):
    x = np.arange(len(obs))
    pend_obs = estad.linregress(x, obs)[0]
    pends_sim = [estad.linregress(x, s)[0] for s in sim]
    pend_obs_pos = pend_obs > 0
    if pend_obs_pos:
        correctos = np.greater(pends_sim, 0)
    else:
        correctos = np.less_equal(pends_sim, 0)

    return correctos.mean()


def coeff_agreement(obs_linear, sim_linear, obs_shps, sim_shps, poly, sim_val, obs_dt, best_behaviors=None):
    # kp<0, the agreement is worsen than random, kp=1 perfect agreement. ICC=1 perfect!
    mine = minepy.MINE()
    def _kp(sim, obs):
        bp_s = []
        bp_o = []
        for i, v in enumerate(sim):
            t_v = (v, obs[i])
            kp = [(1, 1) if all(v >= 0 for v in t_v) else (0, 0) if all(v < 0 for v in t_v) else (0, 1)
            if (t_v[0] < 0) & (t_v[0] >= 0) else (1, 0)][0]

            bp_s.append(kp[0])
            bp_o.append(kp[1])
        return bp_o, bp_s

    l_sign = [
        (1, 1) if all(round(v, 2) > 0 for v in t_v) else (0, 0) if all(round(v, 2) <= 0 for v in t_v) else (
            0, 1) if (round(t_v[0], 2) <= 0) & (round(t_v[1], 2) > 0) else (1, 0) for t_v in
        [(obs_linear[p]['bp_params']['slope'], sim_linear[p]['bp_params']['slope']) for p in poly]]
    no_true_sign = sum([(i[0] == i[1]) for i in l_sign])
    trend_agreement = {'slope': {'data': l_sign, 'no_same_signs': np.round(no_true_sign/len(l_sign), 2)}, 'points_diff': {}}
     #'slp_kp': cohen_kappa_score([i[0] for i in l_sign], [i[1] for i in l_sign]), 'slp_sign': l_sign}

    # trend_agreement['slope'].update({'sign_kendal': estad.kendalltau([i[0] for i in l_sign], [i[1] for i in l_sign])[0]})
    # trend_agreement['slope'].update({'sign_spearmanr':estad.spearmanr([i[0] for i in l_sign], [i[1] for i in l_sign])[0]})
    # obs_bp_slp =[obs_linear[p]['bp_params']['slope'] for p in poly]
    # sim_bp_slp = [sim_linear[p]['bp_params']['slope'] for p in poly ]

    # trend_agreement['slope'].update({'bp_kendal': estad.kendalltau(obs_bp_slp, sim_bp_slp)[0]})
    # trend_agreement['slope'].update({'bp_spearmanr':  estad.spearmanr(obs_bp_slp, sim_bp_slp)[0]})

    # l_sim = [bp for l_bp in [list(sim_linear[p]['bp_params'].values()) for p in poly] for bp in l_bp]
    # l_obs = [bp for l_bp in [list(obs_linear[p]['bp_params'].values()) for p in poly] for bp in l_bp]

    # trend_agreement['slope_intercept'].update({'bp_icc': ICC_rep_anova(np.asarray([l_obs, l_sim]).T)[0]})
    # trend_agreement['slope_intercept'].update({'bp_kendall': estad.kendalltau(l_obs, l_sim)[0]})
    # trend_agreement['slope_intercept'].update({'bp_spearmanr': estad.spearmanr(l_obs, l_sim)[0]})
    # mine.compute_score(l_obs, l_sim)
    # trend_agreement['slope_intercept'].update({'bp_mic':  mine.mic()})

    # kpx, kpy = _kp(l_sim, l_obs)
    # trend_agreement['slope_intercept'].update({'sign_kendall': estad.kendalltau(kpx, kpy)[0]})
    # trend_agreement['slope_intercept'].update({'sign_spearmanr': estad.spearmanr(kpx, kpy)[0]})

    # trend_agreement['slope_intercept'].update({'kappa': cohen_kappa_score(_kp(l_sim, l_obs))})
    # 1why kp<icc ? 2.why slp_kp<0 # as kp is qualitative and quantitative

    # l_s = [list(sim_linear[p]['bp_params'].values()) for p in poly]
    # l_o = [list(obs_linear[p]['bp_params'].values()) for p in poly]
    #
    # trend_agreement['slope_intercept']['poly_icc'] = {}
    # for p in range(len(poly)):
    #     trend_agreement['slope_intercept']['poly_icc'].update({p: ICC_rep_anova(np.asarray([l_o[p], l_s[p]]).T)[0]})
    # if sim_shps is not None:
    #     b_sim = [bp for l_bp in [list(sim_shps[p]['bp_params'].values()) for p in poly] for bp in l_bp]
    #     b_obs = [bp for l_bp in [list(obs_shps[p]['bp_params'].values()) for p in poly] for bp in l_bp]

    #     trend_agreement['bp'] = {'icc': ICC_rep_anova(np.asarray([b_obs, b_sim]).T)[0]}
    #     mine.compute_score(b_obs, b_sim)
    #     trend_agreement['bp'].update({'mic': mine.mic()})
    #
    # for stat in ['icc', 'mic']:
    #     trend_agreement['points_diff'][stat] = np.zeros([len(poly)])
    if best_behaviors is not None:
        x_data = np.arange(1, len(obs_dt)+1)
        trend_agreement['points_diff']['rmse'] = [ ]
        for i, (p, v) in enumerate(best_behaviors.items()):
            y_pred = predict(x_data, obs_shps[p]['bp_params'], v)
            y_obs = predict(x_data, sim_shps[p]['bp_params'], v)
            trend_agreement['points_diff']['rmse'].append(compute_rmse(y_pred, y_obs))
    # for p in range(len(poly)):
    #     wt_sim = np.delete(sim_val[:, p], np.where(np.isnan(obs_dt[:, p])))
    #     obs = obs_dt[:, p][np.where(~np.isnan(obs_dt[:, p]))]
    #     trend_agreement['points_diff']['icc'][p] = ICC_rep_anova(np.asarray([wt_sim, obs]).T)[0]
    #     mine.compute_score(wt_sim, obs)
    #     trend_agreement['points_diff']['mic'][p] = mine.mic()  # data*2
    # trend_agreement['bp'].update({'kappa': _kp(b_sim, b_obs)})
    # trend_agreement['bp'].update({'kendal': estad.kendalltau(b_sim, b_obs)[0]})
    # b_s = [list(sim_shps[p]['bp_params'].values()) for p in poly]
    # b_o = [list(obs_shps[p]['bp_params'].values()) for p in poly]
    #
    # trend_agreement['bp']['poly_icc'] = np.zeros([len(poly)])
    # for p in range(len(poly)):
    #     trend_agreement['bp']['poly_icc'][p] = ICC_rep_anova(np.asarray([b_o[p], b_s[p]]).T)[0]
    #     trend_agreement['bp']['poly_icc_points'][p] = ICC_rep_anova(np.asarray(weight_sim[p], obs_dt[p]).T)[0]

    return trend_agreement


def _ci(data, obs_data, confidence=0.95, probs=None, counts=None):
    if not isinstance(data, np.ndarray):
        a = 1.0 * np.array(data)
    else:
        a = data

    n = len(a)

    if probs is None:
        m, se = np.nanmean(a), estad.sem(a)
    else:
        weight = np.asarray([(p - np.min(probs)) / np.ptp(probs) for p in probs])
        m = np.average(a, weights=weight)
        se = np.sqrt(np.average((a - m) ** 2, weights=weight) / len(weight))

    h = se * estad.t.ppf((1 + confidence) / 2., n - 1)
    intervals = [m - h, m + h]

    if counts is not None:
        if intervals[0]<=obs_data<=intervals[1]:
            counts.append(1)

        if probs is not None:
            return m
    else:
        return intervals


def generate_weighted_simulations(all_simulations, likes, mask):
    len_time =  all_simulations.shape[1]
    no_polys = all_simulations.shape[2]

    top_likes = np.take(likes, mask)
    processed_simulations = {'weighted_sim': np.zeros([len_time, no_polys]),
                             'top_weighted_sim': np.zeros([len_time, no_polys]),
                             'all_res': all_simulations,
                             'top_res': np.zeros([len(mask), len_time, no_polys])
                            }

    all_weights = np.asarray([(p - np.min(likes)) / np.ptp(likes) for p in likes])
    top_weights = np.asarray([(p - np.min(likes)) / np.ptp(likes) for p in top_likes])

    for p in range(no_polys):
        for t in range(len_time):
            a = all_simulations[:, t, p]
            processed_simulations['weighted_sim'][t, p] = np.average(a, weights=all_weights)

            top_res = np.take(a, mask)
            processed_simulations['top_weighted_sim'][t, p] =  np.average(top_res, weights=top_weights) #np.average(a[mask], weights=top_weights)
            processed_simulations['top_res'][:, t, p] = top_res

    return processed_simulations


def shape_consist_check(ideal_shape, input_shape):
    if not (ideal_shape == input_shape.shape):
        input_shape = input_shape.T
    return input_shape


class PatrónValidTest(object):
    def __init__(símismo, obs, obs_norm, tipo_proc, sims_norm, l_vars, save_plot, gard=None):
        símismo.obs = obs
        símismo.tipo_proc = tipo_proc
        símismo.vars_interés = l_vars
        símismo.obs_norm = obs_norm  # {wtd: t*18}
        símismo.sim_norm = sims_norm  # {wtd: N*t*18}
        símismo.save_plot = save_plot
        símismo.gard = gard

    @staticmethod
    def trend_compare(obs, obs_norm, sim_norm, tipo_proc, vars_interés, save_plot):
        poly = obs['x0'].values
        trend = {'t_obs': {}, 't_sim': {}}
        linear = {'t_obs': {}, 't_sim': {}}

        for vr in vars_interés:
            print("\n****Start detecting observed data****\n")
            t_obs = patro_proces(tipo_proc, poly, obs_norm[vr], valid=True)
            trend['t_obs'] = t_obs[0]
            linear['t_obs'] = t_obs[1]

            print("\n****Start detecting simulated data****\n")
            t_sim = patro_proces(tipo_proc, poly, sim_norm[vr], valid='valid_multi_tests')
            trend['t_sim'].update({'best_patt': [list(v.keys())[0] for i, v in t_sim[0].items()]})
            linear['t_sim'].update({p: list(t_sim[2][p]['linear']['bp_params'].values()) for p in t_sim[2]})

            # if the pattens are the same and the values with index are the same
            trend['t_sim']['same_patt'] = {}
            trend['t_sim']['diff_patt'] = {}
            for ind, patt in enumerate(trend['t_obs']['best_patt']):
                if patt == trend['t_sim']['best_patt'][ind]:
                    # if np.count_nonzero(np.isnan(t_obs[poly[ind]]['y_pred']
                    #                              [~np.isnan(t_sim[poly[ind]]['y_pred'])])) == 0:
                    trend['t_sim']['same_patt'][poly[ind]] = t_sim[0][poly[ind]]
                else:
                    trend['t_sim']['diff_patt'][poly[ind]] = {f'{patt}': t_sim[2][poly[ind]][patt]}

            x_data = np.arange(len(obs['n']))

            for p, d_v in trend['t_sim']['diff_patt'].items():
                y_pred = np.asarray(
                    predict(x_data, list(trend['t_sim']['diff_patt'][p].values())[0]['bp_params'], list(d_v.keys())[0]))
                trend['t_sim']['diff_patt'][p] = {list(d_v.keys())[0]: list(d_v.values())[0], 'y_pred': y_pred}

            if save_plot is not None:
                plt.ioff()
                for p in poly:
                    if p in trend['t_sim']['same_patt']:
                        same_occr = trend['t_sim']['same_patt'][p]['y_pred']
                        plt.plot(trend['t_obs'][p]['y_pred'], 'g--', label="t_obs")
                        plt.plot(same_occr, 'r-.', label=f"same")
                        plot_save(p, 'Same_pattern', save_plot)
                    else:
                        diff_occr = trend['t_sim']['diff_patt'][p]['y_pred']
                        plt.plot(trend['t_obs'][p]['y_pred'], 'g--', label="t_obs")
                        plt.plot(diff_occr, 'b.', label=f"diff")
                        plot_save(p, 'Diff_pattern', save_plot)
                    plt.close('all')

            return trend, linear

    def detrend(símismo, poly, trend, obs_norm, sim_norm):
        def _d_trend(d_bparams, patt, dt_norm):
            x_data = np.arange(dt_norm.shape[0])
            return dt_norm - predict(x_data, d_bparams, patt)

        detrend = {'d_obs': {}, 'd_sim': {}}
        for vr, dt in obs_norm.items():
            for p in poly:
                if p in trend['t_sim']:
                    detrend['d_obs'][p] = _d_trend(list(trend['t_obs'][p].values())[0]['bp_params'],
                                                   trend['t_obs']['best_patt_obs'][list(poly).index(p)],
                                                   obs_norm[vr][list(poly).index(p), :])  # 18*41

                    obs = list(list(trend['t_obs'][p].values())[0]['bp_params'].values())
                    sim = list(list(trend['t_sim'][p].values())[0]['bp_params'].values())
                    stat, alpha = estad.ttest_ind(obs, sim)
                    if alpha > 0.05:
                        if not len(detrend['d_sim']) or p not in detrend['d_sim']:
                            detrend['d_sim'][p] = {}
                        detrend['d_sim'][p] = _d_trend(list(trend['t_sim'][p].values())[0]['bp_params'],
                                                       trend['t_obs']['best_patt_obs'][list(poly).index(p)],
                                                       sim_norm[vr][list(poly).index(p), :])  # 2*41*18--18*41--41

            if símismo.save_plot is not None:
                plt.ioff()
                for p in detrend['d_sim']:
                    plt.plot(detrend['d_obs'][p], 'g--', label="d_obs")
                    plt.plot(detrend['d_sim'][p], 'r-.', label=f"d_sim")
                    handles, labels = plt.gca().get_legend_handles_labels()
                    handle_list, label_list = [], []
                    for handle, label in zip(handles, labels):
                        if label not in label_list:
                            handle_list.append(handle)
                            label_list.append(label)
                    plt.legend(handle_list, label_list)
                    plt.savefig(símismo.save_plot + f'd_poly_{p}')
                    plt.close('all')
        if símismo.gard:
            np.save(símismo.gard + '-detrend', detrend)
        return detrend

    def period_compare(símismo, detrend):
        def _compute_var_rk(auto_corr):
            var_rk = []
            N = len(auto_corr) - 1
            for k in range(N):
                summ = []
                for i in range(1, N - 1):
                    if k + i > N:
                        break
                    summ.append((N - i) * (auto_corr[k - i] + auto_corr[k + i] -
                                           2 * auto_corr[k] * auto_corr[i]) ** 2)
                var_rk.append(np.nansum(summ) / (N * (N + 2)))
            return np.asarray(var_rk)

        def _se(var_rs, var_ra):
            se = []
            for k in range(len(var_ra)):
                se.append(np.sqrt(var_rs[k] - var_ra[k]))
            return np.asarray(se)

        def _rk(vec):
            N = len(vec) - 1
            cov = []
            for k in range(N):
                summ = []
                for i in range(1, N - k):
                    if i + k > N:
                        break
                    summ.append((vec[i] - np.nanmean(vec)) * (vec[i + k] - np.nanmean(vec)))
                cov.append((np.nansum(summ) / N) / np.nanvar(vec))
            return np.asarray(cov)

        autocorrelation = {'corr_obs': {}, 'corr_sim': {}}
        for p, v in detrend['d_obs'].items():
            s_obs = _rk(v)
            plt.ioff()
            autocorrelation_plot(s_obs, color='green', linestyle='dashed', label='obs_autocorr')
            autocorrelation['corr_obs'].update({p: s_obs})

            s_sim = _rk(detrend['d_sim'][p])
            plt.plot(s_sim, 'r-.', label=f"autocorr_sim")
            if not autocorrelation['corr_sim'] or p not in autocorrelation['corr_sim']:
                autocorrelation['corr_sim'][p] = {}
            autocorrelation['corr_sim'][p] = s_sim
            plt.title("Autocorrelations of the time patterns (after trend removal)")
            handles, labels = plt.gca().get_legend_handles_labels()
            handle_list, label_list = [], []
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            plt.legend(handle_list, label_list)
            plt.savefig(símismo.save_plot + f'autocorr_{p}')
            plt.close('all')

        autocorr_var = {'var_obs': {}, 'var_sim': {}, 'diff': {}, 'se': {}, 'pass_diff': {}}
        for p, vec_obs in autocorrelation['corr_obs'].items():
            autocorr_var['var_obs'].update({p: _compute_var_rk(vec_obs)})
            var_sim = _compute_var_rk(autocorrelation['corr_sim'][p])
            vec_sim = np.asarray(autocorrelation['corr_sim'][p])
            if not autocorr_var['var_sim'] or p not in autocorr_var['var_sim']:
                autocorr_var['var_sim'][p] = {}
                autocorr_var['diff'][p] = {}
                autocorr_var['se'][p] = {}
                autocorr_var['pass_diff'][p] = {}

            autocorr_var['var_sim'][p] = var_sim
            diff = vec_sim - vec_obs
            se = _se(vec_sim, vec_obs)
            autocorr_var['diff'][p] = diff
            autocorr_var['se'][p] = se
            autocorr_var['pass_diff'][p] = []
            for k in range(len(diff)):
                if np.negative(se)[k] <= diff[k] <= se[k]:
                    autocorr_var['pass_diff'][p].append(diff[k])
                else:
                    autocorr_var['pass_diff'][p].append(np.nan)

        if símismo.save_plot is not None:
            plt.ioff()
            for p in autocorr_var['diff']:
                autocorrelation_plot(autocorr_var['diff'][p], y_label='Differences', label='diff', color='green',
                                     linestyle='dashed')
                plt.plot(autocorr_var['se'][p], 'r-.', label=f"2SE")
                plt.plot(sorted(np.negative(autocorr_var['se'][p])), 'r-.')
                plt.title("Differences of ACFs, 2Se confidence band")
                handles, labels = plt.gca().get_legend_handles_labels()
                handle_list, label_list = [], []
                for handle, label in zip(handles, labels):
                    if label not in label_list:
                        handle_list.append(handle)
                        label_list.append(label)
                plt.legend(handle_list, label_list)
                plt.savefig(símismo.save_plot + f'diff_{p}')
                plt.close('all')

        if símismo.gard:
            np.save(símismo.gard + '-Rk', autocorrelation)
            np.save(símismo.gard + '-autocorr_variables', autocorr_var)
        return autocorr_var

    def mean_compare(símismo, detrend, autocorr_var):
        l_sim = []  # [p1, p2, p3]
        for p in autocorr_var['pass_diff']:
            if np.count_nonzero(~np.isnan(autocorr_var['pass_diff'][p])):
                l_sim.append(p)

        mu = {}
        for p in l_sim:
            mu_obs = np.nanmean(detrend['d_obs'][p])
            mu_sim = np.nanmean(detrend['d_sim'][p])
            n_mu = abs(mu_sim - mu_obs) / mu_obs
            if n_mu <= 0.05:
                mu[p] = n_mu

        if símismo.gard:
            np.save(símismo.gard + '-Mean', mu)

        return mu

    def amplitude_compare(símismo, detrend, mu):
        std = {}
        for p in mu:
            std_obs = np.nanstd(detrend['d_obs'][p])
            std_sim = np.nanstd(detrend['d_sim'][p])
            n_std = abs(std_sim - std_obs) / std_obs
            if n_std <= 0.3:
                std[p] = n_std

        if símismo.gard:
            np.save(símismo.gard + '-Std', std)

        return std

    def phase_lag(símismo, detrend, amplitude):
        # Max-Min>0.8
        def _csa(ai, si):
            csa = []
            mu_s = np.nanmean(si)
            mu_a = np.nanmean(ai)
            std_s = np.nanstd(si)
            std_a = np.nanstd(ai)
            N = len(ai) - 1
            for k in range(N):
                summ = []
                for i in range(k, N):
                    summ.append((si[i] - mu_s) * (ai[i - k] - mu_a))
                csa.append(np.nansum(summ) / (std_s * std_a * N))
            return np.asarray(csa)

        def _cas(ai, si):
            cas = []
            mu_s = np.nanmean(si)
            mu_a = np.nanmean(ai)
            std_s = np.nanstd(si)
            std_a = np.nanstd(ai)
            N = len(ai) - 1
            for lag in range(N):
                neg_k = np.negative(lag)
                summ = []
                for i in range(lag, N):
                    summ.append((ai[i] - mu_a) * (si[i + neg_k] - mu_s))
                cas.append(np.nansum(summ) / (std_s * std_a * N))
            return np.asarray(cas)

        phase = {}
        for p in amplitude:
            len_k = len(detrend['d_obs'][p]) - 1
            k = sorted([np.negative(i) for i in np.arange(len_k)])
            k.extend(np.arange(len_k))
            cas = _cas(detrend['d_sim'][p], detrend['d_obs'][p])
            csa = _csa(detrend['d_sim'][p], detrend['d_obs'][p])
            lag_v = np.concatenate([cas, csa])
            if np.nanmax(lag_v) - np.nanmin(lag_v) >= 0.8:  # >0.5
                if not len(phase) or p not in phase:
                    phase[p] = {}
                phase[p].update({'lag': lag_v, f'|max − min|': np.nanmax(lag_v) - np.nanmin(lag_v)})

                if símismo.save_plot is not None:
                    plt.ioff()
                    autocorrelation_plot(lag_v, y_label='Correlations', label='ccf',
                                         color='blue', linestyle='dashed')
                    plt.title(f"Cross-correlation between obs and sim patterns of poly-{p},sim")
                    plt.legend()
                    plt.savefig(símismo.save_plot + f'ccf_{p}')
                    plt.close('all')

        if símismo.gard:
            np.save(símismo.gard + '-phase', phase)
        return phase

    def discrepancy(símismo, detrend, phase):
        def _sse(vec):
            sse_sum = np.nansum((vec - np.nanmean(vec)) ** 2)
            sse = np.sqrt(sse_sum)
            return sse

        u = {}
        for p in phase:
            si = detrend['d_obs'][p]
            for v in phase[p].items():
                if isinstance(v, np.ndarray):
                    ai = detrend['d_sim'][p]
                    si_ai = si - ai
                    n_u = _sse(si_ai) / (_sse(si) + _sse(ai))
                    if n_u <= 0.7:  # < 0.7
                        if not len(u) or p not in u:
                            u[p] = {}
                        u[p].update(n_u)
        return u

    def conduct_validate(símismo, poly=None, trend=None):
        if poly is None and trend is None:
            poly, trend, linear = símismo.trend_compare(símismo.obs, símismo.obs_norm, símismo.sim_norm,
                                                        símismo.tipo_proc,
                                                        símismo.vars_interés, símismo.save_plot, símismo.gard)
        detrend = símismo.detrend(poly, trend, símismo.obs_norm, símismo.sim_norm)
        autocorr_var = símismo.period_compare(detrend)
        mu = símismo.mean_compare(detrend, autocorr_var)
        amplitude = símismo.amplitude_compare(detrend, mu)
        phase = símismo.phase_lag(detrend, amplitude)
        u = símismo.discrepancy(detrend, phase)
        return u
