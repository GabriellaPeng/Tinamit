import numpy as np
import scipy.optimize as optim
import scipy.stats as estad
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from tinamit.Análisis.Calibs import aplastar, patro_proces, gen_gof
from tinamit.Análisis.Sens.behavior import predict, ICC_rep_anova


def validar_resultados(obs, matrs_simul, tol=0.65, tipo_proc=None, obj_func=None, save_plot=None,
                       gard=None, warmup_period=None):
    """

    patt_vec['bp_params']
    ----------
    obs : pd.DataFrame
    matrs_simul : dict[str, np.ndarray]

    Returns
    -------

    """
    l_vars = list(obs.data_vars)

    if tipo_proc is None:
        mu_obs = obs.mean()
        sg_obs = obs.std()
        mu_obs_matr = mu_obs.to_array().values  # array[3] for ['y'] should be --> array[1]
        sg_obs_matr = sg_obs.to_array().values  # [3]
        sims_norm = {vr: (matrs_simul[vr] - mu_obs_matr[í]) / sg_obs_matr[í] for í, vr in
                     enumerate(l_vars)}  # {'y': 50*21}
        obs_norm = {vr: ((obs - mu_obs) / sg_obs)[vr].values for vr in l_vars}  # {'y': 21}
        todas_sims = np.concatenate([x for x in sims_norm.values()], axis=1)  # 50*63 (63=21*3), 100*26*6,
        todas_obs = np.concatenate([x for x in obs_norm.values()])  # [63]
        sims_norm_T = {vr: np.array([sims_norm[vr][d, :].T for d in range(len(sims_norm[vr]))]) for vr in l_vars}
    else:
        mu_obs_matr, sg_obs_matr, obs_norm = aplastar(obs, l_vars)  # 6, 6, 6*21, obs_norm: dict {'y': 6*21}
        obs_norm = {va: obs_norm for va in l_vars}  # 63, 21*6, [nparray[38, 61]]
        sims_norm = {vr: np.zeros_like(matrs_simul[vr]) for vr in l_vars}
        for vr, matrs in matrs_simul.items():
            for i in range(len(matrs)):
                for t in range(len(obs['tiempo'])):
                    sims_norm[vr][i][t] = (matrs[i, t, :] - mu_obs_matr) / sg_obs_matr  # 1*41*19
                sims_norm_T = {vr: np.array(sims_norm[vr][i, : ].T) for vr in l_vars}  # {'y': 41*19}

    egr = {vr: {} for vr in l_vars}
    if tipo_proc is None:
        egr = {
            'total': _valid(obs=todas_obs, sim=todas_sims, comport=False),
            'vars': {vr: _valid(obs=obs_norm[vr], sim=sims_norm[vr]) for vr in l_vars},
        }

        egr['éxito'] = all(v >= tol for ll, v in egr['total'].items() if 'ens' in ll) and \
                       all(v >= tol for vr in l_vars for ll, v in egr['vars'][vr].items() if 'ens' in ll)

    elif tipo_proc == 'multidim':
        if obj_func.lower() == 'nse':
            for vr in l_vars:
                egr[vr]['NSE'] = gen_gof('multidim', sim=sims_norm_T[vr], obj_func='NSE', eval=obs_norm[vr], valid=True)
                egr[vr]['RMSE'] = gen_gof("multidim", sim=sims_norm_T[vr], obj_func='rmse', eval=obs_norm[vr],
                                          valid=True)
        else:
            egr = {
                'vars': {vr: _valid(obs=obs_norm[vr].T, sim=sims_norm[vr], tipo_proc=tipo_proc, comport=False) for vr in
                         l_vars}}
            egr['éxito'] = all(v >= tol for vr in l_vars for v in egr['vars'][vr]['ens'])

    elif tipo_proc == 'CI':
        def _ci(sim_norm, obs_norm):  # 495*41*19, 41*19
            a = {f'season-{t+warmup_period}': [ ] for t in range(len(obs_norm))}
            for t in range(len(obs_norm)):
                for p in range(obs_norm.shape[1]):
                    if np.isnan(obs_norm[t, p]):
                        a[f'season-{t + warmup_period}'].append(np.nan)
                    else:
                        a[f'season-{t + warmup_period}'].append(estad.percentileofscore(sim_norm[:, t, p], obs_norm[t, p]))
                a[f'season-{t + warmup_period}'] = [estad.percentileofscore(
                    np.asarray([x for x in a[f'season-{t + warmup_period}'] if not np.isnan(x)]), j, kind='weak')
                    for j in np.arange(10, 110, 10)]
                pyplot.ioff()
                pyplot.plot(np.arange(0.1, 1.1, 0.1), 'g-.', label="Theoratical CI")
                pyplot.plot([i*0.01 for i in a[f'season-{t+warmup_period}']], 'r.-', label=f"Season {t} CI")
                pyplot.xticks(np.arange(0, 11), (f'{j}0%CI' for j in range(1, 11)))
                pyplot.yticks(np.arange(0.1, 1.1, 0.1), [float(f"{i}"[:3]) for i in np.arange(0.1, 1.1, 0.1)])
                _plot_poly(t, 'CI_season' + tipo_proc, save_plot)
            return a
        egr[vr] = _ci(sims_norm[vr],obs_norm[vr].T)

    elif tipo_proc == 'patrón':
        trend, linear, eval, len_bparam= PatrónValidTest.trend_compare(obs, obs_norm, sims_norm_T, tipo_proc,
                                                                        l_vars, save_plot=save_plot)
        egr[vr]['AIC'] = gen_gof(tipo_proc, sim=sims_norm_T[vr], obj_func='AIC', eval=eval,
                                 len_bparam=len_bparam, valid='point-pattern') #19*41
        kappa, icc = coeff_agreement(linear, trend, obs['x0'].values)
        egr[vr]['kappa'] = kappa
        egr[vr]['icc'] = icc
        diff_like, diff_wt =gen_gof(tipo_proc, sim=np.asarray(
            [trend['t_sim']['diff_patt'][p]['y_pred'] for p in trend['t_sim']['diff_patt']]), obj_func='AIC',
                      eval=np.asarray([trend['t_obs'][p]['y_pred'] for p in trend['t_sim']['diff_patt']]),
                      len_bparam=len_bparam, valid='pattern-pattern')
        egr[vr]['aic_diff_patt'] = {'like': diff_like, 'wt': diff_wt}
        if gard:
            np.save(gard+'_'+tipo_proc+'_trend', trend)
        return egr
    else:
        egr[vr] = {obj_func.lower(): gen_gof('multidim', sim=sims_norm_T[vr], obj_func=obj_func,
                                             eval=obs_norm[vr])}
        egr['éxito_nse'] = all(v >= tol for vr in l_vars for v in egr[vr]['NSE'])
    return egr

def _plot_poly(p, name, save_plot):
        handles, labels = pyplot.gca().get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        pyplot.legend(handle_list, label_list)
        pyplot.savefig(save_plot + f'{name}_{p}')
        pyplot.close('all')


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
            t_obs = patro_proces(tipo_proc, obs, obs_norm[vr], valid='valid_multi_tests')
            trend['t_obs'] = t_obs[0]
            trend['t_obs'].update({'best_patt': [list(v.keys())[0] for i, v in t_obs[0].items()]})
            linear['t_obs'].update({p: list(t_obs[2][p]['linear']['bp_params'].values()) for p in t_obs[2]})

            print("\n****Start detecting simulated data****\n")
            t_sim = patro_proces(tipo_proc, obs, sim_norm[vr], valid='valid_multi_tests')
            trend['t_sim'].update({'best_patt': [list(v.keys())[0] for i, v in t_sim[0].items()]})
            linear['t_sim'].update({p: list(t_sim[2][p]['linear']['bp_params'].values()) for p in t_sim[2]})

            matriz_vacía = np.asarray(
                [t_obs[0][p]['y_pred'] for p, d_data in t_obs[0].items() if not isinstance(p, str)])

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
                pyplot.ioff()
                for p in poly:
                    if p in trend['t_sim']['same_patt']:
                        same_occr = trend['t_sim']['same_patt'][p]['y_pred']
                        pyplot.plot(trend['t_obs'][p]['y_pred'], 'g--', label="t_obs")
                        pyplot.plot(same_occr, 'r-.', label=f"same")
                        _plot_poly(p, 'Same_pattern', save_plot)
                    else:
                        diff_occr = trend['t_sim']['diff_patt'][p]['y_pred']
                        pyplot.plot(trend['t_obs'][p]['y_pred'], 'g--', label="t_obs")
                        pyplot.plot(diff_occr, 'b.', label=f"diff")
                        _plot_poly(p,'Diff_pattern', save_plot)
                    pyplot.close('all')

            return trend, linear, matriz_vacía, t_obs[1]

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
                pyplot.ioff()
                for p in detrend['d_sim']:
                    pyplot.plot(detrend['d_obs'][p], 'g--', label="d_obs")
                    pyplot.plot(detrend['d_sim'][p], 'r-.', label=f"d_sim")
                    handles, labels = pyplot.gca().get_legend_handles_labels()
                    handle_list, label_list = [], []
                    for handle, label in zip(handles, labels):
                        if label not in label_list:
                            handle_list.append(handle)
                            label_list.append(label)
                    pyplot.legend(handle_list, label_list)
                    pyplot.savefig(símismo.save_plot + f'd_poly_{p}')
                    pyplot.close('all')
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
            pyplot.ioff()
            autocorrelation_plot(s_obs, color='green', linestyle='dashed', label='obs_autocorr')
            autocorrelation['corr_obs'].update({p: s_obs})

            s_sim = _rk(detrend['d_sim'][p])
            pyplot.plot(s_sim, 'r-.', label=f"autocorr_sim")
            if not autocorrelation['corr_sim'] or p not in autocorrelation['corr_sim']:
                autocorrelation['corr_sim'][p] = {}
            autocorrelation['corr_sim'][p] = s_sim
            pyplot.title("Autocorrelations of the time patterns (after trend removal)")
            handles, labels = pyplot.gca().get_legend_handles_labels()
            handle_list, label_list = [], []
            for handle, label in zip(handles, labels):
                if label not in label_list:
                    handle_list.append(handle)
                    label_list.append(label)
            pyplot.legend(handle_list, label_list)
            pyplot.savefig(símismo.save_plot + f'autocorr_{p}')
            pyplot.close('all')

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
            pyplot.ioff()
            for p in autocorr_var['diff']:
                autocorrelation_plot(autocorr_var['diff'][p], y_label='Differences', label='diff', color='green',
                                     linestyle='dashed')
                pyplot.plot(autocorr_var['se'][p], 'r-.', label=f"2SE")
                pyplot.plot(sorted(np.negative(autocorr_var['se'][p])), 'r-.')
                pyplot.title("Differences of ACFs, 2Se confidence band")
                handles, labels = pyplot.gca().get_legend_handles_labels()
                handle_list, label_list = [], []
                for handle, label in zip(handles, labels):
                    if label not in label_list:
                        handle_list.append(handle)
                        label_list.append(label)
                pyplot.legend(handle_list, label_list)
                pyplot.savefig(símismo.save_plot + f'diff_{p}')
                pyplot.close('all')

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
                    pyplot.ioff()
                    autocorrelation_plot(lag_v, y_label='Correlations', label='ccf',
                                         color='blue', linestyle='dashed')
                    pyplot.title(f"Cross-correlation between obs and sim patterns of poly-{p},sim")
                    pyplot.legend()
                    pyplot.savefig(símismo.save_plot + f'ccf_{p}')
                    pyplot.close('all')

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


def coeff_agreement(linear, trend, poly):
    kappa = {}
    icc = {'same': {}, 'diff': {}}
    for p in poly:
        obs_l = np.asarray(linear['t_obs'][p])
        sim_l = np.asarray(linear['t_sim'][p])

        tf = ((obs_l == sim_l) & (obs_l == 0)) | (obs_l * sim_l > 0)
        kappa.update({p: {'sign_bparam': 1, 'sign_slope': 1} if all(i for i in tf) else {'sign_bparam': 0.5,
                                                                                         'sign_slope': 1} if tf[
            0] else None})

        if p in trend['t_sim']['same_patt']:
            obs_bp = np.asarray(list(list(trend['t_obs'][p].values())[0]['bp_params'].values()))
            sim_bp = np.asarray(list(list(trend['t_sim']['same_patt'][p].values())[0]['bp_params'].values()))
            icc['same'][p] = ICC_rep_anova(np.asarray([obs_bp, sim_bp]).T)[0]
        else:
            obs_bp = np.asarray(list(list(trend['t_obs'][p].values())[0]['bp_params'].values()))
            sim_bp = np.asarray(list(list(trend['t_sim']['diff_patt'][p].values())[0]['bp_params'].values()))
            icc['diff'][p] = ICC_rep_anova(np.asarray([obs_bp, sim_bp]).T)[0]

    return kappa, icc