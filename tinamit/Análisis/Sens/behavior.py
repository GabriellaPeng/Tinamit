import minepy

import numpy as np
from scipy import optimize
from numpy import ones, kron, mean, eye, hstack, dot, tile
from numpy.linalg import pinv

def predict(x_data, parameters, pattern):
    if pattern == 'linear':
        return parameters['slope'] * x_data + parameters['intercept']
    elif pattern == 'exponencial':
        return parameters['y_intercept'] * (parameters['g_d'] ** x_data) + parameters['constant']
    elif pattern == 'logístico':
        return parameters['maxi_val'] / (1 + np.exp(-parameters['g_d'] * x_data + parameters['mid_point'])) + \
               parameters['constant'] #overflow encountered in exp
    elif pattern == 'inverso':
        return parameters['g_d'] / (x_data + parameters['phi']) + parameters['constant']
    elif pattern == 'log':
        return parameters['g_d'] * np.log(x_data + parameters['phi']) + parameters['constant'] #invalid value encountered in log
    elif pattern == 'oscilación':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant']
    elif pattern == 'oscilación_aten':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant']
    elif pattern == 'spp_oscil_linear':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + parameters['slope_1'] * x_data + parameters['intercept_1']
    elif pattern == 'spp_oscil_aten_linear':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + parameters[
                   'slope_1'] * x_data + parameters['intercept_1']
    elif pattern == 'spp_oscil_exponencial':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + parameters['y_intercept_1'] * (parameters['g_d_1'] ** x_data)
    elif pattern == 'spp_oscil_aten_exponencial':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + parameters[
                   'y_intercept_1'] * (parameters['g_d_1'] ** x_data)
    elif pattern == 'spp_oscil_logístico':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + parameters['maxi_val_1'] / (
                       1 + np.exp(-parameters['g_d_1'] * x_data + parameters['mid_point_1']))
    elif pattern == 'spp_oscil_aten_logístico':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + parameters[
                   'maxi_val_1'] / (1 + np.exp(-parameters['g_d_1'] * x_data + parameters['mid_point_1']))
    elif pattern == 'spp_oscil_inverso':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + parameters['g_d_1'] / (x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_aten_inverso':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + parameters[
                   'g_d_1'] / (x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_log':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + parameters['g_d_1'] * np.log(x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_aten_log':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + parameters[
                   'g_d_1'] * np.log(x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_oscilación':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + parameters['amplitude_1'] * np.sin(parameters['period_1'] * x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_aten_oscilación':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + parameters[
                   'amplitude_1'] * np.sin(parameters['period_1'] * x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_oscilación_aten':
        return parameters['amplitude'] * np.sin(parameters['period'] * x_data + parameters['phi']) + parameters[
            'constant'] + np.exp(parameters['g_d_1'] * x_data) * parameters['amplitude_1'] * \
               np.sin(parameters['period_1'] * x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_aten_oscilación_aten':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + np.exp(
            parameters['g_d_1'] * x_data) * parameters['amplitude_1'] * \
               np.sin(parameters['period_1'] * x_data + parameters['phi_1'])
    elif pattern == 'spp_oscil_aten_inverso':
        return np.exp(parameters['g_d'] * x_data) * parameters['amplitude'] * \
               np.sin(parameters['period'] * x_data + parameters['phi']) + parameters['constant'] + \
               parameters['g_d_1'] * np.log(x_data + parameters['phi_1'])


def linear(x, x_data):  # x is an np.array
    return x[0] * x_data + x[1]


def exponencial(x, x_data):
    return x[0] * (x[1] ** x_data) + x[2]


def logístico(x, x_data):
    return (x[0] / (1 + np.exp(-x[1] * x_data + x[2]))) + x[3] #overflow encountered in exp


def inverso(x, x_data):
    return (x[0] / (x_data + x[1])) + x[2]


def log(x, x_data):
    return x[0] * np.log(x_data + x[1]) + x[2]


def oscilación(x, x_data):
    return x[0] * np.sin(x[1] * x_data + x[2]) + x[3]


def oscilación_aten(x, x_data):
    return np.exp(x[0] * x_data) * x[1] * np.sin(x[2] * x_data + x[3]) + x[4]


def simple_shape(x_data=None, y_data=None, tipo_egr='linear', gof=False, gof_type=['aic'], type_bnds=None):
    def f_opt(x, x_data, y_data, f):
        return compute_rmse(f(x, x_data), y_data)

    norm_y_data = (y_data - np.nanmean(y_data)) / np.nanstd(y_data)
    # norm_y_data = y_data

    if tipo_egr == 'linear':
        params = optimize.minimize(f_opt, x0=[1, 1], method='Nelder-Mead', args=(x_data, norm_y_data, linear)).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        # slope, intercept, r_value, p_value, std_err = estad.linregress(x_data, norm_y_data)
        # b_params = {'bp_params': de_standardize(np.asarray([slope, intercept]), y_data, tipo_egr)}
        if gof:
            k = len(b_params['bp_params'])
            y_pred = linear(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    elif tipo_egr == 'exponencial':
        params = optimize.minimize(f_opt, x0=[0.1, 1.1, 0], method='Powell',
                                   args=(x_data, norm_y_data, exponencial)).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        if gof:
            k=len(b_params['bp_params'])
            y_pred = exponencial(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    elif tipo_egr == 'logístico':
        bnds=((-5, 10),(-0.1, 1),(None, 10),(None, 10))
        params = optimize.minimize(f_opt, x0=[5.0, 0.85, 3.0, 0], method= 'SLSQP', #'Powell',
                                   args=(x_data, norm_y_data, logístico), bounds=bnds).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        if gof:
            k = len(b_params['bp_params'])
            y_pred = logístico(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    elif tipo_egr == 'inverso':
        params = optimize.minimize(f_opt, x0=[3.0, 0.4, 0], method='Nelder-Mead', args=(x_data, norm_y_data, inverso)).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        if gof:
            k = len(b_params['bp_params'])
            y_pred =inverso(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    elif tipo_egr == 'log':
        if type_bnds == -1:
            bnds = ((None, 0), (0.001, None), (None, None))
        else:
            bnds = ((0, None), (0.001, None), (None, None))

        params = optimize.minimize(f_opt, x0=[0.3, 0.1, 0], method='SLSQP', args=(x_data, norm_y_data, log), bounds=bnds).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        if gof:
            k = len(b_params['bp_params'])
            y_pred =log(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    elif tipo_egr == 'oscilación':
        params = optimize.minimize(f_opt, x0=[2, 1.35, 0, 0], method='Powell',  # for wtd 7, 1.6, 0, 0
                                   args=(x_data, norm_y_data, oscilación)).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        if gof:
            k = len(b_params['bp_params'])
            y_pred =oscilación(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    elif tipo_egr == 'oscilación_aten':
        # x0 assignment is very tricky, the period (3rd arg should always>= real period)
        bnds=((-0.1, 1),(None, None),(None, None),(None, None), (None, None))
        params = optimize.minimize(f_opt, x0=[0.1, 2, 2, 0, 0], method='SLSQP',  # 0.1, 1, 2, 0.01, 0, SLSQP, Powell
                                   args=(x_data, norm_y_data, oscilación_aten), bounds=bnds).x
        b_params = {'bp_params': de_standardize(params, y_data, tipo_egr)}
        if gof:
            k = len(b_params['bp_params'])
            y_pred = oscilación_aten(np.asarray(list(b_params['bp_params'].values())), x_data)
            b_params.update({'gof': {i: fc[i](k,y_pred, y_data) for i in gof_type}})

    else:
        raise ValueError(tipo_egr)

    return b_params


def forma(x_data, y_data, gof_type=['aic'], behaviours=None):
    if behaviours is None:
        behaviors_aics = {'linear': {},
                          'exponencial': {},
                          'logístico': {},
                          'inverso': {},
                          'log': {},
                          'oscilación': {},
                          'oscilación_aten': {}}
    else:
        behaviors_aics = {beh: { } for beh in behaviours}

    for behavior in behaviors_aics.keys():
        type_bnds = check_behaviour_bounds(behavior, y_data)
        behaviors_aics[behavior] = simple_shape(x_data, y_data, behavior, gof=True, gof_type=gof_type, type_bnds=type_bnds)

    return behaviors_aics


def check_behaviour_bounds(behavior, y_data):
    if behavior == 'log':
        type_bnds = [-1 if y_data[-1]-y_data[0]< 0 else None][0]
    else:
        type_bnds = None

    return type_bnds

def find_best_behavior(all_beh_dt, trans_shape=None, gof_type=['aic']):
    fited_behaviors = {g: [ ] for g in gof_type}

    if trans_shape is None:
        # gof_dict = {key: val['gof']['aic'] for key, val in all_beh_dt.items() if not np.isnan(val['gof']['aic'])}
        gof_dict = {key: {g: v for g, v in va['gof'].items()} for key, va in all_beh_dt.items() if isinstance(va['gof'], dict)}
    else:
        gof_dict = {key: {g: v[0, trans_shape] for g, v in va['gof'].items()} for key, va in all_beh_dt.items() if isinstance(va['gof'], dict)}

    for gof in gof_type:
        g_dict = sorted({p: gf[gof] for p, gf in gof_dict.items()}.items(), key=lambda x: x[1])  # list of tuple [('ocsi', -492),()]
        if gof != 'mic':
            fited_behaviors[gof].extend(g_dict[0])
        else:
            fited_behaviors[gof].extend(g_dict[-1])
        # m = 1
        # while m < len(gof_dict):
        #     if gof_dict[m][1] - gof_dict[0][1] > 2:
        #         break
        #     else:
        #         fited_behaviors.append(gof_dict[m])
        #     m += 1

    return fited_behaviors, gof_dict


def superposition(x_data, y_data, gof_type = ['aic'], behaviours=None):
    behaviors_aics = forma(x_data, y_data, gof_type, behaviours)

    b_param = dict(behaviors_aics)
    for behavior in behaviors_aics:
        y_predict = predict(x_data, behaviors_aics[behavior]['bp_params'], behavior)  ## how to use linear(x, x_data) ??
        # if any(np.isnan(y_predict)):
        #     for b in gof_type:
        #         behaviors_aics[behavior]['gof'][b] = np.nan
        resid = y_data - y_predict

        osci = simple_shape(x_data=x_data, y_data=resid, tipo_egr='oscilación', gof_type=gof_type, gof=True)
        y_spp_osci = predict(x_data, osci['bp_params'], 'oscilación') + y_predict
        # if np.isnan(y_spp_osci).any():
        #     spp_osci = np.nan
        # else:
        spp_osci = {i: fc[i](len(osci['bp_params']), y_spp_osci, y_data) for i in gof_type}

        osci_atan = simple_shape(x_data=x_data, y_data=resid, tipo_egr='oscilación_aten', gof_type=gof_type, gof=True)
        y_spp_osci_atan = predict(x_data, osci_atan['bp_params'], 'oscilación_aten') + y_predict
        # if np.isnan(y_spp_osci_atan).any():
        #     spp_osci_atan = np.nan
        # else:
        spp_osci_atan = {i: fc[i](len(osci_atan['bp_params']), y_spp_osci_atan, y_data) for i in gof_type}

        if 'constant' in behaviors_aics[behavior]['bp_params']:
            osci['bp_params']['constant'] = behaviors_aics[behavior]['bp_params']['constant'] + osci['bp_params'][
                'constant']
            osci_atan['bp_params']['constant'] = behaviors_aics[behavior]['bp_params']['constant'] + \
                                                 osci_atan['bp_params'][
                                                     'constant']

            osci['bp_params'].update(
                {k + "_1": v for k, v in behaviors_aics[behavior]['bp_params'].items() if k != 'constant'})
            osci_atan['bp_params'].update(
                {k + "_1": v for k, v in behaviors_aics[behavior]['bp_params'].items() if k != 'constant'})
        else:
            osci['bp_params'].update(
                {k + "_1": v for k, v in behaviors_aics[behavior]['bp_params'].items()})
            osci_atan['bp_params'].update(
                {k + "_1": v for k, v in behaviors_aics[behavior]['bp_params'].items()})

        b_param.update({f'spp_oscil_{behavior}':
                            {'bp_params': osci['bp_params'], 'gof': spp_osci}})
        b_param.update({f'spp_oscil_aten_{behavior}':
                            {'bp_params': osci_atan['bp_params'], 'gof': spp_osci_atan}})

    return b_param, behaviors_aics


def de_standardize(norm_b_param, y_data, tipo_egr):
    if all(np.isnan(i) for i in y_data):
        y_data = np.zeros_like(y_data)
    if tipo_egr == 'linear':
        return {'slope': norm_b_param[0] * np.nanstd(y_data),
                'intercept': norm_b_param[1] * np.nanstd(y_data) + np.nanmean(y_data)}
    elif tipo_egr == 'exponencial':
        return {'y_intercept': norm_b_param[0] * np.nanstd(y_data),
                'g_d': norm_b_param[1],
                'constant': norm_b_param[2] * np.nanstd(y_data) + np.nanmean(y_data)}
    elif tipo_egr == 'logístico':
        return {'maxi_val': norm_b_param[0] * np.nanstd(y_data),
                'g_d': norm_b_param[1],
                'mid_point': norm_b_param[2],
                'constant': norm_b_param[3] * np.nanstd(y_data) + np.nanmean(y_data)}
    elif tipo_egr == 'inverso':
        return {'g_d': norm_b_param[0] * np.nanstd(y_data),
                'phi': norm_b_param[1],
                'constant': norm_b_param[2] * np.nanstd(y_data) + np.nanmean(y_data)}
    elif tipo_egr == 'log':
        return {'g_d': norm_b_param[0] * np.nanstd(y_data),
                'phi': norm_b_param[1],
                'constant': norm_b_param[2] * np.nanstd(y_data) + np.nanmean(y_data)}
    elif tipo_egr == 'oscilación':
        return {'amplitude': norm_b_param[0] * np.nanstd(y_data),
                'period': abs(norm_b_param[1]),
                'phi': norm_b_param[2],
                'constant': norm_b_param[3] * np.nanstd(y_data) + np.nanmean(y_data)}
    elif tipo_egr == 'oscilación_aten':
        return {'g_d': norm_b_param[0],
                'amplitude': norm_b_param[1] * np.nanstd(y_data),
                'period': abs(norm_b_param[2]),
                'phi': norm_b_param[3],
                'constant': norm_b_param[4] * np.nanstd(y_data) + np.nanmean(y_data)}


def compute_gof(y_predict, y_obs):
    return compute_nsc(y_predict, y_obs), compute_rmse(y_predict, y_obs)


def compute_rmse(y_predict, y_obs):
    return np.sqrt(np.nanmean(((y_predict - y_obs) ** 2))) #overflow encountered in square, Mean of empty slice


def nse(obs, sim):
    s, e = np.array(sim), np.array(obs)
    # s,e=simulation,evaluation
    mean_observed = np.nanmean(e)
    # compute numerator and denominator
    numerator = np.nansum((e - s) ** 2)
    denominator = np.nansum((e - mean_observed) ** 2)
    # compute coefficient
    return 1 - (numerator / denominator)


def compute_nsc(y_predict, y_obs):
    # Nash-Sutcliffe Coefficient
    return 1 - np.nansum(((y_predict - y_obs) ** 2) / np.nansum((y_obs - np.nanmean(y_obs)) ** 2))


def L(y_predict, y_obs, N=5):
    # likelihood function
    return np.exp(-N * np.nansum((y_predict - y_obs) ** 2) / np.nansum((y_obs - np.nanmean(y_obs)) ** 2))


def compute_rcc(y_predict, y_obs):
    n = len(y_predict)

    s_yy = s_ysys = s_yys = 0
    for i in range(n):
        s_yy += (y_obs[i] - np.nanmean(y_obs)) ^ 2
        s_ysys += (y_predict[i] - np.nanmean(y_predict)) ^ 2
        s_yys += (y_obs[i] - np.nanmean(y_obs)) * (y_predict[i] - np.nanmean(y_predict))

    s_yy_s = s_ybyb = s_yyb = 0
    for k in range(n):
        for i in range(k + 1, n):
            y_star = y_b = 0
            for j in range(k + 1, n):
                y_star += y_obs[j]
                y_b += y_obs[j - k]
            y_star = (1 / (n - k)) * y_star
            y_b = (1 / (n - k)) * y_b
            s_yy_s += (y_obs[i] - y_star) ^ 2
            s_ybyb += (y_obs[i - k] - y_b) ^ 2
            s_yyb += (y_obs[i] - y_star) * ((y_obs[i - k] - y_b))

    return (s_yys / np.sqrt(s_yy * s_ysys)) / (s_yyb / np.sqrt(s_yy_s * s_ybyb))


def aic(k, y_predict, y_obs):
    # https://www.researchgate.net/post/What_is_the_AIC_formula
    # deltaAIC = AICm - AIC* <2(great); (4, 7)less support; (>10)no support
    # https://stats.stackexchange.com/questions/486/negative-values-for-aicc-corrected-akaike-information-criterion
    n = len(y_obs)
    resid = y_obs - y_predict
    sse = np.nansum(resid ** 2)
    # k = of variables, small Ns is no/k<40 20/2 or 20/3
    # [2*k+(2*k+1)/(n-k-1)-2*np.log(np.exp(2*np.pi*sse/n))]
    # 2*k - 2*np.log(2*np.pi*sse/ n) +2*k*(k + 1)/(n-k-1)
    # return 2*k - 2*np.log(sse) + 2*k*(k + 1)/(n-k-1)
    return 2 * k + n * np.log(sse / n)
    # 2*k - n*np.log(np.exp(2 * np.pi * sse / n) + 1)


def bic(k, y_predict, y_obs):
    # lowest BIC is preferred.
    n = len(y_obs)
    resid = y_obs - y_predict
    sse = np.nansum(resid ** 2)
    # no = number of observations
    # sse/n +np.log(n)*(k/n)*(sse/(n-b))
    return n * np.log(sse / n) + k * np.log(n)


def mic(k, y_predict, y_obs):
    mine = minepy.MINE()
    mine.compute_score( y_predict, y_obs)
    return mine.mic()

def srm(k, y_predict, y_obs):
    '''smaller the better, with the smaller risk'''
    sse = np.nansum((y_obs - y_predict) ** 2)
    n = len(y_obs)
    #(sse/n) * (1/(1-np.sqrt((k / n) - ((k / n) * np.log(k / n)) + (np.log(n) / (2 * n)))))
    # http://home.deib.polimi.it/gatto/Articoli_miei/CoraniGattoEcography.pdf
    #https://onlinelibrary.wiley.com/doi/full/10.1111/j.0906-7590.2007.04863.x
    return (sse/n) * (1/(1-np.sqrt((k / n) - ((k / n) * np.log(k / n)) + (np.log(k / n) / (2 * n)))))

def press(k, y_predict, y_obs):
    sse = np.nansum((y_obs - y_predict) ** 2)
    n = len(y_obs)
    return (sse/n) * (1 + ((2*k)/n))

def fpe(k, y_predict, y_obs):
    '''Akaike's final prediction error (FPE) (Akaike 1970), the model with minimum fpe is selected'''
    #https://onlinelibrary.wiley.com/doi/full/10.1111/j.0906-7590.2007.04863.x
    sse = np.nansum((y_obs - y_predict) ** 2)
    n = len(y_obs)
    return (sse/n) * ((n+k)/(n-k))


def ICC_rep_anova(Y):
    '''
    Calculate Calculate intraclass correlation coefficient for model validation
    Code coppied from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = mean(Y)
    SST = ((Y - mean_Y)**2).sum()

    # create the design matrix for the different levels
    x = kron(eye(nb_conditions), ones((nb_subjects, 1)))  # sessions
    x0 = tile(eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = hstack([x, x0])

    # Sum Square Error
    predicted_Y = dot(dot(dot(X, pinv(dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals**2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((mean(Y, 0) - mean_Y)**2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) /
    #            (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    return ICC, r_var, e_var, session_effect_F, dfc, dfe


def icc(Y, icc_type='icc2'):
    ''' Calculate intraclass correlation coefficient for data within
                     Brain_Data class

                 Code coppied from:
                 https://github.com/cosanlab/nltools/blob/master/nltools/data/brain_data.py

                 ICC Formulas are based on:
                 Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
                 assessing rater reliability. Psychological bulletin, 86(2), 420.
                 icc1:  x_ij = mu + beta_j + w_ij
                 icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
                 Code modifed from nipype algorithms.icc
                 https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py
                 Args:
                     icc_type: type of icc to calculate (icc: voxel random effect,
                             icc2: voxel and column random effect, icc3: voxel and
                             column fixed effect)
                 Returns:
                     ICC: (np.array) intraclass correlation coefficient
                 '''

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc / n

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'icc2':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'icc3':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC

def theil_inequal(y_predict, y_obs):
    not_nan = np.where(~np.isnan(y_obs))
    y_obs = y_obs[not_nan]
    y_predict = y_predict[not_nan]

    mu_pred = np.average(y_predict)
    mu_obs = np.average(y_obs)

    std_pred = np.std(y_predict)
    std_obs = np.std(y_obs)

    r = np.average(((y_obs - mu_obs) * (y_predict - mu_pred)) / (std_obs * std_pred))

    mse = np.average((y_predict - y_obs) ** 2)
    # um = np.abs((mu_pred ** 2 - mu_obs ** 2 ) / mse)
    # us = np.abs((std_pred ** 2 - std_obs ** 2) / mse)
    # uc = np.abs(2 * (1 - r) * std_pred * std_obs / mse)
    um = (mu_pred - mu_obs) ** 2 / mse
    us = (std_pred - std_obs) ** 2 / mse
    uc = 2 * (1 - r) * std_pred * std_obs / mse

    return mse, um, us, uc

fc = {'aic': aic, 'bic': bic, 'mic': mic, 'srm': srm, 'press': press, 'fpe': fpe}
