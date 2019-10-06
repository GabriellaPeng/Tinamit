import os
import multiprocessing as mp
import time
from copy import deepcopy
from datetime import datetime
from xarray import Dataset

from tinamit.Análisis.Sens.behavior import find_best_behavior
from tinamit.Análisis.Sens.corridas import simul_sens
from tinamit.Análisis.Sens.muestr import cargar_mstr_paráms, gen_problema, muestrear_paráms
# from tinamit.Calib.ej.sens_análisis import plot_4_select_criteria
from tinamit.Calib.ej.sens_análisis import plot_4_select_criteria
from tinamit.cositas import cargar_json
from tinamit.Análisis.Sens import behavior

import numpy as np
from SALib.analyze import morris
from SALib.analyze import fast
from collections import Counter

from tinamit.config import _


def _anlzr_salib(método, problema, mstr, simul, ops_método):
    '''
    Parameters
    ----------
    método: ['morris', 'fast']
    líms_paráms, mapa_paráms se usan para generar el problema.
    mstr: sampled data
    simul: simulated model output
    ops_método: dict
    #{'num_levels': 8, 'grid_jump': 4}

    Returns
    -------

    '''

    if isinstance(mstr, list):
        mstr = np.asarray(mstr)

    if método == 'morris':
        if isinstance(simul, list):
            simul = np.asarray(simul)
        ops = {'X': mstr, 'Y': simul, 'num_levels': 16}
        ops.update(ops_método)

        mor = {}
        Si = morris.analyze(problema, **ops)

        mor.update(
            {'mu_star': {j: Si['mu_star'][i] for i, j in enumerate(problema['names'])},
             'sigma': {j: Si['sigma'][i] for i, j in enumerate(problema['names'])},
             # 'sum_sigma': np.sum(Si['sigma'])
             })
        return mor

    elif método == 'fast':
        ops = {'Y': simul}

        fa = {}
        Si = fast.analyze(problema, **ops)

        fa.update({'Si': {j: Si['S1'][i] for i, j in enumerate(problema['names'])},
                   'ST': {j: Si['ST'][i] for i, j in enumerate(problema['names'])},
                   'St-Si': {j: Si['ST'][i] - Si['S1'][i] for i, j in enumerate(problema['names'])},
                   # 'sum_s1': np.sum(Si['S1']),
                   # 'sum_st': np.sum(Si['ST']),
                   # 'sum_st-s1': np.absolute(np.sum(Si['ST']) - np.sum(Si['S1']))
                   })
        return fa

    else:
        raise ValueError(_('Algoritmo "{}" no reconocido.').format(método))


def analy_by_file(método, líms_paráms, mapa_paráms, mstr_arch, simul_arch, var_egr=None,
                  ops_método=None, tipo_egr=None, ficticia=True, f_simul_arch=None, dim=None, gof_type=None, plot_path=None):
    método = método.lower()
    if tipo_egr:
        tipo_egr = tipo_egr.lower()

    if ops_método is None:
        ops_método = {}

    mstr = cargar_mstr_paráms(mstr_arch)
    # mstr_new = {p: [] for p in mstr}
    # for p in mstr:
    #     for i in range(4):
    #         mstr_new[p].append(mstr[p][5]) # take sample 5

    simulation, var_egr = carg_simul_dt(simul_arch['arch_simular'], simul_arch['num_samples'],
                                        var_egr, dim, método=método,
                                        tipo_egr=tipo_egr)  # simulation: '625' * xarray(21*215); '625'*ndarray[21]

    return anlzr_simul(
        método=método, líms_paráms=líms_paráms, mstr=mstr, mapa_paráms=mapa_paráms, ficticia=ficticia,
        simulation=simulation, var_egr=var_egr, ops_método=ops_método, tipo_egr=tipo_egr,
        f_simul_arch=f_simul_arch, dim=dim, gof_type=gof_type, plot_path=plot_path
    )

def singular_behav_proc(simul_arch, num_samples, var_egr=None, tipo_egr=None, dim=None, guardar=None, gof_type=None, plot_path=None, f_simul_path=None):
    if tipo_egr:
        tipo_egr = tipo_egr.lower()

    counted_all_behaviors = {gof: [] for gof in gof_type}
    i = num_samples

    if isinstance(dim, list):
        dim =[i-1 for i in dim]

    simulation, var_egr = carg_simul_dt(simul_arch, i, var_egr, dim, load_i_file=True)

    # f_simul = format_simul(simulation=simulation, vr=vr, tipo_egr=tipo_egr, dim=dim)
    print(f"for sample {i} ")

    uni_behav_anlzr(tipo_egr, simulation[str(i)], var_egr, i, dim, counted_all_behaviors, gof_type, None,
                                            guardar + f'f_simul_{i}', plot_path)

    # np.save(guardar + f'counted_all_behaviors_{start}_{end}', counted_all_behaviors)


def parallel_behav_proc(simul_arch, var_egr=None, tipo_egr=None, dim=None, guardar=None, gof_type=None, plot_path=None, f_simul_path=None):
    if tipo_egr:
        tipo_egr = tipo_egr.lower()

    counted_all_behaviors = {gof: [] for gof in gof_type}
    pool = mp.Pool(processes=4)

    count = 1
    start = 0
    end = 625 #simul_arch['num_samples'] #625
    for i in range(start, end):
        simulation, var_egr = carg_simul_dt(simul_arch, i, var_egr, dim, load_i_file=True)

        # f_simul = format_simul(simulation=simulation, vr=vr, tipo_egr=tipo_egr, dim=dim)

        print(f"for sample {i} ")

        pool.apply_async(uni_behav_anlzr, args=(tipo_egr, simulation[str(i)],
                                                var_egr, i, dim, counted_all_behaviors, gof_type, None,
                                                guardar + f'f_simul_{i}', plot_path))

        if count % 4 == 0:
            time.sleep(60)
        count += 1

    np.save(guardar + f'counted_all_behaviors_{start}_{end}', counted_all_behaviors)


def anlzr_sens(mod, método, líms_paráms, mapa_paráms, t_final, var_egr,
               ops_método=None, tipo_egr=None, ficticia=True, f_simul_arch=None, simulation=None, dim=None):
    método = método.lower()
    if tipo_egr:
        tipo_egr = tipo_egr.lower()

    if ops_método is None:
        ops_método = {}

    mstr = muestrear_paráms(líms_paráms, método, mapa_paráms=mapa_paráms, ficticia=ficticia)
    # n_param * Ns
    if simulation is None:
        simulation = simul_sens(mod, mstr_paráms=mstr, mapa_paráms=mapa_paráms, t_final=t_final, var_egr=var_egr,
                                guardar=False)  # 250(Ns)* xarray
    else:
        simulation = simulation
    # 150 NS * xarray (ts*3)=simulation['0']['y'].values (150*21*3)
    # import matplotlib.pyplot as ppt
    # for i in range(len(simulation)):
    #     ppt.plot(simulation[str(i)]['y'].values)
    #     ppt.savefig(f"D:\\Thesis\\pythonProject\\png\\{i}.png")
    #     ppt.close()

    # sample_size*xarray --> 100*
    return anlzr_simul(
        método=método, líms_paráms=líms_paráms, mstr=mstr, mapa_paráms=mapa_paráms, ficticia=ficticia,
        simulation=simulation, var_egr=var_egr, ops_método=ops_método, tipo_egr=tipo_egr,
        f_simul_arch=f_simul_arch, dim=dim
    )


def anlzr_simul(método, líms_paráms, mstr, mapa_paráms, ficticia, var_egr, f_simul_arch, dim,
                tipo_egr="promedio", simulation=None, ops_método=None, gof_type=None, plot_path=None):
    if ops_método is None:
        ops_método = {}
    if isinstance(tipo_egr, str):
        tipo_egr = [tipo_egr]
    if isinstance(var_egr, str):
        var_egr = [var_egr]

    problema, líms_paráms_final = gen_problema(líms_paráms, mapa_paráms, ficticia)
    egr = {tp: {v: {} for v in var_egr} for tp in tipo_egr}

    if isinstance(mstr, str):
        mstr = cargar_mstr_paráms(mstr)

    f_mstr = [np.array([mstr[para][i].tolist() for para in problema['names']]) for i in range(len(list(mstr.values())[0]))] #625*24ndarray

    for tp in tipo_egr:
        d_tp = egr[tp]

        for vr in var_egr:
            if tipo_egr == 'forma' or tipo_egr == 'superposition':
                    if método == 'morris' or método == 'fast':
                        f_simul = carg_fsimul_data(f_simul_arch['arch'],
                                                   f_simul_arch['num_sample'], f_simul_arch['counted_behaviors'], dim=dim)
                    else:
                        f_simul = carg_fsimul_data(f_simul_arch['arch'])
            else:
                f_simul = format_simul(simulation=simulation, vr=vr, tipo_egr=tp, dim=dim, método=método,
                                       gof_type=gof_type, save_path=f_simul_arch, plot_path=plot_path)

            d_var = anlzr_simul_salib(problema, f_simul, método, f_mstr, ops_método, líms_paráms, tp, dim, gof_type)

            d_tp[vr] = d_var
    return egr


def carg_fsimul_data(f_simul_path, num_sample=None, counted_behaviors=None, dim=None, gof_type=['aic', 'mic']):
    if num_sample is None and counted_behaviors is None:
        return np.load(f_simul_path).tolist()

    counted_all = np.load(counted_behaviors).tolist()

    sample_patt = { }
    for gof in gof_type:
        print(f"\nProcessing GOF {gof}\n")
        sam_patt = {patt: np.load(f"{f_simul_path}_0.npy").tolist()[patt] for patt in counted_all[gof]}
        sample_patt[gof] = _gen_d_patt(num_sample, counted_all[gof], sam_patt, gof, dim)

        for j in range(num_sample):
            behav = np.load(f"{f_simul_path}_{j}.npy").tolist()
            for patt in counted_all[gof]:
                for bpp, d1 in behav[patt]['bp_params'].items():
                    if num_sample < 10000:
                        sample_patt[gof][patt]['bp_params'][bpp][j, :] = d1[0]
                        sample_patt[gof][patt]['gof'][gof][j, :] = behav[patt]['gof'][gof][0]
                    else:
                        sample_patt[gof][patt]['bp_params'][bpp][j, :] = d1[0, dim]
                        sample_patt[gof][patt]['gof'][gof][j, :] = behav[patt]['gof'][gof][0, dim]
    return sample_patt


def _gen_d_patt(num_sample, counted_behaviors, sam_patt, gof, dim):
    for behav in counted_behaviors:
        for bpp, d in sam_patt[behav]['bp_params'].items():
            if num_sample < 10000:
                sam_patt[behav]['bp_params'][bpp] = np.zeros([num_sample, dim])  # 625*215
                sam_patt[behav]['gof'] = {gof : np.zeros([num_sample, dim])}
            else:
                sam_patt[behav]['bp_params'][bpp] = np.zeros([num_sample, 1])
                sam_patt[behav]['gof'] = {gof : np.zeros([num_sample, 1])}

    return sam_patt


def anlzr_simul_salib(problema, f_simul, método, f_mstr, ops_método, líms_paráms, tipo_egr, dim, gof_type=['aic']): #TODO need to fix here!!
    def _f_simul_behav(f_simul, dict_var, dim):
        for ll, v in f_simul.items():
            if v.ndim == 1:  # only Ns
                res = _anlzr_salib(método=método, problema=problema,
                                   mstr=f_mstr, simul=v, ops_método=ops_método)

                for si, val in res.items():
                    if isinstance(val, dict):
                        for pa, p_var in val.items():
                            dict_var[si][pa][ll] = p_var
                    else:
                        dict_var[si][ll] = val

            else:
                for i in range(v.shape[1]):  # [1] -> dimentions (poly)
                    res = _anlzr_salib(método=método, problema=problema,
                                       mstr=f_mstr, simul=v[:, i], ops_método=ops_método)
                    for si, val in res.items():
                        if isinstance(val, dict):
                            for key, va in líms_paráms.items():
                                if isinstance(va, list):
                                    if dict_var[si][key][ll].shape != (len(va), dim):  # 4*215 TODO maybe need to change dim
                                        dict_var[si][key][ll] = np.zeros([len(va), dim])  # make it to 4*215
                                    for j in range(len(va)):  # j-4
                                        dict_var[si][key][ll][j, i] = val[key + '_' + str(j)]
                                else:
                                    dict_var[si][key][ll][i] = res[si][key]
                        else:
                            dict_var[si][ll][i] = val

    líms_paráms = deepcopy(líms_paráms)
    if 'Ficticia' in problema['names']:
        líms_paráms['Ficticia'] = (0, 1)

    if isinstance(f_simul, dict):
        if tipo_egr == 'forma' or tipo_egr == 'superposition':
            d_var = {gof: {b_p: {bp_gof: {} for bp_gof in d_vals} for b_p, d_vals in f_simul[gof].items()} for gof
                     in gof_type}
            for gof in gof_type:
                for bp, bp_dict in f_simul[gof].items():
                    print(f'Processing Pattern {bp}\n')
                    for b_g, bg_val in bp_dict.items():
                        dict_var = gen_d_var(dim, bg_val, líms_paráms, método)  # only the {si: n-dim}
                        _f_simul_behav(bg_val, dict_var, dim)
                        d_var[gof][bp][b_g] = dict_var

        elif tipo_egr == 'paso_tiempo':
            d_var = gen_d_var(dim, f_simul, líms_paráms, método)
            _f_simul_behav(f_simul, d_var, dim)


    elif f_simul.ndim > 1:  # mean, final (Ns*dim)
        tmñ = f_simul.shape[1:]  # simple=215, multidim=ts * n_dim (6*3), paso=(ts*n)
        d_var = gen_d_var(tmñ, f_simul, líms_paráms, método)

        for i in range(tmñ[0]):  # i=dim
            res = _anlzr_salib(método=método, problema=problema,
                               mstr=f_mstr, simul=f_simul[:, i], ops_método=ops_método)

            for si, val in res.items():
                if len(list(res.values())[0]) == len(líms_paráms):  # dummy 24==9
                    if isinstance(val, dict):  # sum_sigma
                        for pa, p_var in val.items():
                            d_var[si][pa][i] = p_var
                    else:
                        d_var[si][i] = val
                else:
                    if isinstance(val, dict):  # val={prms: val}
                        for key, va in líms_paráms.items():
                            if isinstance(va, list):  # all bf params
                                if d_var[si][key].shape != (len(va), tmñ[0]):
                                    d_var[si][key] = np.zeros([len(va), tmñ[0]])
                                for j in range(len(va)):
                                    d_var[si][key][j, i] = val[key + '_' + str(j)]

                            else:
                                d_var[si][key][i] = res[si][key]
                    else:
                        d_var[si][i] = val
    else:
        d_var = {}
        d_var.update(
            _anlzr_salib(método=método, problema=problema, mstr=f_mstr,
                         simul=f_simul, ops_método=ops_método))
    return d_var


def gen_d_var(tmñ, f_simul, líms_paráms, método):
    if isinstance(f_simul, dict):

        if método == 'morris':
            d_var = {'mu_star': {name: {ll: np.zeros(tmñ) for ll in f_simul} for name in líms_paráms},
                     'sigma': {name: {ll: np.zeros(tmñ) for ll in f_simul} for name in líms_paráms},
                     # 'sum_sigma': {ll: np.zeros(tmñ) for ll in f_simul}
                     }
        elif método == 'fast':
            d_var = {'Si': {name: {ll: np.zeros(tmñ) for ll in f_simul} for name in líms_paráms},
                     'ST': {name: {ll: np.zeros(tmñ) for ll in f_simul} for name in líms_paráms},
                     'St-Si': {name: {ll: np.zeros(tmñ) for ll in f_simul} for name in líms_paráms},
                     # 'sum_s1': {ll: np.zeros(tmñ) for ll in f_simul},
                     # 'sum_st': {ll: np.zeros(tmñ) for ll in f_simul},
                     # 'sum_st-s1': {ll: np.zeros(tmñ) for ll in f_simul}
                     }
        else:
            raise ValueError(método)
    else:
        if método == 'morris':
            d_var = {'mu_star': {name: np.zeros(tmñ) for name in líms_paráms},
                     'sigma': {name: np.zeros(tmñ) for name in líms_paráms},
                     # 'sum_sigma': np.zeros(tmñ)
                     }
        elif método == 'fast':
            d_var = {'Si': {name: np.zeros(tmñ) for name in líms_paráms},
                     'ST': {name: np.zeros(tmñ) for name in líms_paráms},
                     'St-Si': {name: np.zeros(tmñ) for name in líms_paráms},
                     # 'sum_s1': np.zeros(tmñ),
                     # 'sum_st': np.zeros(tmñ),
                     # 'sum_st-s1': np.zeros(tmñ)
                     }
        else:
            raise ValueError(método)
    return d_var


def gen_bf_simul(tipo_egr, tmñ, gof_type, gof=False):
    if tipo_egr == "linear":
        bf_simul = {'slope': np.zeros(tmñ), 'intercept': np.zeros(tmñ)}

    elif tipo_egr == "exponencial":
        bf_simul = {'y_intercept': np.zeros(tmñ), 'g_d': np.zeros(tmñ), 'constant': np.zeros(tmñ)}

    elif tipo_egr == "logístico":
        bf_simul = {'maxi_val': np.zeros(tmñ), 'g_d': np.zeros(tmñ),
                    'mid_point': np.zeros(tmñ), 'constant': np.zeros(tmñ)}

    elif tipo_egr == "inverso":
        bf_simul = {'g_d': np.zeros(tmñ), 'phi': np.zeros(tmñ), 'constant': np.zeros(tmñ)}

    elif tipo_egr == "log":
        bf_simul = {'g_d': np.zeros(tmñ), 'phi': np.zeros(tmñ), 'constant': np.zeros(tmñ)}

    elif tipo_egr == "oscilación":
        bf_simul = {'amplitude': np.zeros(tmñ), 'period': np.zeros(tmñ),
                    'phi': np.zeros(tmñ), 'constant': np.zeros(tmñ)}

    elif tipo_egr == "oscilación_aten":
        bf_simul = {'g_d': np.zeros(tmñ), 'amplitude': np.zeros(tmñ),
                    'period': np.zeros(tmñ), 'phi': np.zeros(tmñ), 'constant': np.zeros(tmñ)}

    elif tipo_egr == "forma":
        bf_simul = {}

    elif tipo_egr == 'superposition':
        bf_simul = {}
        basic_behaviors = ['linear', 'exponencial', 'logístico', 'inverso', 'log', 'oscilación', 'oscilación_aten']
        for beh in basic_behaviors:
            beh_simul = gen_bf_simul(beh, tmñ, gof_type=gof_type, gof=False)
            oscil = gen_bf_simul("oscilación", tmñ, gof_type=gof_type, gof=True)
            oscil_aten = gen_bf_simul("oscilación_aten", tmñ, gof_type=gof_type, gof=True)
            oscil['bp_params'].update({k + "_1": v for k, v in beh_simul.items()})
            oscil_aten['bp_params'].update({k + "_1": v for k, v in beh_simul.items()})

            bf_simul.update({beh: gen_bf_simul(beh, tmñ, gof_type=gof_type, gof=True)})
            bf_simul.update({f'spp_oscil_{beh}': oscil})
            bf_simul.update({f'spp_oscil_aten_{beh}': oscil_aten})

    else:
        raise ValueError(tipo_egr)

    if gof:
        bf_simul = {'bp_params': bf_simul, 'gof': {b: np.zeros(tmñ) for b in gof_type} }

    return bf_simul


def format_simul(simulation, vr, tipo_egr, dim, método=None, gof_type=None, save_path=None, plot_path=None):
    """

    :param simulation: { 'sample_id': {'y': []}}
    :param vr:
    :param tipo_egr:
    :return: [] or {'p': []}
    """
    if método == 'morris':
        if dim is not None:
            tmñ = [len(simulation), dim]  # Ns * n-dim (625*215)
        else:
            tmñ = [len(simulation)]
    elif método == 'fast':
        tmñ = [len(simulation), 1]

    if tipo_egr == "promedio":
        f_simul = np.zeros(tmñ)
        for i, val in enumerate(simulation.values()):
            if método == 'morris' and dim is not None:
                for j in range(dim):
                    f_simul[i, j] = np.average(val[:, j])
            else:
                f_simul[i] = np.average(val)

    elif tipo_egr == "final":
        f_simul = np.zeros(tmñ)
        for i, val in enumerate(simulation.values()):
            f_simul[i] = val[-1]

    elif tipo_egr == "paso_tiempo":
        if método == 'morris' and dim is not None:
            paso = [0, 4, 9, 14, 20]
            f_simul = {f'paso_{j}': np.zeros(tmñ) for j in paso}
            for i, sam in simulation.items():
                for j in paso:
                    f_simul[f'paso_{j}'][int(i)] = simulation[i][j, :]

        else:
            f_simul = {f'paso_{i}': np.zeros(tmñ) for i in range(len(simulation['0']))}
            for i, sam in simulation.items():
                for j, val in enumerate(sam):
                    f_simul[f'paso_{j}'][int(i)] = val

    else:
        f_simul = behavior_anlzr(simulation, vr, tipo_egr, dim, gof_type, tmñ, save_path=save_path, plot_path=plot_path)
    return f_simul


def behavior_anlzr(simulation, vr, tipo_egr, dim, gof_type, tmñ, save_path=None, plot_path=None):
    counted_all_behaviors = {gof: [ ] for gof in gof_type}

    start_time = datetime.now().strftime("%H:%M:%S")
    FMT = '%H:%M:%S'
    print(f"Initializing behavior {start_time} ")

    bf_simul = gen_bf_simul(tipo_egr, tmñ, gof_type)
    for i, val in enumerate(simulation.values()):
        bf_simul = uni_behav_anlzr(tipo_egr, val, vr, i, dim, counted_all_behaviors, gof_type, bf_simul, save_path=None, plot_path=plot_path)

    print(
        f"Counting over \
        {datetime.strptime(datetime.now().strftime('%H:%M:%S'), FMT) - datetime.strptime(start_time, FMT)}")

    if len(counted_all_behaviors):
        all_behaviors = set(counted_all_behaviors)

        new_bf = {patt: bf_simul[patt] for patt in all_behaviors}
        return new_bf

    print(
        f"Behavior analysis elapse \
    {datetime.strptime(datetime.now().strftime('%H:%M:%S'), FMT) - datetime.strptime(start_time, FMT)}")
    return bf_simul


def uni_behav_anlzr(tipo_egr, val, vr, sam_ind, dim, counted_all_behaviors, gof_type, bf_simul=None, save_path=None,
                    plot_path=None):
    # all_beh_dt = np.load("D:\Gaby\Tinamit\Dt\Mor\\f_simul\\aug\\all_beh_dt.npy").tolist()
    if not isinstance(val, np.ndarray):
        y_data = val[vr].values[2:, :]  ## add [1:, :]  for test file
    else:
        y_data = val

    if dim is None:
        dim = y_data.shape[1]
    # if dim is None and y_data.ndim == 1:
    #     b_param = behavior.simple_shape(np.asarray(range(val[vr].values.size)), y_data, tipo_egr)['bp_params']
    #
    #     for k, v in bf_simul.items():
    #         bf_simul[k][sam_ind] = b_param[k]

    # else:  # simple shape val = 21*10 (21*215)

    all_beh_dt = []

    for j in range(y_data.shape[1]):  # j- dimension polys
        y_dt = y_data[:, j]
        print(f"Polygon {j} is under processing!")

        if tipo_egr == 'superposition':
            b_param = behavior.superposition(np.arange(len(y_dt)), y_dt, gof_type)[0]
            all_beh_dt.append(b_param)
        elif tipo_egr == 'forma':
            b_param = behavior.forma(np.arange(len(y_dt)), y_dt, gof_type)
            all_beh_dt.append(b_param)
        else:
            b_param = behavior.simple_shape(np.arange(len(y_dt)), y_dt, tipo_egr, gof_type)
            for k, v in bf_simul.items():
                bf_simul[k][sam_ind, j] = b_param['bp_params'][k]

    fited_behaviors = {i: { } for i in range(len(all_beh_dt))}

    for i, d_beh in enumerate(all_beh_dt):
        fited_behaviors[i].update(find_best_behavior(d_beh, gof_type=gof_type)[0])
        for gof in gof_type:
            counted_all_behaviors[gof].extend([fited_behaviors[i][gof][0]])
            # counted_all_behaviors[gof].extend(list(Counter([k for k, v in fited_behaviors[i][gof]]).keys()))
        # detail_info = {pattern: [i for i, val in enumerate(fited_behaviors) if val[0] == pattern] for pattern in counted_behaviors}
    if plot_path is not None:
        plot_4_select_criteria(sam_ind, y_data, counted_all_behaviors, all_beh_dt, plot_path, dim)

    else:
        if bf_simul is None:
            bf_simul = gen_bf_simul(tipo_egr, [1, dim], gof_type)
            ind = 0
        else:
            ind = sam_ind

        for fit_behav in bf_simul:  # detected best behaviors within the samples
            for j in range(len(all_beh_dt)):  # 215 or 1
                for b_g, bg_dict in all_beh_dt[j][fit_behav].items():
                    for bpp, bpp_va in bg_dict.items():
                        bf_simul[fit_behav][b_g][bpp][ind, j] = bpp_va

        if save_path is not None:
            np.save(save_path, bf_simul)
        else:
            return bf_simul


def carg_simul_dt(arch_simular, num_samples, var_egr=None, dim=None, tipo_egr=None, método=None, load_i_file=False):
    if arch_simular is None:
        raise ValueError(_("la ruta de simulación es ninguna."))

    simulation_data = {}  # {625('0')× xarray} or {'0': ndarray}

    if dim is None:
        for i in range(num_samples):
            if tipo_egr == 'promedio':
                # print(i, float(getsizeof(simulation_data)/(1024*1024)), 'MB')
                if método == 'morris':
                    simulation_data.update(
                        {str(i): Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{i}')))
                                 [var_egr].values[2:, :]})
                elif método == 'fast':
                    print(f'Loading sample the-{i}th')
                    simulation_data.update(
                        {str(i): np.average(Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{i}')))
                                            [var_egr].values[2:, ])})
            elif tipo_egr == 'paso_tiempo':
                if método == 'morris':
                    simulation_data.update(
                        {str(i):Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{i}')))[var_egr].values})
                elif método == 'fast':
                    print(f'Loading sample the-{i}th')
                    indices = [0, 4, 9, 14, 20]
                    simulation_data.update(
                        {str(i): np.take(
                            Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{i}')))[var_egr].values[:, ],
                            indices)})
            else:
                simulation_data.update({str(i): Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{i}')))[var_egr].values[2:, :]})

    elif load_i_file and not isinstance(dim, list):
        simulation_data.update(
            {str(num_samples): Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{num_samples}')))[var_egr].values[2:, :]})

    elif load_i_file and isinstance(dim, list):
        simulation_data.update(
            {str(num_samples): Dataset.from_dict(cargar_json(os.path.join(arch_simular, f'{num_samples}')))[
                                   var_egr].values[2:, :][:, [i for i in dim]]})

    if var_egr is None:
        var_egr = [list(list(simulation_data.values())[0].data_vars.variables.mapping)[1]]  # [0] - soil salinity, 1 wtd

    return simulation_data, var_egr