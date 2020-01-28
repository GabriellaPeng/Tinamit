import numpy as np
from tinamit.Calib.ej.sens_análisis import gen_mod
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.info_paráms import calib_líms_paráms, calib_mapa_paráms, calib_líms_paráms_reduce_sd
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid, path

líms_paráms =  calib_líms_paráms_reduce_sd

líms_paráms_final = [gen_problema(líms_paráms=líms_paráms, mapa_paráms=calib_mapa_paráms, ficticia=False)[1]]

mod = gen_mod()

obj_beh = ['nse', 'rmse' ]
obj_num = ['aic', 'mic']

def _calib(tipo_proc, guardar, método, guar_sim, obj_func, cls=False, reverse=True, egr_spotpy=None, warmup_period=None, simluation_res=None, ind_simul=None):
    if reverse:
        bd = ori_valid
    else:
        bd = ori_calib
    calib_res = mod.calibrar(paráms=líms_paráms_final, bd=bd, líms_paráms=líms_paráms,
                             vars_obs='mds_Watertable depth Tinamit', final_líms_paráms=líms_paráms_final[0],
                             mapa_paráms=calib_mapa_paráms, tipo_proc=tipo_proc, guardar=guardar, método=método,
                             n_iter=500,
                             guar_sim=guar_sim, warmup_period=warmup_period, cls=cls, obj_func=obj_func,
                             egr_spotpy=egr_spotpy, simluation_res=simluation_res, ind_simul=ind_simul)
    return calib_res


def _valid(tipo_proc, guardar, valid_sim, n_sim, lg, obj_func, reverse=True, warmup_period=None, calib_res=None):
    if reverse:
        bd = ori_calib
    else:
        bd = ori_valid
    valid_res = mod.validar(bd=bd, var='mds_Watertable depth Tinamit', tipo_proc=tipo_proc,
                            guardar=guardar, lg=lg, paralelo=True, valid_sim=valid_sim,
                            n_sim=n_sim, warmup_period=warmup_period, obj_func=obj_func, calib_res=calib_res)
    return valid_res


def _input_dt(method, obj_func, c_v, month, egr_spotpy=None, simluation_res=None):
    guard = path + f"real_run\\{method}\\{month}\\"
    tipo_proc = ['patrón' if obj_func in ['aic', 'mic'] else 'multidim' if obj_func in ['nse', 'rmse'] else ''][0]

    if c_v == 'valid':
        lg = np.load(guard + f'calib_{obj_func}.npy').tolist()
        sim_path = path + f"simular\\{month}\\{method}\\"

        if month == 'nov':
            d_val = {'dream': {'aic': (144, 644), 'nse': (0, 500), 'rmse': (0, 500), 'mic': (0, 500)},
                     'mle': {'aic': (0, 500), 'nse': (500, 969), 'rmse': (500, 1000), 'mic': (500, 1000)},
                     'fscabc': {'aic': (0, 144), 'nse': (0, 500), 'rmse': (0, 500), 'mic': (0, 144)},
                     'demcz': {'aic': (644, 1144), 'nse': (500, 1000), 'rmse': (1000, 1500), 'mic': (0, 500)}}

        elif month == 'dec':
            if obj_func in ['aic', 'mic']:
                d_val = {'dream': {'aic': (1000, 1500), 'mic': (0, 500)},
                         'mle': {'aic': (288, 788), 'mic': (788, 1288)},
                         'fscabc': {'aic': (0, 144), 'mic': (144, 288)},
                         'demcz': {'aic': (0, 500), 'mic': (500, 1000)}}
                valid_sim = sim_path + f'{obj_func}\\'

            elif obj_func in ['nse', 'rmse']:
                d_val = {'dream': {'nse': (0, 500), 'rmse': (0, 500)},
                         'mle': {'nse': (500, 969), 'rmse': (500, 1000)},
                         'fscabc': {'nse': (0, 500), 'rmse': (0, 500)},
                         'demcz': {'nse': (500, 1000), 'rmse': (1000, 1500)}}
                valid_sim = path + f"simular\\aug\\{method}\\{obj_func}\\"

        n_sim = d_val[method][obj_func]
        guardar = guard + f'valid_{obj_func}'

        return tipo_proc, n_sim, guardar, valid_sim, lg

    elif c_v == 'calib':
        # sim_path = path + f"simular\\{month}\\{method}\\"
        sim_path = path + f"simular\\aug\\{method}\\"

        if egr_spotpy is not None:
            egr_spotpy =egr_spotpy + f'{method}_{obj_func}.csv'

        if simluation_res:
            d_val = {'dream': {'nse': (0, 500), 'rmse': (0, 500)},
                     'mle': {'nse': (500, 969), 'rmse': (500, 1000)},
                     'fscabc': {'nse': (0, 500), 'rmse': (0, 500)},
                     'demcz': {'nse':(500, 1000), 'rmse': (1000, 1500)}}
            simluation_res = sim_path+ f'{obj_func}\\'
            ind_simul = d_val[method][obj_func]

        else:
            ind_simul=None

        guardar = guard + f'calib_{obj_func}'
        guar_sim = sim_path + f'{obj_func}\\'

        return tipo_proc, guardar, guar_sim, egr_spotpy, simluation_res, ind_simul


if __name__ == "__main__":
    egr_spotpy = None
    warmup_period = None # 2
    simluation_ress = None
    calib_valid = 'valid'
    month='dec'

    for m in ['demcz', 'dream', 'fscabc','mle']: #'demcz', 'dream', 'fscabc','mle']: #
        for obj_func in ['nse','rmse', 'aic',  'mic']: #, 'nse', 'rmse', 'aic',  'mic'
            if calib_valid == 'calib':
                tipo_proc, guardar, guar_sim, egr_spotpy, simluation_res, ind_simul = _input_dt(method=m, obj_func=obj_func, c_v=calib_valid, egr_spotpy=egr_spotpy, month=month, simluation_res=simluation_ress)
                _calib(tipo_proc=tipo_proc, guardar=guardar, método=m, obj_func=obj_func,
                       guar_sim=guar_sim, egr_spotpy=egr_spotpy, warmup_period=warmup_period, simluation_res=simluation_res, ind_simul=ind_simul)

            elif calib_valid == 'valid':
                calib_poly = list(ori_valid)
                tipo_proc, n_sim, guardar, valid_sim, lg = _input_dt(method=m, obj_func=obj_func, c_v=calib_valid, month=month)
                _valid(tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func, n_sim=n_sim, lg=lg,
                       warmup_period=warmup_period, calib_res=calib_poly)
