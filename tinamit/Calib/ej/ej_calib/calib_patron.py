import numpy as np
from tinamit.Calib.ej.sens_análisis import gen_mod
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.info_paráms import calib_líms_paráms, calib_mapa_paráms
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid, path

líms_paráms_final = []
líms_paráms_final.append(gen_problema(líms_paráms=calib_líms_paráms, mapa_paráms=calib_mapa_paráms, ficticia=False)[1])

mod = gen_mod()

obj_beh = ['nse', 'rmse' ]
obj_num = ['aic', 'mic']

def _calib(tipo_proc, guardar, método, guar_sim, obj_func, cls=False, reverse=True, egr_spotpy=None, warmup_period=None):
    if reverse:
        bd = ori_valid
    else:
        bd = ori_calib
    calib_res = mod.calibrar(paráms=list(líms_paráms_final), bd=bd, líms_paráms=calib_líms_paráms,
                             vars_obs='mds_Watertable depth Tinamit', final_líms_paráms=líms_paráms_final[0],
                             mapa_paráms=calib_mapa_paráms, tipo_proc=tipo_proc, guardar=guardar, método=método,
                             n_iter=500,
                             guar_sim=guar_sim, warmup_period=warmup_period, cls=cls, obj_func=obj_func,
                             egr_spotpy=egr_spotpy)
    return calib_res


def _valid(tipo_proc, guardar, valid_sim, n_sim, lg, obj_func, reverse=True, warmup_period=None):
    if reverse:
        bd = ori_calib
    else:
        bd = ori_valid
    valid_res = mod.validar(bd=bd, var='mds_Watertable depth Tinamit', tipo_proc=tipo_proc,
                            guardar=guardar, lg=lg, paralelo=True, valid_sim=valid_sim,
                            n_sim=n_sim, warmup_period=warmup_period, obj_func=obj_func)
    return valid_res


def _input_dt(method, obj_func, c_v, egr_spotpy=None):
    guard = path + f"real_run\\{method}\\"

    if obj_func in ['aic', 'mic']:
        tipo_proc = 'patrón'
        sim_path = path + f"simular\\dec\\{method}\\"
        if c_v == 'valid':
            lg = np.load(guard + f'nov\\calib_{obj_func}.npy').tolist()

    elif obj_func in ['nse', 'rmse']:
        tipo_proc = 'multidim'
        sim_path = path + f"simular\\dec\\{method}\\"
        if c_v == 'valid':
            lg = np.load(guard + f'nov\\calib_{obj_func}.npy').tolist()

    if c_v == 'valid':

        d_val = {'dream': {'aic': (144, 644), 'nse': (0, 500), 'rmse': (0, 500), 'mic': (0, 500)},
                 'mle': {'aic': (0, 500), 'nse': (500, 969), 'rmse': (500, 1000), 'mic': (500, 1000)},
                 'fscabc': {'aic': (0, 144), 'nse': (0, 500), 'rmse': (0, 500), 'mic': (0, 144)},
                 'demcz': {'aic': (644, 1144), 'nse': (500, 1000), 'rmse': (1000, 1500), 'mic': (0, 500)}}

        n_sim = d_val[method][obj_func]

        guardar = guard + f'nov\\valid_{obj_func}41'
        valid_sim = sim_path + f'{obj_func}\\'

        return tipo_proc, n_sim, guardar, valid_sim, lg

    elif c_v == 'calib':
        if egr_spotpy is not None:
            egr_spotpy = sim_path + f'nov\\{method}_{obj_func}.csv'

        guardar = guard + f'dec\\calib_{obj_func}'
        guar_sim = sim_path + f'{obj_func}\\'
        return tipo_proc, guardar, guar_sim, egr_spotpy


if __name__ == "__main__":
    egr_spotpy = None
    warmup_period = None # 2
    calib_valid = 'calib'

    for m in ['dream','fscabc', 'mle', 'demcz']:
        for obj_func in ['aic', 'mic', 'nse', 'rmse']:
            if calib_valid == 'calib':
                tipo_proc, guardar, guar_sim, egr_spotpy = _input_dt(method=m, obj_func=obj_func, c_v=calib_valid, egr_spotpy=egr_spotpy)
                _calib(tipo_proc=tipo_proc, guardar=guardar, método=m, obj_func=obj_func,
                       guar_sim=guar_sim, egr_spotpy=egr_spotpy, warmup_period=warmup_period)

            elif calib_valid == 'valid':
                tipo_proc, n_sim, guardar, valid_sim, lg = _input_dt(method=m, obj_func=obj_func, c_v=calib_valid)
                _valid(tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func, n_sim=n_sim, lg=lg, warmup_period=warmup_period)