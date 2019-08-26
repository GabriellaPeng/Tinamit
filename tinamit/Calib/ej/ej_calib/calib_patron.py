import numpy as np
from tinamit.Calib.ej.sens_análisis import gen_mod
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.info_paráms import calib_líms_paráms, calib_mapa_paráms
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid, path

líms_paráms_final = []
líms_paráms_final.append(gen_problema(líms_paráms=calib_líms_paráms, mapa_paráms=calib_mapa_paráms, ficticia=False)[1])

mod = gen_mod()


def _calib(tipo_proc, guardar, método, guar_sim, obj_func, cls=False, reverse=False, egr_spotpy=None):
    if reverse:
        bd = ori_valid
    else:
        bd = ori_calib
    calib_res = mod.calibrar(paráms=list(líms_paráms_final), bd=bd, líms_paráms=calib_líms_paráms,
                             vars_obs='mds_Watertable depth Tinamit', final_líms_paráms=líms_paráms_final[0],
                             mapa_paráms=calib_mapa_paráms, tipo_proc=tipo_proc, guardar=guardar, método=método,
                             n_iter=500,
                             guar_sim=guar_sim, warmup_period=2, cls=cls, obj_func=obj_func,
                             egr_spotpy=egr_spotpy)
    return calib_res


def _valid(tipo_proc, guardar, valid_sim, n_sim, save_plot, lg, obj_func, método, reverse=True):
    if reverse:
        bd = ori_calib
    else:
        bd = ori_valid
    valid_res = mod.validar(bd=bd, var='mds_Watertable depth Tinamit', tipo_proc=tipo_proc,
                            guardar=guardar, lg=lg, paralelo=True, valid_sim=valid_sim,
                            n_sim=n_sim, save_plot=save_plot, warmup_period=2, obj_func=obj_func, método=método)
    return valid_res


def _input_dt(tipo_proc, método, obj_func, c_v, egr_spotpy=None):
    obj_fc = [obj_func[:obj_func.index("_")] if "_" in obj_func else obj_func][0]
    sim_path = path + "simular\\{método}\\"
    save_plot =path + "plot\\{método}\\"
    guard = path + "npy_res\\{método}\\"

    if c_v == 'valid':

        d_val = {'dream': {'aic_rev': (500, 1000), 'nse_rev': (0, 500), 'rmse_rev': (0, 500)},
                 'mle': {'aic_rev': (0, 500), 'nse_rev': (500, 969), 'rmse_rev': (500, 1000)},
                 'fscabc': {'aic_rev': (144, 288), 'nse_rev': (0, 500), 'rmse_rev': (0, 500)},
                 'demcz': {'aic_rev': (0, 500), 'nse_rev': (500, 1000), 'rmse_rev': (1000, 1500)}}

        n_sim = d_val[método][obj_func]

        guardar = guard + f'valid_{obj_func}'
        valid_sim = sim_path + f'{obj_func}\\'
        save_plot = save_plot + f'{obj_func}\\'
        lg = np.load(guard + f'{método}_{obj_func}.npy').tolist()
        return tipo_proc, método, n_sim, obj_fc, guardar, valid_sim, save_plot, lg

    elif c_v == 'calib':
        guardar = guard + f'{método}_{obj_func}'
        guar_sim = sim_path + f'{obj_func}\\'
        if egr_spotpy is not None:
            egr_spotpy = sim_path + f'{método}_{obj_func}.csv'
        return tipo_proc, método, obj_fc, guardar, guar_sim, egr_spotpy


if __name__ == "__main__":
    for m in ['fscabc', 'demcz', 'mle','dream']: #'fscabc', 'demcz', 'mle','dream'
        # tipo_proc, método, obj_func, guardar, guar_sim, egr_spotpy = _input_dt(tipo_proc='patrón', método=m,
        #                                                                        obj_func='aic_rev',
        #                                                                        c_v='calib', egr_spotpy=True)
        # _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
        #        guar_sim=guar_sim, egr_spotpy=egr_spotpy)

        # tipo_proc, método, obj_func, guardar, guar_sim, egr_spotpy = _input_dt(tipo_proc='multidim', método=m,
        #                                                                        obj_func='nse_rev',
        #                                                                        c_v='calib', egr_spotpy=True)
        # _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
        #        guar_sim=guar_sim, egr_spotpy=egr_spotpy)

        # tipo_proc, método, obj_func, guardar, guar_sim, egr_spotpy = _input_dt(tipo_proc='multidim', método=m,
        #                                                                        obj_func='rmse_rev',
        #                                                                        c_v='calib', egr_spotpy=True)
        # _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
        #        guar_sim=guar_sim, egr_spotpy=egr_spotpy)
        tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
            _input_dt(tipo_proc='multidim', método=m, obj_func='nse_rev', c_v='valid')
        _valid(reverse=True, tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
               método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)

        tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
            _input_dt(tipo_proc='multidim', método=m, obj_func='rmse_rev', c_v='valid')
        _valid(reverse=True, tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
               método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)

        tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
            _input_dt(tipo_proc='patrón', método=m, obj_func='aic_rev', c_v='valid')
        _valid(reverse=True, tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
               método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)
