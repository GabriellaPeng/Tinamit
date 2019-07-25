import numpy as np
from tinamit.Calib.ej.sens_análisis import gen_mod
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.info_paráms import calib_líms_paráms, calib_mapa_paráms
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid

líms_paráms_final = []
líms_paráms_final.append(gen_problema(líms_paráms=calib_líms_paráms, mapa_paráms=calib_mapa_paráms, ficticia=False)[1])

mod = gen_mod()


def _calib(tipo_proc, guardar, método, guar_sim, obj_func, cls=False, reverse=False):
    if reverse:
        bd = ori_valid
    else:
        bd = ori_calib
    calib_res = mod.calibrar(paráms=list(líms_paráms_final), bd=bd, líms_paráms=calib_líms_paráms,
                             vars_obs='mds_Watertable depth Tinamit', final_líms_paráms=líms_paráms_final[0],
                             mapa_paráms=calib_mapa_paráms, tipo_proc=tipo_proc, guardar=guardar, método=método,
                             n_iter=500,
                             guar_sim=guar_sim, warmup_period=2, cls=cls, obj_func=obj_func,
                             egr_spotpy=None)
    return calib_res


def _valid(tipo_proc, guardar, valid_sim, n_sim, save_plot, lg, obj_func, método, reverse=False):
    if reverse:
        bd = ori_calib
    else:
        bd = ori_valid
    valid_res = mod.validar(bd=bd, var='mds_Watertable depth Tinamit', tipo_proc=tipo_proc,
                            guardar=guardar, lg=lg, paralelo=True, valid_sim=valid_sim,
                            n_sim=n_sim, save_plot=save_plot, warmup_period=2, obj_func=obj_func, método=método)
    return valid_res


def _input_dt(tipo_proc, método, obj_func, c_v, n_sim=None):
    obj_fc = [obj_func[:obj_func.index("_")] if "_" in obj_func else obj_func][0]
    sim_path = f"D:\Gaby\Tinamit\\Dt\Calib\\simular\\new\\{método}\\"
    save_plot = f"D:\Gaby\Tinamit\Dt\Calib\plot\\new\\{método}\\"
    guard = f"D:\Gaby\Tinamit\Dt\Calib\\real_run\\{método}\\"

    if c_v == 'valid':
        guardar = guard + f'valid_{obj_func}'
        valid_sim = sim_path + f'{obj_func}\\'
        save_plot = save_plot + f'{obj_func}\\'
        lg = np.load(guard + f'{método}_{obj_func}.npy').tolist()
        return tipo_proc, método, n_sim, obj_fc, guardar, valid_sim, save_plot, lg

    elif c_v == 'calib':
        guardar = guard + f'{obj_func}'
        guar_sim = sim_path + f'{obj_func}\\'
        return tipo_proc, método, obj_fc, guardar, guar_sim


if __name__ == "__main__":
    # tipo_proc, método, obj_func, guardar, guar_sim = _input_dt(tipo_proc='multidim', método='fscabc', obj_func='rmse_rev',
    #                                                            c_v='calib')
    # _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
    #        guar_sim=guar_sim)

    tipo_proc, método, obj_func, guardar, guar_sim = _input_dt(tipo_proc='multidim', método='mle', obj_func='nse_rev',
                                                               c_v='calib')
    _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
           guar_sim=guar_sim)

    tipo_proc, método, obj_func, guardar, guar_sim = _input_dt(tipo_proc='patrón', método='demcz', obj_func='aic_rev',
                                                               c_v='calib')
    _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
           guar_sim=guar_sim)

    tipo_proc, método, obj_func, guardar, guar_sim = _input_dt(tipo_proc='multidim', método='demcz', obj_func='nse_rev',
                                                               c_v='calib')
    _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
           guar_sim=guar_sim)

    tipo_proc, método, obj_func, guardar, guar_sim = _input_dt(tipo_proc='multidim', método='demcz', obj_func='rmse_rev',
                                                               c_v='calib')
    _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
           guar_sim=guar_sim)

    tipo_proc, método, obj_func, guardar, guar_sim = _input_dt(tipo_proc='multidim', método='dream',
                                                               obj_func='rmse_rev', c_v='calib')
    _calib(reverse=True, tipo_proc=tipo_proc, guardar=guardar, método=método, obj_func=obj_func,
           guar_sim=guar_sim)

    # _calib(reverse=True, tipo_proc='patrón', guardar=guardar_abc + 'aic_rev', método='fscabc', obj_func='aic',
    #        guar_sim=sim_abc + 'aic_rev\\')
    #
    # _calib(reverse=True, tipo_proc='multidim', guardar=guardar_abc + 'rmse', método='fscabc', obj_func='rmse',
    #        guar_sim=sim_abc + 'rmse\\')

    # tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
    #     _input_dt(tipo_proc='patrón', método='dream', n_sim=(0, 500), obj_func='aic', c_v ='valid')
    #
    # _valid(tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
    #        método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)
    #
    # tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
    #     _input_dt(tipo_proc='patrón', método='dream', n_sim=(500, 1000), obj_func='aic_rev', c_v='valid')
    #
    # _valid(reverse=True, tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
    #        método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)

    # #### multidim
    # tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
    #     _input_dt(tipo_proc='multidim', método='mle', n_sim=(0, 500), obj_func='nse', c_v='valid')
    #
    # _valid(tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
    #        método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)

    # tipo_proc, método, n_sim, obj_func, guardar, valid_sim, save_plot, lg = \
    #     _input_dt(tipo_proc='multidim', método='mle', n_sim=(500, 1000), obj_func='rmse_rev', c_v='valid')
    #
    # _valid(reverse=True, tipo_proc=tipo_proc, guardar=guardar, valid_sim=valid_sim, obj_func=obj_func,
    #        método=método, n_sim=n_sim, save_plot=save_plot, lg=lg)
