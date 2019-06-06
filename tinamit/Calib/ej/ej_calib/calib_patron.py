from tinamit.Calib.ej.ej_calib.info_calib import *
from tinamit.Calib.ej.sens_análisis import gen_mod
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.info_paráms import calib_líms_paráms, calib_mapa_paráms
from tinamit.Calib.ej.cor_patrón import ori_valid, ori_calib

líms_paráms_final = []
líms_paráms_final.append(
    gen_problema(líms_paráms=calib_líms_paráms, mapa_paráms=calib_mapa_paráms, ficticia=False)[1])

mod = gen_mod()
method = ['dream', 'fscabc']


def _calib(tipo_proc, obj_func, guardar, método, guar_sim, reverse=False):
    if reverse:
        bd = ori_valid
    else:
        bd = ori_calib
    calib_res = mod.calibrar(paráms=list(líms_paráms_final), bd=bd, líms_paráms=calib_líms_paráms,
                             vars_obs='mds_Watertable depth Tinamit', final_líms_paráms=líms_paráms_final[0],
                             mapa_paráms=calib_mapa_paráms, tipo_proc=tipo_proc, obj_func=obj_func,
                             guardar=guardar, método=método, n_iter=500, guar_sim=guar_sim, egr_spotpy=False, warmup_period=2)
    return calib_res


def _valid(tipo_proc, guardar, valid_sim, n_sim, save_plot, lg, reverse=False, obj_func=None):
    if reverse:
        bd = ori_calib
    else:
        bd = ori_valid
    valid_res = mod.validar(bd=bd, var='mds_Watertable depth Tinamit', tipo_proc=tipo_proc,
                            obj_func=obj_func, guardar=guardar, lg=lg, paralelo=True, valid_sim=valid_sim,
                            n_sim=n_sim, save_plot=save_plot, warmup_period=2)
    return valid_res


if __name__ == "__main__":
    for m in method:
        # if m == 'fscabc':
        if m == 'dream':
            _calib(tipo_proc='patrón', obj_func='AIC', guardar=guardar_dream+'aic', método=m,
                   guar_sim=sim_dream)
        #
        #     _calib(reverse=True, tipo_proc='patrón', obj_func='AIC', guardar=guardar_abc+'aic_rev', método=m,
        #            guar_sim=sim_abc_rev)
        #
        #     _calib(tipo_proc='multidim', obj_func='NSE', guardar=guardar_abc+'nse', método=m,
        #            guar_sim=sim_abc_nse)
        #
        #     _calib(reverse=True, tipo_proc='multidim', obj_func='NSE', guardar=guardar_abc+'nse_rev', método=m,
        #            guar_sim=sim_abc_nse_rev)

            # valid_dream
            # aic
            # _valid(tipo_proc='CI', guardar=guardar_abc + 'valid_aic', valid_sim=sim_abc+'aic\\',
            #        n_sim=(0, 144), save_plot=plot_abc+'aic\\', lg=np.load(guardar_abc + "aic.npy").tolist()) #run 4 times

            # _valid(tipo_proc='patrón', guardar=guardar_abc + 'valid_aic', valid_sim=sim_abc,
            #        n_sim=(0, 144), save_plot=plot_abc+'aic\\', lg=np.load(guardar_abc + "aic.npy").tolist())
            # 0, 495; 495, 990, 990, 1485, 1485, 1979; reverse=True
            # 0, 144; 144, 288; 287, 912;  912, 1536; 'nse\\'
            # nse
            # _valid(tipo_proc='multidim', obj_func='NSE', guardar=guardar_abc+'valid_nse', valid_sim=sim_abc,
            #        n_sim=(0, 144), save_plot=plot_abc, lg=np.load(guardar_abc+"nse.npy").tolist())
            #
            # # _valid(tipo_proc='CI', guardar=guardar_abc+'valid_nse', valid_sim=sim_abc,
            #        n_sim=(0, 144), save_plot=plot_abc+'nse_rev', lg=np.load(guardar_abc+"nse.npy").tolist())