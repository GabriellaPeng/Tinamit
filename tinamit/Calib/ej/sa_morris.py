import numpy as np
import platform
from tinamit.Análisis.Sens.anlzr import analy_by_file, singular_behav_proc
from tinamit.Análisis.Sens.corridas import simul_sens
from tinamit.Calib.ej.info_paráms import mapa_paráms, líms_paráms
from tinamit.Calib.ej.muestrear import cargar_mstr_dt
from tinamit.Calib.ej.sens_análisis import gen_rank_map, gen_mod

if platform.release() == '7':
    guardar = 'D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\'
    mstr = guardar+'sampled_data\\muestra_morris_625.json'
    simular_path = guardar+'simular\\'
    direc = guardar + "simular\\625_mor\\"
    plot_path = guardar + "map\\"

else:
    guardar = "D:\Gaby\Tinamit\Dt\Mor\\"
    mstr = guardar+'sampled_data\\muestra_morris_625.json'
    direc = guardar+"simular\\625_mor\\"
    plot_path = guardar+"map\\"

gof_type = ['aic', 'bic', 'mic', 'srm', 'press', 'fpe']

if __name__ == "__main__":
    '''
    Simul
    '''
    # simul_sens(gen_mod(), mstr_paráms=cargar_mstr_dt(), mapa_paráms=mapa_paráms, var_egr='mds_Watertable depth Tinamit', t_final=20,
    #     guardar=simular_path, índices_mstrs=None, paralelo=True)

    '''
    Anlzr
    '''
    # egr = analy_by_file('morris', líms_paráms, mapa_paráms, mstr,
    #                     simul_arch={'arch_simular': direc, 'num_samples': 625}, tipo_egr='superposition',
    #                     var_egr='mds_Watertable depth Tinamit', gof_type=['aic', 'bic', 'mic', 'srm', 'press', 'fpe'], f_simul_arch=f"{guardar}aug\\",
    #                     plot_path=plot_path+'select_criteria\\')
    #
    #                     f_simul_arch= {'arch': "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\corrected_bf\\new_spp\\new_f_simul_sppf_simul",
    #                                    'num_sample': 625,
    #                                    'counted_behaviors':
    #                                        "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\corrected_bf\counted_all\\counted_all_behav.npy"})
    #
    # np.save(guardar+'2019', egr)

    # singular_behav_proc(simul_arch={'arch_simular': direc, 'num_samples': 625}, tipo_egr='superposition',
    #                     dim=215, var_egr='mds_Watertable depth Tinamit', guardar=f"{guardar}aug\\",
    #                     gof_type=['aic', 'bic', 'mic', 'srm', 'press', 'fpe'], plot_path=plot_path+'select_criteria\\')

    # for tipo in ['promedio']: #'promedio' paso_tiempo
    #     egr = analy_by_file('morris', líms_paráms, mapa_paráms, mstr, dim=215,
    #                         simul_arch={'arch_simular': direc, 'num_samples': 625}, tipo_egr=tipo,
    #                         var_egr='mds_Watertable depth Tinamit',
    #                         f_simul_arch= guardar + "f_simul\\f_simul")
    #
    #     np.save(guardar + f"{tipo}_egr", egr)



    '''
    post_processing Anlzr 
    '''

    # fited_behav = analy_behav_by_dims('morris', 625, 215, guardar+'f_simul\\aug\\', # dim_arch=ini,
    #                     gaurdar= guardar+ "fited_behav", gof_type=gof_type)

    # counted_all_behaviors = gen_counted_behavior(guardar+'fited_behav.npy', guardar, gof_type=gof_type)

    # for the immerged cosntant
    # egr = anlzr_simul('morris', líms_paráms,
    #                   "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\sampled_data\\muestra_morris_625.json",
    #                   mapa_paráms, dim=214, tipo_egr='superposition', var_egr='mds_Watertable depth Tinamit', ficticia=True,
    #                   f_simul_arch={
    #                       'arch': "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\f_simul",
    #                       'num_sample': 625,
    #                       'counted_behaviors':
    #                           "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\counted_all_behaviors_noini.npy"})
    #
    # np.save("D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\spp_no_ini_jan_7th_2", egr)

    # for the correct costant values
    # egr = anlzr_simul('morris', líms_paráms,
    #                   "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\sampled_data\\muestra_morris_625.json",
    #                   mapa_paráms, ficticia=True, var_egr='mds_Watertable depth Tinamit', f_simul_arch={
    #         'arch': "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\f_simul_cont",
    #         'num_sample': 625,
    #         'counted_behaviors': "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\count_all.npy"},
    #                   dim=215, tipo_egr="superposition", simulation=None, ops_método=None)
    # np.save("D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\mor_625_spp_const", egr)

    # merge_dict(merg1=behav_data_mor, merg2=behav_cont_data_mor, save_path=behav_const_dt + 'behav_correct_cont_dt')

    '''
    Maping
    '''
    # _gen_poly_dt_for_geog('morris', geog_simul_pct_mor2, geog_simul_pct_mor)
    # gen_geog_map(geog_save_mor, measure='behavior_param', method='Morris', param=None,
    #              fst_cut=0.1, snd_cut=8)

    # final plot
    load_data = {egr: np.load(guardar+f'{egr}.npy').tolist() for egr in ['paso_tiempo_egr', 'promedio_egr', 'behav_pattern_egr', 'fited_behav']}
    gen_rank_map(guardar+'map\\', 'morris', 0.1, 8, 'total_poly', load_data=load_data, cluster=True, cls=6, gof_type=['aic', 'mic'])

