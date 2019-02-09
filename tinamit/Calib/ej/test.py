from tinamit.Análisis.Sens.anlzr import anlzr_simul
from tinamit.Análisis.Sens.corridas import gen_vals_inic
from tinamit.Calib.ej.sens_análisis import *
from tinamit.Calib.ej.info_paráms import *
from tinamit.Calib.ej.info_analr import *
from tinamit.Análisis.Sens.muestr import gen_problema, muestrear_paráms


# def process_constant(data):
#     data['bp_params']['constant'] = np.add(data['bp_params']['constant'], data['bp_params']['constant_1'])
#     del data['bp_params']['constant_1']
#     return data
#
#
# for i in range(625):
#     complete_behav = np.load(
#         f"D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\f_simul_{i}.npy").tolist()
#     b_param = {}
#     b_param.update({'spp_oscil_aten_inverso': process_constant(complete_behav['spp_oscil_aten_inverso'])})
#     b_param.update({'spp_oscil_aten_log': process_constant(complete_behav['spp_oscil_aten_log'])})
#     print(f"finished {i}-th sample")
#     np.save(f"D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\f_simul_cont_{i}", b_param)

# egr = anlzr_simul('morris', líms_paráms,
#                   "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\sampled_data\\muestra_morris_625_corrected_bf.json",
#                   mapa_paráms, ficticia=True, var_egr='mds_Watertable depth Tinamit', f_simul_arch={
#         'arch': "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\f_simul_cont",
#         'num_sample': 625,
#         'counted_behaviors': "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\count_all.npy"},
#                   dim=215, tipo_egr="superposition", simulation=None, ops_método=None)
# np.save("D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\mor_625_spp_const", egr)
