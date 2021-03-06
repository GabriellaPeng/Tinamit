import numpy as np

# morris
mor_root = "D:\Gaby\Dt\Mor\\"
simu_guar_arch_mor = mor_root+"\\simular\\625_mor"
rank_arch_mor = mor_root+"\map\\" ## for rank
no_ini_mor = "D:\Gaby\Dt\Mor\\anlzr\\fited_behav_noini.npy"

paso_data_mor = np.load("D:\Gaby\Dt\Mor\\anlzr\\625\\mor_625_paso.npy").tolist()
mean_data_mor = np.load("D:\Gaby\Dt\Mor\\anlzr\\625\\mor_625_promedio.npy").tolist()

behav_correct_const_dt = np.load("D:\Gaby\Dt\Mor\\anlzr\\625\\mor_625_spp_no_ini.npy").tolist()
behav_data_mor = np.load("D:\Gaby\Dt\Mor\\anlzr\\625\\mor_625_spp_no_ini.npy").tolist()

geog_simul_pct_mor2 = "D:\Gaby\Dt\Mor\\f_simul\\fited_behav_dict_noini.npy"
geog_save_mor = "D:\Gaby\Dt\Mor\map\\"
geog_simul_pct_mor = "D:\Gaby\Dt\Mor\map\\f_simul\\"

behav_const_dt = "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\"

# fast
rank_arch_fast = "D:\Gaby\Dt\\fast\map\\final_plot\\"
no_ini_fast = "D:\Gaby\Dt\Fast\guardar_po_fsim\\fited_behav.npy"
paso_data_fast = np.load("D:\Gaby\Dt\Fast\\anlzr\egr_paso\\egr-0.npy").tolist()
paso_arch_fast = "D:\Gaby\Dt\Fast\\anlzr\egr_paso\\"

fit_beh_poly_fast = "D:\Gaby\Dt\Fast\\f_simul_post\\"

geog_simul_pct_fast = "D:\Gaby\Dt\Fast\map\\"

mean_data_fast = np.load("D:\Gaby\Dt\Fast\\anlzr\\egr_mean\\egr-0.npy").tolist()
mean_arch_fast = "D:\Gaby\Dt\Fast\\anlzr\\egr_mean\\"

behav_data_fast = np.load("D:\Gaby\Dt\Fast\\anlzr\\egr_behav\\egr-0.npy").tolist()
behav_arch_fast = "D:\Gaby\Dt\Fast\\anlzr\\egr_behav\\"

# morris
# simu_guar_arch_mor = "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\simular\\"
# rank_arch_mor = "D:\Thesis\pythonProject\localuse\Dt\Mor\map\\final_plot\\"
# no_ini_mor = "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fited_behav_noini.npy"
#
# paso_data_mor = np.load("D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\mor_625_paso.npy").tolist()
#
# mean_data_mor = np.load("D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\mor_625_promedio.npy").tolist()
#
# behav_data_mor = np.load(
#     "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\mor_625_spp_no_ini.npy").tolist()
#
# behav_cont_data_mor = np.load(
#     "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\mor_625_spp_const.npy").tolist()
#
# behav_correct_const_dt = np.load("D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\anlzr\\625\\behav_correct_cont_dt.npy").tolist()
#
# behav_const_dt = "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\f_simul_no_ini\\fsim_const\\"

# geog_save_mor = "D:\Thesis\pythonProject\localuse\Dt\Mor\map\\"
# geog_simul_pct_mor = "D:\Thesis\pythonProject\localuse\Dt\Mor\Mor_home\\f_simul\\"
#
# # fast
# no_ini_fast = "D:\Thesis\pythonProject\localuse\Dt\Fast\post_f_simul\\fited_behav.npy"
# paso_data_fast = np.load("D:\Thesis\pythonProject\localuse\Dt\Fast\\anlzr\egr_paso\\egr-0.npy").tolist()
# paso_arch_fast = "D:\Thesis\pythonProject\localuse\Dt\Fast\\anlzr\egr_paso\\"
#
# mean_data_fast = np.load("D:\Thesis\pythonProject\localuse\Dt\Fast\\anlzr\\egr_mean\\egr-0.npy").tolist()
# mean_arch_fast = "D:\Thesis\pythonProject\localuse\Dt\Fast\\anlzr\\egr_mean\\"
#
# behav_data_fast = np.load("D:\Thesis\pythonProject\localuse\Dt\Fast\\anlzr\\egr_behav\\egr-0.npy").tolist()
# behav_arch_fast = "D:\Thesis\pythonProject\localuse\Dt\Fast\\anlzr\\egr_behav\\"
#
# geog_save_fast = "D:\Thesis\pythonProject\localuse\Dt\Fast\map\\"
