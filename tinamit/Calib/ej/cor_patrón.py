from tinamit.Calib.ej.obs_patrón import read_obs_csv, read_obs_data, plot_pattern


# calib = "D:\Thesis\data\\old\\calib.csv"
# valid = "D:\Thesis\data\\old\\valid.csv"

calib ='/Users/gabriellapeng/Downloads/calib/calib.csv'
valid ='/Users/gabriellapeng/Downloads/calib/valid.csv'


ori_calib = read_obs_csv(calib)
ori_valid = read_obs_csv(valid)

