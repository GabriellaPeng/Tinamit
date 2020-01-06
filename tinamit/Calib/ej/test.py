import numpy as np

algorithms = [ 'dream', 'fscabc', 'mle']
obj_func = ['mic', 'aic']

dict_stat = {m:{oj: { } for oj in obj_func}for m in algorithms}

for m in algorithms:
    for gof in obj_func:
        dict_stat[m][gof] = np.load(f"D:\Gaby\Dt\Calib\\real_run\{m}\dec\\calib_{gof}.npy").tolist()['prob']

print()