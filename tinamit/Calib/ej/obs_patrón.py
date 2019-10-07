## obeserved data detection ##
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tinamit.Análisis.Sens.behavior import superposition, find_best_behavior, predict, forma, simple_shape


def read_obs_csv(file_name):
    res = pd.read_csv(file_name)

    obs_data = {}
    for row in res.values:
        # if np.isnan(row[1]):
        #     continue
        obs_data[row[1]] = row[2:]
    return res.columns[2:len(res.columns)], obs_data


def read_obs_data(file_name, sheet_name=None):
    res = pd.read_excel(file_name, sheet_name=sheet_name)

    obs_data = {}
    for row in res.values:
        obs_data[row[1]] = row[2:]
    return res.columns[2:len(res.columns)], obs_data


def split_obs_data(obs_data):
    split_data = {}
    for key, val in obs_data.items():
        split_data[key] = [(x, y) for x, y in enumerate(val) if y != "None" and not np.isnan(y)]
    return split_data


def interpolate_data(points, kind, length):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    f = interp1d(x, y, kind=kind)(range(length))
    interpolated_data = []
    for i in range(length):
        if i in x:
            interpolated_data.append((i, y[x.index(i)]))
        else:
            interpolated_data.append((i, f[i]))

    return interpolated_data


def interp_all_data(split_data, kind, length):
    int_all_data = {}
    for key, val in split_data.items():
        interpolated_data = interpolate_data(val, kind, length)
        int_all_data[key] = [point[1] for point in interpolated_data]

    return int_all_data


def write_excel(data, columns, file):
    with pd.ExcelWriter(file,
                        engine='xlsxwriter') as writer:
        for kind, val in data.items():
            df = pd.DataFrame(data=list(val.values()), columns=columns)
            df.to_excel(writer, kind)


def compute_patron(npoly, norm_obs=None, valid=False, obj_func='aic', tipo_proc='patrón'):
    best_behaviors = {}
    linear = { }
    all_bbehav_params = { }
    if isinstance(norm_obs, np.ndarray):
        data = norm_obs
        for i, p in enumerate(npoly):
            print(f"Behavior Detecting of Polygon {p} !")
            if valid and tipo_proc=='multidim':
                linear[p] = simple_shape(np.arange(1, len(data[:, i])+1),data[:, i], 'linear', gof=False)
            elif valid and tipo_proc=='patrón':
                re = superposition(np.arange(1, len(data[:, i])+1), data[:, i], gof_type=[obj_func])[0]
                best_behaviors[p] = find_best_behavior(re, gof_type=[obj_func])[0][obj_func][0] # TODO 'aic or not
                all_bbehav_params[p] = re[best_behaviors[p]]
                linear[p] = re['linear']

        if valid:
            if tipo_proc=='patrón':
                return best_behaviors, linear, all_bbehav_params
            elif tipo_proc=='multidim':
                return None, linear, None
        else:
            return best_behaviors

    else:
        d_calib = {}
        d_numero = {}
        for poly, data in npoly.items():
            data = np.asarray([i for i in data])
            d_numero[poly] = np.asarray(data)
            print(f"Behavior Detecting of Polygon {poly} !")
            re = superposition(np.asarray(1, len(data)+1), data, gof_type=[obj_func])[0]
            best_behav = find_best_behavior(re, gof_type=[obj_func])[0] #TODO
            best_behaviors[poly] = best_behav #TODO
            y_pred = np.asarray(
                predict(np.array(1, len(data)+1), re[best_behav[obj_func][0]]['bp_params'], best_behav[0][0])) #TODO
            d_calib[poly] = {best_behav[obj_func][0]: re[best_behav[obj_func][0]], 'y_pred': y_pred} #TODO
        return best_behaviors, d_calib, d_numero


def plot_pattern(interploated_data, path):
    fited_behaviors = {poly: [] for poly in interploated_data}
    for poly, data in interploated_data.items():
        data = np.asarray([i for i in data])
        print(f"Polygon {poly} is under processing!")
        plt.plot(data)
        re = superposition(range(len(data)), data)[0]
        gof_dict = find_best_behavior(re)[1] #TODO
        fited_behaviors[poly].append(gof_dict[0])
        m = 1
        while m < len(gof_dict):
            if gof_dict[m][1] - gof_dict[0][1] > 10:
                break
            else:
                fited_behaviors[poly].append(gof_dict[m])
            m += 1
        for tup_patt in fited_behaviors[poly]:
            plt.plot(predict(range(len(data)), re[tup_patt[0]]['bp_params'], tup_patt[0]))
        plt.savefig(path + f'{poly}-{fited_behaviors[poly][0][0]}')
        plt.close()

    return fited_behaviors


def plot_obs_best_fit(res, gaur_arch, inplt_arch=None):
    if inplt_arch is not None:
        split_data = split_obs_data(res[1])
        kinds = ["previous", "nearest", "next"]
        kinds_in_d = {}
        for kind in kinds:
            kinds_in_d[kind] = interp_all_data(split_data, kind, 41)
        print(kinds_in_d['previous'])
        write_excel(kinds_in_d, list(res[0]), inplt_arch)
        bd = kinds_in_d['previous']
    else:
        bd = res[1]
    best_behaviors, d_calib, d_numero = compute_patron(bd) #TODO: wrong code！

    np.save("D:\Thesis\pythonProject\localuse\Dt\Calib\\best_behaviors", best_behaviors)
    np.save("D:\Thesis\pythonProject\localuse\Dt\Calib\\d_calib", d_calib)
    fited_behaviors = plot_pattern(bd, path=gaur_arch)

    print(fited_behaviors)