import numpy as np
import spotpy
from tinamit.Análisis.Sens.muestr import gen_problema
from tinamit.Calib.ej.info_paráms import calib_líms_paráms, calib_mapa_paráms

vr = 'mds_Watertable depth Tinamit'

import pandas as pd
import matplotlib.pyplot as plt


def read_obs_csv(file_name):
    res = pd.read_csv(file_name)

    obs_data = {}
    for row in res.values:
        # if np.isnan(row[1]):
        #     continue
        obs_data[row[1]] = row[2:]
    return res.columns[2:len(res.columns)], obs_data


def path_4_plot(plot_type, mtd, obj_func, p_m=None):
    res_path = '/Users/gabriellapeng/Downloads/'
    obs_param = np.load(res_path + 'obs_prm.npy').tolist()
    c_obs_dt = read_obs_csv('/Users/gabriellapeng/Downloads/calib/calib.csv')[1]
    v_obs_dt = read_obs_csv('/Users/gabriellapeng/Downloads/calib/valid.csv')[1]

    if plot_type == 'prm_prb':
        mtd_res = {}
        for m in mtd:
            mtd_res[m] = np.load(res_path + f'calib/{m}_{obj_func}.npy').tolist()

        if 'rev' in obj_func:
            poly = np.asarray(list(v_obs_dt))
        else:
            poly = np.asarray(list(c_obs_dt))

        return  poly, mtd_res, obj_func, obs_param, res_path+f'calib/plot/{obj_func}/'

    elif plot_type == 'valid':
        prob = { }
        for m in mtd:
            prob[m] = np.load(res_path + f'valid/{m}_valid_{obj_func}_{p_m}.npy').tolist()[vr][
                obj_func[:obj_func.index('_')].upper() if '_' in obj_func else obj_func.upper()]

        if 'rev' in obj_func:
            poly = np.asarray(list(c_obs_dt))
        else:
            poly = np.asarray(list(v_obs_dt))

        return poly, prob, obj_func, obs_param, res_path+f'valid/plot/{obj_func}'


def combine_calib_res(res, method, obj_func=None):
    '''
    :param res: [res1, res2]
    :param method: ['abc', 'mle']
    :return:
    '''
    for pm in ['POH Kharif Tinamit', 'POH rabi Tinamit', 'Capacity per tubewell']:
        for m in method:
            res[m]['sampled_prm'][pm] = np.asarray([j.tolist() for j in res[m]['sampled_prm'][pm]])
            res[m][pm] = np.asarray([j.tolist() for j in res[m][pm]])

    for m in method:
        res[m].update({p: np.asarray(res[m][p]) for p, v in res[m].items() if isinstance(v, list)})

    d_param = {m:{ } for m in method}

    for m in method:
        d_param[m] = {p: v for p,v in res[m].items() if len(v) == len(res[m]['buenas']) and p!='buenas'}

    if 'aic' in obj_func:
        prob = {m: np.negative(np.take(res[m]['prob'], res[m]['buenas'])) for m in method}
    else:
        prob = {m: np.take(res[m]['prob'], res[m]['buenas']) for m in method}

    return d_param, prob


def _soil_canal(all_poly_dt):
    cls = ['Buchanan, Head', 'Buchanan, Middle', 'Buchanan, Tail', 'Farida, Head', 'Farida, Middle', 'Farida, Tail',
           'Jhang, Middle', 'Jhang, Tail', 'Chuharkana, Tail']
    s_p = {c: [ ] for c in cls}

    for p in all_poly_dt:
        if p in (17, 52, 185):
            s_p[cls[0]].append(p)
        elif p in (36, 85, 132):
            s_p[cls[1]].append(p)
        elif p in (110, 125, 215):
            s_p[cls[2]].append(p)
        elif p in (7, 13, 76, 71):
            s_p[cls[3]].append(p)
        elif p in (25, 77, 123, 168,171):
            s_p[cls[4]].append(p)
        elif p in (54, 130, 172, 174, 178, 187, 191, 202, 205):
            s_p[cls[5]].append(p)
        elif p in (16, 22, 80, 94):
            s_p[cls[6]].append(p)
        elif p in (50, 121):
            s_p[cls[7]].append(p)
        elif p in (143, 164, 175, 203):
            s_p[cls[8]].append(p)
    return s_p


def plot_prm_prb(obj_func, mtd):
    '''

    :param obj_func: if _rev in obj_func then poly needs to be v_poly
    :param poly:
    :param mtd:
    :return:
    '''

    poly, mtd_res, obj_func, obs_param, save_plot = path_4_plot('prm_prb', mtd, obj_func)

    d_param, prob = combine_calib_res(mtd_res, mtd, obj_func)

    rev = [True if '_' in obj_func else False][0]

    ind_poly = np.asarray([p-1 for p in poly])
    s_cnl_msk = _soil_canal(poly)
    prb = [i for m, l in prob.items() for i in l]

    for p in obs_param:
        y = [d_param[m][p] for m in d_param if p in d_param[m] ]
        if len(y):
            for i, m in enumerate(prob):
                plt.ioff()
                plt.scatter(prob[m], y[i], alpha=0.5, marker='o', label=f'{m}')
            plt.hlines(obs_param[p], xmin=min(prb), xmax=max(prb), colors='g', label='Previously calibrated value')
            plt.xlabel(f"{obj_func.upper()}")
            plt.ylabel(f"{p}")
            plt.legend()
            plt.savefig(save_plot + f'{obj_func}_{p}')
            plt.close()

        else:
            s_cnl = { }
            for m in d_param:
                s_cnl[m]  = np.asarray([np.take(i, ind_poly) for pp in d_param[m] for i in d_param[m][pp] if pp.startswith(f'{p.capitalize()}')]).T

            for c, l_p in s_cnl_msk.items():
                for i, pl in enumerate(poly):
                    if pl in l_p:
                        sl_cl= {m: s_cnl[m][i] for m in s_cnl}
                        for i, m in enumerate(prob):
                            plt.ioff()
                            plt.scatter(prob[m], sl_cl[m], alpha=0.5, marker='o', label=f'{m}')
                        plt.xlabel(f"{obj_func.upper()}")
                        plt.ylabel(f"{p}")
                        plt.title(f"Poly{pl}, Condition:{c}")
                        if rev:
                            plt.hlines(obs_param[p]['rev'][i], xmin=min(prb), xmax=max(prb), colors='g', label='Previously calibrated value')
                        else:
                            plt.hlines(obs_param[p]['ori'][i], xmin=min(prb), xmax=max(prb), colors='g',label='Previously calibrated value')
                        plt.legend()
                        plt.savefig(save_plot+f'{pl}_{obj_func}_{p}')
                        plt.close()


def prb_cls(obj_func, mtd, p_m):
    poly, prob, obj_func, obs_param, save_plot = path_4_plot('valid', mtd, obj_func, p_m)

    s_cnl_msk = _soil_canal(poly)

    x=[i+1 for i in range(len(s_cnl_msk))]
    xlabels = [i[0]+i[i.index(',')+2] for i in s_cnl_msk]

    colors = ['c', 'y', 'm', 'b']
    m_y= {f'm{i}_y': [[] for j in range(len(s_cnl_msk))] for i, m in enumerate(prob)}

    for c, l_p in s_cnl_msk.items():
        for i, pl in enumerate(poly):
            if pl in l_p:
                for j, m in enumerate(prob):
                    m_y[f'm{j}_y'][list(s_cnl_msk).index(c)].append(prob[m][i])
    a=[ ]
    for i, y in enumerate(m_y):
        for xe, ye in zip(x, m_y[y]):
            plt.ioff()
            a.append(plt.scatter([xe] * len(ye), ye, alpha=0.3, marker='o', label=f'{list(prob)[i]}', color=f'{colors[i]}'))

    plt.xticks(x, xlabels, rotation=70, fontsize=10)
    plt.xlabel('Soil Class & Canal Position')
    plt.ylabel(f"{obj_func.upper()}")
    objfc  = [obj_func[:obj_func.index('_')].upper() if '_' in obj_func else obj_func.upper()]
    plt.title(f"{objfc[0]} to Validation Polygons")

    labl =[m.upper() for m in prob]
    ind = np.searchsorted([i for i, m1 in enumerate(prob) for m in [p.get_label() for p in a] if m1 == m],
                    [i for i in range(len(labl))], side='right')
    plt.legend([a[i-1] for i in ind], labl)
    plt.savefig(save_plot)
    plt.close()


# mtd1 =  ['fscabc', 'mle', 'dream']
# mtd2 =  ['fscabc', 'demcz', 'dream']
# mtd3 = [ 'fscabc', 'mle']
#
# valid_mtd = ['fscabc', 'demcz']
#
# obj_func = 'aic_rev'
# p_m = 'patrón'

# plot_prm_prb('aic_rev', mtd=['fscabc'])

prb_cls('aic_rev', ['fscabc'], 'patrón')


import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
from matplotlib.patches import Polygon

data = {'AIC':[], 'poly':[ ], 'canal_loc':[]}

random_dists = ['Normal(1,1)', ' Lognormal(1,1)', 'Exp(1)', 'Gumbel(6,4)',
                'Triangular(2,9,11)']

N = 500

norm = np.random.normal(1, 1, N)
logn = np.random.lognormal(1, 1, N)
expo = np.random.exponential(1, N)
gumb = np.random.gumbel(6, 4, N)
tria = np.random.triangular(2, 9, 11, N)

# Generate some random indices that we'll use to resample the original data
# arrays. For code brevity, just use the same random indices for each array
bootstrap_indices = np.random.randint(0, N, N)
data = [
    norm, norm[bootstrap_indices],
    logn, logn[bootstrap_indices],
    expo, expo[bootstrap_indices],
    gumb, gumb[bootstrap_indices],
    tria, tria[bootstrap_indices],
]

fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('A Boxplot Example')
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

xlim = [0, 10]
ylim = [-5, 40]

a = np.linspace(xlim[0]+0.5, xlim[1], 5, endpoint=False)
b = [[i, i+0.5] for i in a] #based on the poly_dist
c  = [j for i in b for j in i]

bp = ax1.boxplot(data, notch=0, sym='.', positions=np.asarray(c), whis=[5, 95])

# plt.xticks([np.mean(i) for i in b], [np.mean(i) for i in b])

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of IID Bootstrap Resampling Across Five Distributions')
ax1.set_xlabel('Distribution')
ax1.set_ylabel('Value')

import matplotlib.colors as mcolors
# Now fill the boxes with desired colors
box_colors = ['#F5B7B1','#00BCD4', '#FFE082', '#A5D6A7', '#B39DDB']
rbg = [mcolors.to_rgba(c) for c in box_colors]
num_boxes = len(data)
medians = np.empty(num_boxes)

color = [rbg[i_clr] for i_clr, j in enumerate(b) for v in c if v in j]
    # box = bp['boxes'][i]
    # boxX = []
    # boxY = []
    # for j in range(5): #range(9) 9 colors
    #     boxX.append(box.get_xdata()[j])
    #     boxY.append(box.get_ydata()[j])
    # box_coords = np.column_stack([boxX, boxY])

    # Alternate between Dark Khaki and Royal Blue
from matplotlib.patches import Polygon

for i, v in enumerate(c):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])

    box_coords = np.column_stack([boxX, boxY])

    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=color[i]))

    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        ax1.plot(medianX, medianY, 'r')
    medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*' ) #, markeredgecolor='k')

upper_labels = [str(np.round(s, 2)) for s in medians]

for tick, label in zip(range(num_boxes), ax1.get_xticks()):
    ax1.text(label, .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             color=color[tick])

# Set the axes ranges and axes labels
ax1.set_xlim(xlim[0], xlim[1])
ax1.set_ylim(ylim[0], ylim[1])

ax1.set_xticks([np.average(i) for i in b])
ax1.set_xticklabels(random_dists, rotation=45, fontsize=8)

# Finally, add a basic legend
# for i in np.flip(np.arange(0.01, 0.11, 0.01))[9:]:
pos = [i for i in np.flip(np.arange(0.01, 0.5, 0.03))][:len(random_dists)]
for i,v in enumerate(random_dists):
    fig.text(0.9, pos[i], f'{v}',
             backgroundcolor=rbg[i], color='white', weight='roman', size='x-small')

fig.text(0.90, 0.015, '*', color='white', backgroundcolor='black', weight='roman', size='large')
fig.text(0.915, 0.013, ' Average Value', color='black', weight='roman', size='x-small')


