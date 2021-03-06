import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from tinamit.Análisis.Sens.anlzr import carg_simul_dt
from tinamit.Análisis.Sens.behavior import find_best_behavior, predict, simple_shape, compute_rmse
from tinamit.Calib.ej.cor_patrón import ori_calib, ori_valid
from tinamit.Geog.Geog import _gen_clrbar_dic, _gen_d_mapacolores
from tinamit.Calib.ej.soil_class import p_soil_class
from tinamit.Calib.ej.info_paráms import mapa_paráms
from tinamit.Geog.Geog import Geografía
from tinamit.Conectado import Conectado
from tinamit.Ejemplos.en.Ejemplo_SAHYSMOD.SAHYSMOD import Envoltura

import multiprocessing as mp


def gen_mod():
    # Create a coupled model instance
    modelo = Conectado()

    # Establish SDM and Biofisical model paths. The Biofisical model path must point to the Python wrapper for the model
    modelo.estab_mds('D:\Gaby\Tinamit\\tinamit\Ejemplos\en\Ejemplo_SAHYSMOD\\Vensim\\Tinamit_Rechna.vpm')

    modelo.estab_bf(Envoltura)
    modelo.estab_conv_tiempo(mod_base='mds', conv=6)

    # Couple models(Change variable names as needed)
    modelo.conectar(var_mds='Soil salinity Tinamit CropA', mds_fuente=False, var_bf="CrA - Root zone salinity crop A")
    modelo.conectar(var_mds='Soil salinity Tinamit CropB', mds_fuente=False, var_bf="CrB - Root zone salinity crop B")
    modelo.conectar(var_mds='Area fraction Tinamit CropA', mds_fuente=False,
                    var_bf="Area A - Seasonal fraction area crop A")
    modelo.conectar(var_mds='Area fraction Tinamit CropB', mds_fuente=False,
                    var_bf="Area B - Seasonal fraction area crop B")
    modelo.conectar(var_mds='Watertable depth Tinamit', mds_fuente=False, var_bf="Dw - Groundwater depth")
    modelo.conectar(var_mds='ECdw Tinamit', mds_fuente=False, var_bf='Cqf - Aquifer salinity')  ###
    modelo.conectar(var_mds='Final Rainfall', mds_fuente=True, var_bf='Pp - Rainfall')  # True-coming from Vensim
    modelo.conectar(var_mds='Lc', mds_fuente=True, var_bf='Lc - Canal percolation')
    modelo.conectar(var_mds='Ia CropA', mds_fuente=True, var_bf='IaA - Crop A field irrigation')
    modelo.conectar(var_mds='Ia CropB', mds_fuente=True, var_bf='IaB - Crop B field irrigation')
    modelo.conectar(var_mds='Gw', mds_fuente=True, var_bf='Gw - Groundwater extraction')
    modelo.conectar(var_mds='EpA', mds_fuente=True, var_bf='EpA - Potential ET crop A')
    modelo.conectar(var_mds='EpB', mds_fuente=True, var_bf='EpB - Potential ET crop B')
    modelo.conectar(var_mds='Irrigation efficiency', mds_fuente=True, var_bf='FsA - Water storage efficiency crop A')
    modelo.conectar(var_mds='Fw', mds_fuente=True, var_bf='Fw - Fraction well water to irrigation')
    # 'Policy RH' = 1, Fw = 1, Policy Irrigation improvement = 1, Policy Canal lining=1, Capacity per tubewell =(100.8, 201.6),

    modelo.estab_conv_unid_tiempo('mes', 6)

    return modelo


def gen_geog():
    Rechna_Doab = Geografía(nombre='Rechna Doab')

    # base_dir = os.path.join("D:\Thesis\pythonProject\Tinamit\\tinamit\Ejemplos\en\Ejemplo_SAHYSMOD", 'Shape_files')
    base_dir = os.path.join("D:\Gaby\Tinamit\\tinamit\Ejemplos\en\Ejemplo_SAHYSMOD", 'Shape_files')
    Rechna_Doab.agregar_frm_regiones(os.path.join(base_dir, 'Internal_Polygon.shp'), col_id="Polygon_ID")

    Rechna_Doab.agregar_forma(os.path.join(base_dir, 'External_Polygon.shp'), color='#edf4da')
    Rechna_Doab.agregar_forma(os.path.join(base_dir, 'RIVR.shp'), tipo='agua')
    # Rechna_Doab.agregar_forma(os.path.join(base_dir, 'Forst_polygon.shp'), tipo='bosque')
    Rechna_Doab.agregar_forma(os.path.join(base_dir, 'CNL_Arc.shp'), tipo='agua', color='#1ba4c6', llenar=False)
    # Rechna_Doab.agregar_forma(os.path.join(base_dir, 'buildup_Polygon.shp'), tipo='ciudad')
    # Rechna_Doab.agregar_forma(os.path.join(base_dir, 'road.shp'), tipo='calle')

    return Rechna_Doab


devolver = ['Watertable depth Tinamit', 'Soil salinity Tinamit CropA']


def _gen_poly_dt_for_geog(method, d_fit_behav_arch, save_arch, num_sam):
    if method == 'morris':
        d_fit_behav = np.load(d_fit_behav_arch).tolist()

        fit_behav2 = {poly: {sam: v[0][0] for sam, v in samp.items()} for poly, samp in d_fit_behav.items()}
        fit_behav = {poly: [] for poly in fit_behav2}
        for poly, d_samp in fit_behav2.items():
            fit_behav[poly].append(list(d_samp.values()))
        fit_behav = {p: v[0] for p, v in fit_behav.items()}

        counted_all = []
        counted_all2 = {}
        for i, lst in fit_behav.items():
            counted_all.extend(list(set(lst)))
            counted_all2[i] = Counter(lst)
        counted_all = set(counted_all)

        counted_all2 = set(
            [patt for i in range(num_sam) for patt, ct in counted_all2[i].items() if ct / num_sam > 0.1])
        d_patt = {patt: {poly: 0 for poly in range(215)} for patt in counted_all}
        for patt in counted_all:
            for p in range(215):
                d_patt[patt][p] = Counter(fit_behav[p])[patt] / num_sam

        patt_sens_simul = {patt: np.empty([215]) for patt in counted_all}
        for patt, d_pcent in d_patt.items():
            for p in range(215):
                patt_sens_simul[patt][p] = d_pcent[p]

    elif method == 'fast':
        counted_all = []
        fit_behav = {p: {} for p in range(215)}
        for i in range(215):
            print(f'Processing poly {i}')
            poly_behav = np.load(d_fit_behav_arch + f'fit_beh_poly-{i}.npy').tolist()[i]
            for patt, ct in Counter([d_lt[0][0] for d_lt in poly_behav.values()]).items():
                counted_all.append(patt)
                fit_behav[i][patt] = ct / num_sam

        patt_sens_simul = {patt: np.empty([215]) for patt in counted_all}
        for p, d_pcent in fit_behav.items():
            for patt, pct in d_pcent.items():
                patt_sens_simul[patt][p] = pct

    np.save(save_arch + f'patt_sens_simul', patt_sens_simul)


def _read_dt_4_map(method, si=None, load_data=None, egr_path=None):
    paso_data = load_data['paso_tiempo_egr']
    mean_data = load_data['promedio_egr']
    behav_data = load_data['behav_pattern_egr']

    if method == 'morris':
        pasos = \
            verif_sens('morris', 'paso_tiempo', mapa_paráms, p_soil_class, egr=paso_data, si='mu_star')['morris'][
                'paso_tiempo']['mds_Watertable depth Tinamit']

        means = \
            verif_sens('morris', list(mean_data.keys())[0], mapa_paráms, p_soil_class, egr=mean_data,
                       si='mu_star')[
                'morris'][
                list(mean_data.keys())[0]]['mds_Watertable depth Tinamit']

        behaviors = verif_sens('morris', list(behav_data.keys())[0], mapa_paráms, p_soil_class,
                               egr=behav_data, si='mu_star')['morris'][list(behav_data.keys())[0]][
            'mds_Watertable depth Tinamit']

        ps = [0, 5, 10, 15, 20]

        return {'pasos': pasos, 'means': means, 'behaviors': behaviors, 'no_ini': load_data['fited_behav'], 'ps': ps}

    else:
        paso_arch = egr_path['paso_arch']
        mean_arch = egr_path['mean_arch']
        behav_arch = egr_path['behav_arch']

        if si is None:
            si = 'Si'
        pasos = \
            verif_sens('fast', list(paso_data.keys())[0], mapa_paráms, p_soil_class, egr_arch=paso_arch,
                       si=si,
                       dim=215)[
                'fast'][list(paso_data.keys())[0]]['mds_Watertable depth Tinamit']  # 9prms * 215polys
        for param, d_paso in pasos.items():
            for key in d_paso:
                d_paso[key] = np.asarray([0 if np.isnan(val) else val for val in d_paso[key]])

        means = \
            verif_sens('fast', list(mean_data.keys())[0], mapa_paráms, p_soil_class, egr_arch=mean_arch,
                       si=si,
                       dim=215)[
                'fast'][list(mean_data.keys())[0]]['mds_Watertable depth Tinamit']
        for param in means:
            means[param] = np.asarray([0 if np.isnan(v) else v for v in means[param]])

        behaviors = \
            verif_sens('fast', list(behav_data.keys())[0], mapa_paráms, p_soil_class, egr_arch=behav_arch,
                       si=si,
                       dim=215)['fast'][list(behav_data.keys())[0]]['mds_Watertable depth Tinamit']

        ps = [0, 1, 2, 3, 4]

        return {'pasos': pasos, 'means': means, 'behaviors': behaviors, 'no_ini': behav_data, 'ps': ps}


def _integrate_egr(egr_arch, dim, si, mapa_paráms, tipo_egr):
    def _gen_egr_tmp(tipo_egr, egr, si, egr_tmp=None, dim=None):
        if egr_tmp is None:
            for egr_var, patt in egr[tipo_egr].items():
                if tipo_egr == 'paso_tiempo':
                    for para, val in patt[si].items():
                        for paso in val:
                            if para in mapa_paráms:
                                val[paso] = np.empty([4, dim])
                            else:
                                val[paso] = np.empty([dim])
                elif tipo_egr == "promedio":
                    for para in patt[si]:
                        if para in mapa_paráms:
                            patt[si][para] = np.empty([4, dim])
                        else:
                            patt[si][para] = np.empty([dim])
                else:
                    for p_name, bg_name in patt.items():
                        for b_g, d_si in bg_name.items():
                            for para_name, d_bp_gof in d_si[si].items():
                                for bp_gof_name in d_bp_gof:
                                    if para_name in mapa_paráms:
                                        d_bp_gof[bp_gof_name] = np.empty([4, dim])
                                    else:
                                        d_bp_gof[bp_gof_name] = np.empty([dim])
        else:
            for egr_var, patt in egr[tipo_egr].items():
                if tipo_egr == 'paso_tiempo':
                    for para, val in patt[si].items():
                        for paso in val:
                            if para in mapa_paráms:
                                egr_tmp[tipo_egr][egr_var][si][para][paso][:, dim] = val[paso][:, 0]
                            else:
                                egr_tmp[tipo_egr][egr_var][si][para][paso][dim] = val[paso]

                elif tipo_egr == "promedio":
                    for para in patt[si]:
                        if para in mapa_paráms:
                            egr_tmp[tipo_egr][egr_var][si][para][:, dim] = patt[si][para][:, 0]
                        else:
                            egr_tmp[tipo_egr][egr_var][si][para][dim] = patt[si][para]

                else:
                    for p_name, bg_name in patt.items():
                        for b_g, d_si in bg_name.items():
                            for para_name, d_bp_gof in d_si[si].items():
                                for bp_gof_name in d_bp_gof:
                                    if para_name in mapa_paráms:
                                        egr_tmp[tipo_egr][egr_var][p_name][b_g][si][para_name][bp_gof_name][:, dim] = \
                                            d_bp_gof[bp_gof_name][:, 0]
                                    else:
                                        egr_tmp[tipo_egr][egr_var][p_name][b_g][si][para_name][bp_gof_name][dim] = \
                                            d_bp_gof[
                                                bp_gof_name]

    egr_tmp = {}
    egr0 = np.load(egr_arch + f'egr-{0}.npy').tolist()
    _gen_egr_tmp(tipo_egr=tipo_egr, egr=egr0, si=si, dim=dim)
    egr_tmp.update(egr0)

    for i in range(dim):
        _gen_egr_tmp(tipo_egr=tipo_egr, egr=np.load(egr_arch + f'egr-{i}.npy').tolist(), si='Si', egr_tmp=egr_tmp,
                     dim=i)

    return egr_tmp


def _single_poly(samples, i, f_simul_arch, gaurdar):
    fited_behav = {i: {j: {} for j in range(samples)}}
    for j in range(samples):
        print(f'this is {j}-th sample')
        behav = np.load(f_simul_arch + f"f_simul_{j}.npy").tolist()
        fited_behav[i][j] = find_best_behavior(behav, trans_shape=i)[0] #TODO
    if gaurdar is not None:
        np.save(gaurdar + f'fit_beh_poly-{i}', fited_behav)


def _compute_single(i, method, dim_arch, samples, f_simul_arch):
    count_fited_behav_by_poly = []
    if method == 'morris':
        polynal_fited_behav = np.load(dim_arch).tolist()[i]
    elif method == 'fast':
        polynal_fited_behav = np.load(dim_arch + f'fit_beh_poly-{i}.npy').tolist()[i]

    for j in range(samples):
        count_fited_behav_by_poly.extend([key for key, val in polynal_fited_behav[j]])

    count_fited_behav_by_poly = [k for k, v in Counter(count_fited_behav_by_poly).items() if
                                 v / len(count_fited_behav_by_poly) > 0.1]
    # fited_behav.update({i: count_fited_behav_by_poly})

    aic_poly = {k: [] for k in count_fited_behav_by_poly}
    for j in range(samples):
        behav = np.load(f_simul_arch + f"f_simul_{j}.npy").tolist()
        for k in count_fited_behav_by_poly:
            aic_poly[k].append(behav[k]['gof']['aic'][0, i])
        print(f'Processing sample-{j} for poly-{i}')
    # aic_behav.update({i: {k: np.average(v) for k, v in aic_poly.items()}})
    return i, count_fited_behav_by_poly, {k: np.average(v) for k, v in aic_poly.items()}


def merge_dict(method, merg1, merg2, save_path):
    if method == 'morris':
        merg1['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_inverso'] = \
            merg2['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_inverso']

        merg1['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_log'] = \
            merg2['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_log']

        np.save(save_path, merg1)
    elif method == 'fast':
        for i in range(215):
            m1 = np.load(merg1 + f'egr-{i}').tolist()
            m2 = np.load(merg2 + f'fa_cont_egr-{i}').tolist()

            m1['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_inverso'] = \
                m2['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_inverso']

            m1['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_log'] = \
                m2['superposition']['mds_Watertable depth Tinamit']['spp_oscil_aten_log']

            np.save(save_path + f'fa_egr_comp-{i}', m1)


def verif_sens(método, tipo_egr, mapa_paráms, p_soil_class, si, dim=None, egr=None, egr_arch=None):
    if método == 'fast':
        egr = _integrate_egr(egr_arch, dim, si, mapa_paráms, tipo_egr)

    if tipo_egr == "forma" or tipo_egr == 'superposition':
        final_sens = {método: {tipo_egr: {p_name: {gof: {bp_gof: {bp: {para: {f_name: np.asarray([f_val[id][i]
                                                                                for i, id in enumerate(p_soil_class)])
                                                                            for f_name, f_val in val.items()} if para in mapa_paráms else val
                                                                       for para, val in d_gof[si].items()}
                                                                     for bp, d_gof in bp_gof_val.items()}
                                                            for bp_gof, bp_gof_val in d_patt.items()}
                                                   for gof, d_patt in gof_name.items()}
                                          for p_name, gof_name in egr[tipo_egr].items()}}}


    elif tipo_egr == "paso_tiempo":
        final_sens = {método: {tipo_egr: {p_name: {para: {paso: np.asarray([paso_val[id][i]
                                                                            for i, id in enumerate(p_soil_class)])
                                                          for paso, paso_val in
                                                          val.items()} if para in mapa_paráms else val
                                                   for para, val in p_val[si].items()}
                                          for p_name, p_val in egr[tipo_egr].items()}}}
    elif tipo_egr == "promedio":
        final_sens = {método: {tipo_egr: {p_name: {para: np.asarray([val[id][i]
                                                                     for i, id in enumerate(
                p_soil_class)]) if para in mapa_paráms else val
                                                   for para, val in p_val[si].items()}
                                          for p_name, p_val in egr[tipo_egr].items()}}}
    else:
        raise Exception('The type is not defined !')

    return final_sens


def analy_behav_by_dims(method, samples, dims, f_simul_arch, dim_arch=None, gaurdar=None, gof_type=['aic']):
    if dim_arch is None:
        if method == 'morris':
            fited_behav = {i: {j: {} for j in range(samples)} for i in range(dims)}
            for j in range(samples):
                print(f'this is {j}-th sample')
                behav = np.load(f_simul_arch + f"f_simul_{j}.npy").tolist()
                for i in range(dims):
                    fited_behav[i][j] = find_best_behavior(behav, trans_shape=i, gof_type=gof_type)[0] #TODO
                    print(f'processing {i} poly')
            if gaurdar is not None:
                np.save(gaurdar + 'fited_behav', fited_behav)

        elif method == 'fast':
            pool = mp.Pool(processes=10)

            results = []
            for i in range(dims):
                # if i not in [34, 144, 176, 186]:
                #     continue
                print(f'Processing {i} poly')
                results.append(pool.apply_async(_single_poly, args=(samples, i, f_simul_arch, gaurdar,)))

            [result.wait() for result in results]
            # output = [p.get() for p in results]
            print("finished")


    else:
        fited_behav = {}
        aic_behav = {}
        results = []
        pool = mp.Pool(processes=10)
        for i in range(dims):
            results.append(pool.apply_async(_compute_single, args=(i, method, dim_arch, samples, f_simul_arch,)))

        for result in results:
            re = result.get()
            fited_behav.update({re[0]: re[1]})
            aic_behav.update({re[0]: re[2]})

        if gaurdar is not None:
            np.save(gaurdar + 'fited_behav', fited_behav)
            np.save(gaurdar + 'aic_behav', aic_behav)


def clustering(points, n_cls, valid=False):
    kmeans = KMeans(n_clusters=n_cls)
    # fit kmeans object to data
    kmeans.fit(points)
    # print location of clusters learned by kmeans object
    location_clusters = kmeans.cluster_centers_
    # save new clusters for chart
    y_km = kmeans.fit_predict(points)
    km_cls = np.empty([points.shape[0], points.shape[1]])
    d_cls = {cls: len(np.where(y_km == cls)[0]) for cls in range(n_cls)}
    d_km = {}
    km_lst = []
    ct = 0
    for cls in range(n_cls):
        km_lst.extend(np.where(y_km == cls)[0])
        d_km[cls] = np.where(y_km == cls)[0]
        for j in range(d_cls[cls]):
            km_cls[j + ct, :] = points[np.where(y_km == cls)[0][j], :]
        ct += d_cls[cls]
    km_lst = np.asarray(km_lst)

    if valid is False:
        # create dendrogram
        new_order = sch.dendrogram(sch.linkage(points, method='ward'))['leaves']
        n_points = np.empty([points.shape[0], points.shape[1]])
        for i in range(points.shape[0]):
            n_points[i, :] = points[new_order[i], :]

        return {'n_points': n_points, 'new_order': new_order, 'km_lst': km_lst, 'km_cls': km_cls, 'y_km': y_km,
                'd_km': d_km}
    else:
        return {'km_lst': km_lst, 'km_cls': km_cls, 'y_km': y_km, 'd_km': d_km}


def gen_counted_behavior(fited_behav_arch, gaurdar=None, gof_type=['aic']):
    fited_behaviors = np.load(fited_behav_arch).tolist()
    counted_all_behaviors = {gof: [ ] for gof in gof_type}
    for i in range(len(fited_behaviors)):
        for gof in gof_type:
            counted_all_behaviors[gof].extend(list(set([v[gof][0] for i, v in fited_behaviors[i].items()])))

    for gof in gof_type:
        counted_all_behaviors[gof] = set(counted_all_behaviors[gof])

    if gaurdar is not None:
        np.save(gaurdar + 'counted_all_behaviors', counted_all_behaviors)
    else:
        return counted_all_behaviors


def gen_alpha(fited_behav_arch, patt):
    fited_behav = np.load(fited_behav_arch).tolist()
    d_alpha = {i: [] for i in fited_behav}
    for poly, behavs in fited_behav.items():
        if patt in behavs:
            d_alpha[poly] = 1.0
        else:
            d_alpha[poly] = 0.0
    return np.asarray([a for p, a in d_alpha.items()])


def gen_row_col(behaviors, method):
    if method == 'Morris':
        col_labels = [0, 5, 10, 15, 20, 'Mean']
    else:
        col_labels = [0, 1, 2, 3, 4, 'Mean']

    col_l = []

    col_l.extend([f"{behav}_gof" for behav in behaviors])
    col_labels.extend(sorted(col_l, key=lambda word: (word[0], word)))

    col_l.extend([f"{behav}_{bpp}" for behav in behaviors for bpp in behaviors[behav]['bp_params']['Kaq']])
    col_labels.extend(sorted([pt for pt in col_l if not pt.endswith('_gof')], key=lambda word: (word[0], word)))

    gof = [pt for pt in col_l if pt.endswith('_gof')]
    spp_gof = [pt for pt in col_l if pt.startswith('spp') and pt.endswith('_gof')]
    spp = [pt for pt in col_l if pt.startswith('spp')]

    col = [f'n{i}' for i in range(1, 7)]
    col.extend([f'S{i}' for i in range(1, len(col_l) - len(spp)+1)])
    col.extend([f'D{i}' for i in range(1, len(spp)+1)])

    col_new = col[:6]
    col_new.extend([f'a{i}' for i in range(1, len(gof)+1)])
    col_new.extend([f'b{i}' for i in range(1, len(col_l) - len(gof) - len(spp) + len(spp_gof) +1)])
    col_new.extend([f'c{i}' for i in range(1, len(spp) - len(spp_gof) + 1)])

    row = [p for p in behaviors['log']['bp_params']]
    row_labels = ['Ptq', 'Ptr', 'Kaq', 'Peq', 'Pex', 'POH, Summer', 'POH, Winter', 'CTW', 'Dummy']

    return row_labels, col, col_labels, row, col_new


def gen_geog_map(gaurd_arch, measure='paso_tiempo', patt=None, method='Morris', param=None, fst_cut=0.1, snd_cut=8,
                 si=None):
    read_dt = _read_dt_4_map(method, si=si)
    para_labels = {prm: gen_row_col(read_dt['behaviors'], method)[0][i] for i, prm in enumerate(read_dt['pasos'])}

    if measure == 'paso_tiempo':
        for prm, paso in read_dt['pasos'].items():
            prm = para_labels[prm]
            map_sens(gen_geog(), method, measure, prm,
                     paso, fst_cut=fst_cut, snd_cut=snd_cut, ids=[str(i) for i in range(1, 216)],
                     path=gaurd_arch)

    elif measure == 'promedio':
        for prmm, m_aray in read_dt['means'].items():
            prmm = para_labels[prmm]
            map_sens(gen_geog(), method, measure, prmm,
                     m_aray, fst_cut=fst_cut, snd_cut=snd_cut, ids=[str(i) for i in range(1, 216)],
                     path=gaurd_arch)

    elif measure == 'behavior_gof':
        for patt, b_g in read_dt['behaviors'].items():
            alpha = gen_alpha(read_dt['no_ini'], patt)
            if Counter(alpha)[0] == 215:
                alpha = np.zeros([215])
            gof_prm = b_g['gof']
            for prm, bpprm in gof_prm.items():
                prm = para_labels[prm]
                map_sens(gen_geog(), method, measure, prm,
                         bpprm, fst_cut=fst_cut, snd_cut=snd_cut, behav=patt, ids=[str(i) for i in range(1, 216)],
                         alpha=alpha, path=gaurd_arch)

    elif measure == 'behavior_param':
        for patt, b_g in read_dt['behaviors'].items():
            alpha = gen_alpha(read_dt['no_ini'], patt)
            if Counter(alpha)[0] == 215:
                alpha = np.zeros([215])
            bpp_prm = b_g['bp_params']
            for prm, bpprm in bpp_prm.items():
                prm = para_labels[prm]
                map_sens(gen_geog(), method, measure, prm,
                         bpprm, fst_cut=fst_cut, snd_cut=snd_cut, behav=patt, ids=[str(i) for i in range(1, 216)],
                         alpha=alpha, path=gaurd_arch)

    elif measure == 'test':
        alpha = gen_alpha(read_dt['no_ini'], patt)
        b_g = read_dt['behaviors'][patt]
        bpp_prm = b_g['bp_params']
        gof_prm = b_g['gof']
        for prm, bpprm in bpp_prm.items():
            if prm != param:
                continue
            map_sens(gen_geog(), method, measure, prm,
                     bpprm, fst_cut=fst_cut, snd_cut=snd_cut, behav=patt, ids=[str(i) for i in range(1, 216)],
                     alpha=alpha, path=gaurd_arch)

    elif measure == 'geog_simul_percent':
        geog_simul_percent = np.load(gaurd_arch + 'patt_sens_simul.npy').tolist()
        for patt, poly_aray in geog_simul_percent.items():
            map_sens(gen_geog(), method, measure, 'a',
                     poly_aray, fst_cut=0, behav=patt, ids=[str(i) for i in range(1, 216)],
                     unid='% of sensitivity simulation data', path=gaurd_arch)


def gen_rank_map(rank_arch, method, fst_cut, snd_cut, rank_method, load_data, si=None, cluster=False, cls=None,  gof_type=['aic']):
    read_dt = _read_dt_4_map(method, si=si, load_data=load_data)

    for gof in gof_type:
        r_c = gen_row_col(read_dt['behaviors'][gof], method)

        data = np.empty([len(r_c[0]), len(r_c[4])])

        if rank_method == 'polygons':
            for i in range(215):
                # paso
                for prmp, d_paso in read_dt['pasos'].items():
                    for p in read_dt['ps']:
                        data[r_c[3].index(prmp), r_c[2].index(p)] = d_paso[f'paso_{p}'][i]

                # mean
                for prmm, m_aray in read_dt['means'].items():
                    data[r_c[3].index(prmm), r_c[2].index('Mean')] = m_aray[i]

                # behavior
                for patt, d_bg in read_dt['behaviors'].items():
                    alpha = gen_alpha(read_dt['no_ini'], patt)
                    for pbpp, bpp in d_bg['bp_params'].items():
                        if Counter(alpha)[0] == 215 or pbpp == 'Ficticia':
                            alpha = np.zeros([215])
                        for bppm, va in bpp.items():
                            if alpha[i] == 0 and patt != 'linear':
                                data[r_c[3].index(pbpp), r_c[2].index(f'{patt}_{bppm}')] = 0
                            else:
                                data[r_c[3].index(pbpp), r_c[2].index(f'{patt}_{bppm}')] = va[i]
                    for paic, aic in d_bg['gof'].items():
                        if alpha[i] == 0 and patt != 'linear':
                            data[r_c[3].index(paic), r_c[2].index(f'{patt}_gof')] = 0
                        else:
                            data[r_c[3].index(paic), r_c[2].index(f'{patt}_gof')] = aic['aic'][i]

                map_rank(row_labels=r_c[0], col_labels=r_c[1], data=np.round(data, 2),
                         title=f'{method} Sensitivity Ranking Results', y_label='Parameters',
                         archivo=rank_arch + f'poly{i + 1}', fst_cut=fst_cut, snd_cut=snd_cut, maxi=np.round(data, 2).max(),
                         cbarlabel=f"{method} Sensitivity Index", cmap="magma_r")
                print(f'finish the {i}-th poly, yeah!')

        elif rank_method == 'count_poly':
            for p in read_dt['ps']:
                n_dt = {prmp: len(d_paso[f'paso_{p}'][np.where(d_paso[f'paso_{p}'] > fst_cut)[0]]) / 215 for prmp, d_paso in
                        read_dt['pasos'].items()}
                for prmp, v in n_dt.items():
                    data[r_c[3].index(prmp), read_dt['ps'].index(p)] = v

            n_dt2 = {prmm: len(m_aray[np.where(m_aray > fst_cut)[0]]) / 215 for prmm, m_aray in read_dt['means'].items()}
            for prmm, m_aray in n_dt2.items():
                data[r_c[3].index(prmm), r_c[2].index('Mean')] = m_aray

            col_ind = []
            for patt, d_bg in read_dt['behaviors'].items():
                for pbpp, bpp in d_bg['bp_params'].items():
                    for bppm in bpp:
                        bpp[bppm] = np.asarray([0 if np.isnan(v) else v for v in bpp[bppm]])
                        col_ind.append(r_c[2].index(f'{patt}_{bppm}'))
                        data[r_c[3].index(pbpp), r_c[2].index(f'{patt}_{bppm}')] = \
                            len(bpp[bppm][np.where(bpp[bppm] > fst_cut)[0]]) / 215
                for paic, aic in d_bg['gof'].items():
                    for a in aic:
                        aic[a] = np.asarray([0 if np.isnan(v) else v for v in aic[a]])
                    col_ind.append(r_c[2].index(f'{patt}_gof'))
                    data[r_c[3].index(paic), r_c[2].index(f'{patt}_gof')] = \
                        len(aic['aic'][np.where(aic['aic'] > fst_cut)[0]]) / 215

            if len(np.where(np.isnan(data))[1]) != 0:
                data[np.where(np.isnan(data))] = 0

            map_rank(row_labels=r_c[0], col_labels=r_c[1], data=data,
                     title=f"{method} Polygonal Sensitivity occurance", y_label='Parameters',
                     archivo=rank_arch + f'{rank_method}', fst_cut=fst_cut, snd_cut=1, maxi=data.max(),
                     cbarlabel=f"{method} % of polygonal occurance Rank", cmap="magma_r", bin=10, rank_method=rank_method)

        elif rank_method == 'num_poly_rank':
            for p in read_dt['ps']:
                dt = {prmp: max(d_paso[f'paso_{p}']) for prmp, d_paso in read_dt['pasos'].items()}
                r = {key: rank for rank, key in enumerate(sorted(set(dt.values()), reverse=False), 1)}
                n_dt = {k: r[v] if v > fst_cut else 0 for k, v in dt.items()}
                for prmp, v in n_dt.items():
                    data[r_c[3].index(prmp), r_c[2].index(p)] = v

            dt2 = {prmm: max(m_aray) for prmm, m_aray in read_dt['means'].items()}
            r2 = {key: rank for rank, key in enumerate(sorted(set(dt2.values()), reverse=False), 1)}
            n_dt2 = {k: r2[v] if v > fst_cut else 0 for k, v in dt2.items()}
            for prmm, m_aray in n_dt2.items():
                data[r_c[3].index(prmm), r_c[2].index('Mean')] = m_aray
            col_ind = []
            for patt, d_bg in read_dt['behaviors'].items():
                for pbpp, bpp in d_bg['bp_params'].items():
                    for bppm, va in bpp.items():
                        col_ind.append(r_c[2].index(f'{patt}_{bppm}'))
                        data[r_c[3].index(pbpp), r_c[2].index(f'{patt}_{bppm}')] = max(va)
                for paic, aic in d_bg['gof'].items():
                    col_ind.append(r_c[2].index(f'{patt}_gof'))
                    data[r_c[3].index(paic), r_c[2].index(f'{patt}_gof')] = max(aic['aic'])
            lst = list(set(col_ind))
            for c_i in lst:
                dt = {para: data[:, c_i][i] for i, para in enumerate(r_c[3])}
                r = {key: rank for rank, key in enumerate(sorted(set(dt.values()), reverse=False), 1)}
                n_dt = {k: r[v] if v > fst_cut else 0 for k, v in dt.items()}
                for para, rk in n_dt.items():
                    data[r_c[3].index(para), c_i] = rk

            if len(np.where(np.isnan(data))[1]) != 0:
                data[np.where(np.isnan(data))] = 0

            if cluster is False:
                map_rank(row_labels=r_c[0], col_labels=[i[1:] for i in r_c[4]], data=np.round(data, 2),
                         title=None, y_label=None, dpi=1800,
                         archivo=rank_arch + f'{rank_method}', fst_cut=1, snd_cut=data.max(), maxi=data.max(),
                         cbarlabel=None, cmap="magma_r", bin=data.max() + 1,
                         rank_method=rank_method)
            else:
                points = np.transpose(data[:, 1:])
                cluster = clustering(points, cls)
                cls_col_n_od = ['N1']
                cls_col_km = ['N1']
                data_new_od = np.transpose(cluster['n_points'])
                data_km = np.transpose(cluster['km_cls'])
                for j in range(len(r_c[4]) - 1):
                    cls_col_n_od.append(r_c[4][cluster['new_order'][j] + 1])
                    cls_col_km.append(r_c[4][cluster['km_lst'][j] + 1])
                data_new_od = np.concatenate((data[:, 0].reshape(9, 1), data_new_od), axis=1)
                data_km = np.concatenate((data[:, 0].reshape(9, 1), data_km), axis=1)
                for cl in range(cls):
                    print(cl + 1, [r_c[2][i + 1] for i in cluster['d_km'][cl]])
                    print(cl + 1, [r_c[4][i + 1] for i in cluster['d_km'][cl]])

                print('new order: ', [r_c[4][i + 1] for i in cluster['new_order']])
                print('new order: ', [r_c[2][i + 1] for i in cluster['new_order']])

                map_rank(row_labels=r_c[0], col_labels=cls_col_n_od, data=np.round(data_new_od, 2),
                         title=None, y_label=None, dpi=1000,
                         archivo=rank_arch + 'new_order', fst_cut=1, snd_cut=data.max(), maxi=data.max(),
                         cbarlabel=None, cmap="magma_r", bin=data.max() + 1,
                         rank_method=rank_method)
                map_rank(row_labels=r_c[0], col_labels=cls_col_km, data=np.round(data_km, 2),
                         title=f"{method} K-Mean-{cls} Clustering Map", y_label='Parameters', dpi=1000,
                         archivo=rank_arch + f'k-mean-{cls}', fst_cut=1, snd_cut=data.max(), maxi=data.max(),
                         cbarlabel=f"{method} Sensitivity Rank", cmap="magma_r", bin=data.max() + 1,
                         rank_method=rank_method)

        elif rank_method == 'num_poly_rank_n':
            data = np.empty([len(r_c[0]), 12])
            read_dt_mo = _read_dt_4_map('Morris')
            read_dt_fa = _read_dt_4_map('Fast')
            r_c_mo = gen_row_col(read_dt['behaviors'], 'Morris')
            r_c_fa = gen_row_col(read_dt['behaviors'], 'Fast')
            col = r_c_mo[1][:6] * 2
            for p in read_dt_mo['ps']:
                dt = {prmp: max(d_paso[f'paso_{p}']) for prmp, d_paso in read_dt_mo['pasos'].items()}
                r = {key: rank for rank, key in enumerate(sorted(set(dt.values()), reverse=True), 1)}
                n_dt = {k: r[v] if v > 0.1 else 0 for k, v in dt.items()}
                for prmp, v in n_dt.items():
                    data[r_c_mo[3].index(prmp), r_c_mo[2].index(p)] = v
            for p in read_dt_fa['ps']:
                dt = {prmp: max(d_paso[f'paso_{p}']) for prmp, d_paso in read_dt_fa['pasos'].items()}
                r = {key: rank for rank, key in enumerate(sorted(set(dt.values()), reverse=True), 1)}
                n_dt = {k: r[v] if v > 0.01 else 0 for k, v in dt.items()}
                for prmp, v in n_dt.items():
                    data[r_c_mo[3].index(prmp), r_c_fa[2].index(p) + 6] = v

            if len(np.where(np.isnan(data))[1]) != 0:
                data[np.where(np.isnan(data))] = 0

            map_rank(row_labels=r_c[0], col_labels=col, data=np.round(data, 2),
                     title=f"Sensitivity Ranking Map", y_label='Parameters',
                     archivo=rank_arch + f'{rank_method}', fst_cut=1, snd_cut=data.max(), maxi=data.max(),
                     cbarlabel=f"Sensitivity Ranking Order", cmap="magma_r", bin=data.max() + 1, rank_method=rank_method)

        elif rank_method == 'total_poly':
            for prmp, d_paso in read_dt['pasos'].items():
                for p in read_dt['ps']:
                    data[r_c[3].index(prmp), r_c[2].index(p)] = max(d_paso[f'paso_{p}'])
            # mean
            for prmm, m_aray in read_dt['means'].items():
                data[r_c[3].index(prmm), r_c[2].index('Mean')] = max(m_aray)

            # behavior
            for patt, d_bg in read_dt['behaviors'].items():
                for pbpp, bpp in d_bg['bp_params'].items():
                    for bppm, va in bpp.items():
                        data[r_c[3].index(pbpp), r_c[2].index(f'{patt}_{bppm}')] = max(va)
                for paic, aic in d_bg['gof'].items():
                    data[r_c[3].index(paic), r_c[2].index(f'{patt}_gof')] = max(aic['aic'])

            if len(np.where(np.isnan(data))[1]) != 0:
                data[np.where(np.isnan(data))] = 0
            map_rank(row_labels=r_c[0], col_labels=r_c[4], data=np.round(data, 2),
                     title=f"{method} Sensitivity Ranking Results", y_label='Parameters',
                     archivo=rank_arch + f'{rank_method}', fst_cut=fst_cut, snd_cut=snd_cut, maxi=np.round(data, 2).max(),
                     cbarlabel=f"{method} Sensitivity Index", cmap="magma_r")


def map_sens(geog, metodo, measure, para_name, data, fst_cut, path, snd_cut=None, alpha=None, behav=None, paso=None,
             ids=None, unid=None):
    metodo = metodo.capitalize()
    vars_interés = {measure: {
        'col': ['#f8fef7', '#e7fde5', '#d6fcd2', '#c5fbc0', '#b4f9ad', '#b4f9ad', '#5ff352',
                '#5395a0', '#4a8590', '#42767f', '#39666e', '#30565d', '#28474c', '#1f373c',
                '#ffe6e6', '#ffcccc', '#ffb3b3', '#ff9999', '#ff8080', '#ff2f2f', '#ff0808'
                ],
        'escala_núm': [min(data), max(data)]}}

    clr_bar_dic = {'green': ['#e6fff2', '#b3ffd9'],
                   'blue': ['#80ff9f', '#80ff80', '#9fff80', '#bfff80', '#dfff80', '#ffff80', '#ffdf80', '#ff8080'],
                   'red': ['#ff0000', '#ff0000']}
    if unid is None:
        unid = f'{metodo} Sensitivity Index'

    for v, d in vars_interés.items():
        if measure == 'paso_tiempo':
            if metodo == 'Morris':
                ll = [0, 5, 10, 15, 20]
            elif metodo == 'Fast':
                ll = [0, 1, 2, 3, 4]
            for i in ll:
                max_val = max(data[f'paso_{i}'])
                if max_val > snd_cut:
                    data[f'paso_{i}'][np.where(data[f'paso_{i}'] > snd_cut)] = snd_cut
                    max_val = snd_cut
                # geog.dibujar(archivo=path + f'{i}-{para_name}', valores=data[f'paso_{i}'],
                #              título=f"{metodo}-{para_name}-Timestep-{i} to WTD",
                #              unidades=unid, colores=d['col'], ids=ids, fst_cut=fst_cut, snd_cut=snd_cut,
                #              clr_bar_dic=clr_bar_dic, escala_num=(0, max_val))
                geog.dibujar(archivo=path + f'{i*5}-{para_name}', valores=data[f'paso_{i}'],
                             título=f"{metodo}-{para_name}-Timestep-{i*5} to WTD",
                             unidades=unid, colores=d['col'], ids=ids, fst_cut=fst_cut, snd_cut=snd_cut,
                             clr_bar_dic=clr_bar_dic, escala_num=(0, max_val))

        elif measure == 'promedio':
            max_val = max(data)
            if max_val > snd_cut:
                data[np.where(data > snd_cut)] = snd_cut
                max_val = snd_cut
            geog.dibujar(archivo=path + f'mean-{para_name}', valores=data,
                         título=f"{metodo}-{para_name}-to WTD-Mean", fst_cut=fst_cut, snd_cut=snd_cut,
                         clr_bar_dic=clr_bar_dic, unidades=unid, colores=d['col'], ids=ids, escala_num=(0, max_val))

        elif measure == 'behavior_gof' or measure == 'behavior_param':
            for bpprm in data:
                if np.round(max(data[bpprm]), 4) == 0:
                    max_val = 0

                elif len(np.where(data[bpprm])[0][np.isin(np.where(data[bpprm]), np.where(alpha > 0))[0]]) == 0:
                    max_val = 0
                else:
                    max_val = max(
                        data[bpprm][[np.where(data[bpprm])[0][np.isin(np.where(data[bpprm]), np.where(alpha > 0))[0]]]])
                if max_val > snd_cut:
                    data[bpprm][np.where(data[bpprm] > snd_cut)] = snd_cut
                    data[bpprm][np.where(data[bpprm] < fst_cut)] = 0 - snd_cut * 0.1
                    fst_cut = fst_cut + snd_cut * 0.1
                    snd_cut = snd_cut - 0.1
                    max_val = snd_cut + 0.1 + snd_cut * 0.1

                geog.dibujar(archivo=path + f"{metodo}-{para_name}-{behav}-{bpprm}",
                             valores=data[bpprm],
                             título=f"{metodo}-{para_name}-{behav.capitalize()}-{bpprm}",
                             alpha=alpha, unidades=unid, colores=d['col'], ids=ids, fst_cut=fst_cut,
                             snd_cut=snd_cut, clr_bar_dic=clr_bar_dic, escala_num=(0, max_val)
                             )

        elif measure == 'test':
            for bpprm in data:
                # if bpprm != 'amplitude':
                #     continue
                # new_alpha = np.zeros([215])
                # new_alpha[np.where(data[bpprm])[0][np.isin(np.where(data[bpprm]), np.where(alpha > 0))[0]]] = 1
                if np.round(max(data[bpprm]), 4) == 0:
                    max_val = 0
                # elif behav == 'linear':
                #     alpha = 1
                #     max_val = max(data[bpprm])
                elif len(np.where(data[bpprm])[0][np.isin(np.where(data[bpprm]), np.where(alpha > 0))[0]]) == 0:
                    max_val = 0
                else:
                    max_val = max(
                        data[bpprm][[np.where(data[bpprm])[0][np.isin(np.where(data[bpprm]), np.where(alpha > 0))[0]]]])
                    # if max_val < max(data[bpprm]):
                    #     data[bpprm][np.where(data[bpprm] > max_val)[0]] = max_val
                if max_val > snd_cut:
                    data[bpprm][np.where(data[bpprm] > snd_cut)] = snd_cut
                    data[np.where(data < fst_cut)] = 0 - snd_cut * 0.1
                    max_val = snd_cut

                geog.dibujar(archivo=path + f"{metodo[:2]}-{para_name[: 5]}-{behav}-{bpprm}",  # fst_cut=fst_cut,
                             valores=data[bpprm],
                             título=f"{metodo[:2].capitalize()}-{para_name.capitalize()}-{behav.capitalize()}-{bpprm}",
                             alpha=alpha, unidades=unid, colores=d['col'], ids=ids, fst_cut=fst_cut + snd_cut * 0.1,
                             snd_cut=snd_cut,
                             clr_bar_dic=clr_bar_dic,
                             escala_num=(0, max_val)
                             )

        elif measure == 'geog_simul_percent':
            geog.dibujar(archivo=path + f"{metodo}-{behav}",
                         valores=data, título=f"{metodo}-{behav.capitalize()}",
                         unidades=unid, colores=d['col'], ids=ids, fst_cut=None, escala_num=(0, 1), n_bin=10
                         )


def map_rank(fst_cut, snd_cut, maxi, row_labels, col_labels, data, title, y_label, archivo, cbarlabel,
             ax=None, dpi=1000, cbar_kw={}, bin=None, rank_method=None, **kwargs):
    '''

    Parameters
    ----------
    row_labels:  A list or array of length N with the labels for the rows
    ["cucumber", "tomato", "lettuce", "asparagus", "potato", "wheat", "barley"]
    col_labels: A list or array of length M with the labels for the columns
    ["Farmer Joe", "Upland Bros.", "Smith Gardening", "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    data: A 2D numpy array of shape (N,M)
    np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
            [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
            [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
            [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
            [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
            [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
            [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.

    cmap="Wistia", cmap="magma_r" , cmap=plt.get_cmap("PiYG", 7) yellow-black, cmap="PuOr" brown
    vmin=0, vmin=-1, vmax=1
    cbar_kw=dict(ticks=np.arange(-3, 4),

    Returns
    -------

    '''
    if not ax:
        fig, ax = plt.subplots()

    # clr_bar_dic = {'green': ['#e6fff2', '#80ffbf', '#33ff99', '#00cc66', '#006633'],
    #                'blue': ['#80dfff', '#00bfff', '#1ac6ff', '#00ace6', '#007399'],
    #                'red': ['#ff8000', '#ff8c1a', '#ff8000']}

    clr_bar_dic = {'green': ['#e6fff2', '#b3ffd9'],
                   'blue': ['#80ff9f', '#80ff80', '#9fff80', '#bfff80', '#dfff80', '#ffff80', '#ffdf80', '#ff8080'],
                   'red': ['#ff0000', '#ff0000']}

    divider = make_axes_locatable(ax)
    if cbarlabel is not None:
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
    else:
        cax = divider.append_axes("bottom", size="10%", pad=0.05)

    if bin is not None:
        if rank_method == 'num_poly_rank' or rank_method == 'num_poly_rank_n':
            dic_c = _gen_d_mapacolores(
                ['#e6fff2', '#b3ffd7', '#80ff80', '#9fff80', '#bfff80', '#dfff80', '#ffff80', '#ffdf80', '#ff8080'],
                maxi=None)
        elif rank_method == 'count_poly':
            dic_c = _gen_d_mapacolores(
                ['#e6fff2', '#b3ffd7', '#80ff9f', '#9fff80', '#bfff80', '#dfff80', '#ffff80', '#ffdf80',
                 '#ff8080'],
                maxi=None)

        mapa_color = LinearSegmentedColormap('mapa_color', dic_c, N=bin)
        im = ax.imshow(data, mapa_color)
        if rank_method == 'count_poly':
            cbar = fig.colorbar(im, cax=cax, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            cbar.ax.set_yticklabels(
                ['0', '~10%', '~20%', '~30%', '~40%', '~50%', '~60%', '~70%', '~80%', '~90%', '~100%'],
                fontsize=5)
        elif rank_method == 'num_poly_rank':
            cbar = fig.colorbar(im, cax=cax, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], orientation="horizontal")
            cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=6)
        elif rank_method == 'num_poly_rank_n':
            cbar = fig.colorbar(im, cax=cax, ticks=[0, 6, 5, 4, 3, 2, 1])
            cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=6)

    else:
        if data.max() > snd_cut:
            data[np.where(data > snd_cut)] = snd_cut + 0.1
            data[np.where(data < fst_cut)] = 0 - snd_cut * 0.1
            dic_c = _gen_clrbar_dic(fst_cut=fst_cut + snd_cut * 0.1, snd_cut=snd_cut - 0.1,
                                    maxi=snd_cut + 0.1 + snd_cut * 0.1,
                                    first_set=clr_bar_dic['green'],
                                    second_set=clr_bar_dic['blue'], third_set=clr_bar_dic['red'])
            mapa_color = LinearSegmentedColormap('mapa_color', dic_c)
            im = ax.imshow(data, mapa_color)

            cbar = fig.colorbar(im, cax=cax, ticks=[data.min(), fst_cut, snd_cut - 0.1])
            cbar.ax.set_yticklabels(
                ['0', f'Screnning threshold, {fst_cut}', f'+{snd_cut} High sensitivity zone'],
                fontsize=3)

        elif fst_cut < data.max() < snd_cut:
            data[np.where(data < fst_cut)] = 0 - maxi * 0.1
            dic_c = _gen_clrbar_dic(fst_cut=fst_cut + maxi * 0.1, snd_cut=None, maxi=maxi + maxi * 0.1,
                                    first_set=clr_bar_dic['green'], second_set=clr_bar_dic['blue'],
                                    third_set=clr_bar_dic['red'])
            mapa_color = LinearSegmentedColormap('mapa_color', dic_c)
            im = ax.imshow(data, mapa_color)
            cbar = fig.colorbar(im, cax=cax, ticks=[data.min(), fst_cut, data.max()])
            # cbar = ax.figure.colorbar(im, ax=cax, ticks=[0, fst_cut, data.max()], **cbar_kw)
            cbar.ax.set_yticklabels(['0', f'Screnning threshold, {fst_cut}', f'maximum val, {np.round(data.max(), 3)}'],
                                    fontsize=3)

        elif data.max() < fst_cut:
            dic_c = _gen_clrbar_dic(fst_cut=None, snd_cut=None, maxi=maxi, first_set=clr_bar_dic['green'],
                                    second_set=clr_bar_dic['blue'], third_set=clr_bar_dic['red'])
            mapa_color = LinearSegmentedColormap('mapa_color', dic_c)
            im = ax.imshow(data, mapa_color)
            cbar = fig.colorbar(im, cax=cax, ticks=[0, data.max()])
            # cbar = ax.figure.colorbar(im, ax=cax, ticks=[0, data.max()], **cbar_kw)
            cbar.ax.set_yticklabels(['0', f'maximum val, {data.max()}'], fontsize=3)

    if cbarlabel is not None:
        cbar.ax.set_ylabel(cbarlabel, rotation=0, va="bottom", fontsize=7)  # fontsize=15)
    ax.tick_params(width=0.1)  # (width=1)
    cbar.ax.tick_params(width=0.1)  # (width=1) #(width=0.1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if rank_method == 'num_poly_rank_n':
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticklabels(row_labels, fontsize=10)
    else:
        ax.set_xticklabels(col_labels, fontsize=4)
        ax.set_yticklabels(row_labels, fontsize=4)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    def func(x, pos):
        return "{:.2f}".format(x).replace("-0.80", "0").replace("8.10", "")

    def func2(x):
        return "{:}".format(x).replace("0", "")

    def _annotate_rankmap(fst_cut=fst_cut, snd_cut=snd_cut, im=im, valfmt='{:.2f}',
                          txtclr=["white", "black", ''],
                          **textkw):
        """
        A function to annotate a rankmap.

        Arguments:
            im         : The AxesImage to be labeled.
        Optional arguments:
            data       : Data used to annotate. If None, the image's data is used.
            valfmt     : The format of the annotations inside the heatmap.
                         This should either use the string format method, e.g.
                         "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
            textcolors : A list or array of two color specifications. The first is
                         used for values below a threshold, the second for those
                         above.
            threshold  : Value in data units according to which the colors from
                         textcolors are applied. If None (the default) uses the
                         middle of the colormap as separation.

        Further arguments are passed on to the created text labels.
        valfmt="{x:d}" (integer format), valfmt="{x:.1f}",
        **textkw: size=7, fontweight="bold",
        """

        # Normalize the threshold to the images color range.
        fst_cut = im.norm(fst_cut)
        snd_cut = im.norm(snd_cut)

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        if bin is None:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    kw.update(color=txtclr[fst_cut < im.norm(data[i, j]) < snd_cut])
                    text = im.axes.text(j, i, matplotlib.ticker.FuncFormatter(func)(data[i, j], None), **kw)
                    texts.append(text)
        if rank_method == 'num_poly_rank':
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    texts.append(
                        ax.text(j, i, int(data[i, j]), ha="center", va="center", color=txtclr[0 < im.norm(data[i, j])],
                                size=2))
        elif rank_method == 'num_poly_rank_n':
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    texts.append(
                        ax.text(j, i, int(data[i, j]), ha="center", va="center", color=txtclr[0 < im.norm(data[i, j])],
                                size=6))
        elif rank_method == 'count_poly':
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    texts.append(
                        ax.text(j, i, f"{np.round(data[i, j] * 100, 1)}%", ha="center", va="center",
                                color=txtclr[0 < im.norm(data[i, j])],
                                size=1.5))

        return texts

    if bin is None:
        _annotate_rankmap(im=im, size=0.5)
    else:
        _annotate_rankmap(im=im, size=0.5, txtclr=["white", "black"], valfmt=matplotlib.ticker.FuncFormatter(func2))
    # matplotlib.ticker.FuncFormatter(
    #     "{:.2f}".format(data).replace("0.", ".").replace("0.00", "")))
    # Loop over data dimensions and create text annotations.
    if title is not None:
        ax.set_title(title, fontsize=10, y=1.5)  # fontsize=20, y=1.1)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=10)  # fontsize=20)

    fig.tight_layout(h_pad=1)
    fig.savefig(archivo, dpi=dpi)
    plt.close()

def criteria_stat(num_sample, sample_path, simul_arch, gof_type=['aic', 'bic', 'mic', 'srm', 'press', 'fpe']):
    var_egr='mds_Watertable depth Tinamit'
    # poly = np.sort(np.concatenate((list(ori_calib[1]), list(ori_valid[1]))))
    poly  = np.arange(1, 216)
    gof = {g: {cri: np.zeros([num_sample, len(poly)]) for cri in ['converage', 'linear', 'rmse']} for g in gof_type}

    for i in range(num_sample):
        simulation = carg_simul_dt(simul_arch, i, var_egr, 215, load_i_file=True)[0][str(i)]
        sample_data = np.load(sample_path+f'{i}.npy').tolist()

        for j, d in enumerate(poly):
            print(f"processing sample {i}, poly-{j}")
            sam_dt = {
                patt: {bp_g: {bpg: v[0][d - 1] for bpg, v in d_bp_g.items()} for bp_g, d_bp_g in d_dt.items()} for
                patt, d_dt in sample_data.items()}

            d_bp = find_best_behavior(sam_dt, gof_type=gof_type)[0]

            sim_linear = simple_shape(np.arange(len(simulation)), simulation[:, d - 1], tipo_egr='linear', gof=False)['bp_params'][
                'slope']

            for g in gof_type:
                pred =  predict(np.arange(len(simulation)), sam_dt[d_bp[g][0]]['bp_params'], d_bp[g][0])
                gof[g]['converage'][i][j] =  np.count_nonzero(~np.isnan(pred))/len(simulation)
                gof[g]['linear'][i][j] = (np.sign(sim_linear) == np.sign(
                    simple_shape(np.arange(len(simulation)), pred, tipo_egr='linear', gof=False)['bp_params']['slope']))
                gof[g]['rmse'][i][j] = compute_rmse(simulation[:,j], pred)

    np.save("D:\Gaby\Tinamit\Dt\Mor\\gof_stat_all", gof)
    return gof

def plot_4_select_criteria(sam_ind, y_data, counted_all_behaviors, all_beh_dt, plot_path, dim, all_sims):
    print(f"plot path for sample {sam_ind}")
    colors = {'aic': 'g', 'bic': 'b', 'mic': 'magenta', 'srm': 'cyan', 'press': 'black', 'fpe': 'yellow'}
    markers = {'aic': 'x', 'bic': 4, 'mic': ',', 'srm': '+', 'press': '.', 'fpe': 5}

    for i, poly in enumerate(dim):
        plt.ioff()

        if all_sims is not False:
            q5, q95 = [], []
            time = np.arange(all_sims.shape[1])

            for t in time:
                q5.append(np.percentile(all_sims[:, t, poly], 2.5))
                q95.append(np.percentile(all_sims[:, t, poly], 97.5))

            fig, ax = plt.subplots()
            ax.plot(q5, color='lightblue', linestyle='solid')
            ax.plot(q95, color='lightblue', linestyle='solid')
            ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='lightblue', zorder=0,
                            linewidth=0, label='5th-95th uncertainty bounds', alpha=0.4)

            ax.set_xticklabels([str(t + 2) for t in time], size=8)
            # plt.legend(prop={'size': 8}, frameon=False)

        x = np.arange(len(y_data))
        plt.plot(x, y_data[:, i], label=f'Polygon{poly+1}\nSimulation', color='r', linewidth=2, alpha=0.4)

        for criteria, l_pattern in counted_all_behaviors.items():
            y_criteria = predict(x, all_beh_dt[i][l_pattern[i]]['bp_params'], l_pattern[i])
            plt.plot(x, y_criteria, label=f'{criteria.upper()}', color=colors[criteria], linestyle='dashed', marker=markers[criteria], alpha=0.5)
            # plt.plot(x, y_criteria, label=f'{criteria.upper()}: {l_pattern[i]}', color=colors[criteria], linestyle='dashed', marker=markers[criteria], alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_path + f'{sam_ind}_{poly+1}', dpi=500)
        # plt.legend(prop={'size': 8})
        plt.savefig(plot_path + f'{sam_ind}_{poly+1}', dpi=1000)
        plt.clf()
        plt.close('all')


def hist_conv(gof_type, data, save):
    plt.ioff()
    sns.set()
    polys = np.concatenate((list(ori_calib[1]), list(ori_valid[1])))
    plot_dt = {gof: np.zeros(len(polys))for gof in gof_type}

    for gof in gof_type:
        for i, poly in enumerate(polys):
            plot_dt[gof][i] = np.average(data[gof]['coverage'][:, i])
        x = pd.Series(plot_dt[gof], name="Polygons")
        ax = sns.distplot(x)
        ax.savefig(save + f"{gof}-hist.png", dpi=300)
        plt.clf()


def confidence_interval_simulations(simul_arch, dims, save_path, método='morris'):

    simulation, var_egr = carg_simul_dt(simul_arch['arch_simular'], simul_arch['num_samples'],
                                        var_egr='mds_Watertable depth Tinamit', dim=None, método=método,
                                        tipo_egr='superposition')

    simulation = np.asarray([v for i, v in simulation.items()])
    dims =  [d-1 for d in dims]
    q5, q95 = [], []
    time = np.arange(simulation.shape[1])

    for d in dims:
        for t in time:
            q5.append(np.percentile(simulation[:, t, d], 2.5))
            q95.append(np.percentile(simulation[:, t, d], 97.5))

        fig, ax = plt.subplots()
        ax.plot(q5, color='lightblue', linestyle='solid')
        ax.plot(q95, color='lightblue', linestyle='solid')
        ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='lightblue', zorder=0,
                    linewidth=0, label='5th-95th percentile parameter uncertainty', alpha=0.4)

        ax.set_xticklabels([str(t + 2) for t in time], size=8)
        # plt.legend(prop={'size': 8}, frameon=False)
        plt.savefig(save_path + f'{d + 1}', dpi=500)

        plt.clf()
        plt.cla()
