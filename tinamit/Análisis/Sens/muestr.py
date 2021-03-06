import json
import os
from copy import deepcopy

import numpy as np
from SALib.sample import morris
from SALib.sample import fast_sampler

from tinamit.config import _


def muestrear_paráms(líms_paráms, método, mapa_paráms=None, ops_método=None, ficticia=True):
    """
    sampling data from the selected senc método
    Parameters
    ----------
    líms_paráms: dict[str, tuple | list[tuple]]
    método: ['morris', 'fast']
        El tipo de algoritmo para el análisis de sensitvidad.
    ops_método: dict

    Returns
    -------
    sampled data
    """
    método = método.lower()
    if ops_método is None:
        ops_método = {}

    if ficticia is True:
        ficticia = {"Ficticia": (0, 1)}
        líms_paráms = deepcopy(líms_paráms)
        líms_paráms.update(ficticia)

    problema, líms_paráms_final = gen_problema(líms_paráms, mapa_paráms)

    if método == 'morris':
        ops = {'N': 25, 'num_levels': 8, 'grid_jump': 4}
        ops.update(ops_método)
        mstr = morris.sample(problema, **ops)

    elif método == 'fast':
        ops = {'N': 65} #10000
        ops.update(ops_método)
        mstr = fast_sampler.sample(problema, **ops)
    else:
        raise ValueError(_('Algoritmo "{}" no reconocido.').format(método))

    dic_mstr = {p: mstr[:, í] for í, p in enumerate(líms_paráms_final)}

    return dic_mstr


def gen_problema(líms_paráms, mapa_paráms, ficticia=True):
    if ficticia is True:
        ficticia = {"Ficticia": (0, 1)}
        líms_paráms = deepcopy(líms_paráms)
        líms_paráms.update(ficticia)

    if mapa_paráms is None:
        if not all(isinstance(r, tuple) for r in líms_paráms.values()):
            raise TypeError(_('Debes especificar `mapa_paráms` si quieres empleaer '
                              'una lista de rangos para un parámetro.'))
        líms_paráms_final = líms_paráms
    else:
        faltan = {p for p in mapa_paráms if p not in líms_paráms}
        if len(faltan):
            raise ValueError(_('Los siguientes parámetros aparecen en `mapa_paráms` pero no en `líms_paráms`:'
                               '\n\t{}').format(', '.join(faltan)))

        líms_paráms_final = {p: r for p, r in líms_paráms.items() if p not in mapa_paráms}

        for p, mapa in mapa_paráms.items():
            if isinstance(mapa, list):
                mapa = np.array(mapa)
            if isinstance(mapa, np.ndarray):
                n_versiones_parám = len(líms_paráms[p])
                if not np.all(np.isin(mapa, range(n_versiones_parám))):
                    raise ValueError(
                        _('Los índices en `mápa_paráms` no corresponden con el número de rangos en `líms_paráms` para'
                          'el parámetro "{}".').format(p))
                líms_paráms_final.update(
                    {f'{p}_{í}': líms_paráms[p][í] for í in range(n_versiones_parám)}
                )

            elif isinstance(mapa, dict):
                n_versiones_parám = len(líms_paráms[p])
                conj_vals = set(l for v_p in mapa['mapa'].values() for r in v_p for l in r)
                if not np.all(np.isin(list(conj_vals), range(n_versiones_parám))):
                    raise ValueError(
                        _('Los índices en `mápa_paráms` no corresponden con el número de rangos en `líms_paráms` para'
                          'el parámetro "{}".').format(p))

                # líms_paráms_final.update({f"{key}_{i}": líms_paráms[p][i] for key in mapa['mapa'] for i in range(n_versiones_parám)})
                líms_paráms_final.update(
                    {f'{p}_{í}': líms_paráms[p][í] for í in range(n_versiones_parám)}
                )
            else:
                raise TypeError(_('Tipo "{}" inválido para `mapa_paráms`.').format(type(mapa_paráms)))

    problema = {
        'num_vars': len(líms_paráms_final),
        'names': list(líms_paráms_final),
        'bounds': list(líms_paráms_final.values())
    }

    return problema, líms_paráms_final


def guardar(contenido, archivo):
    arch, ext = os.path.splitext(archivo)
    if ext != '.json':
        archivo = arch + '.json'
    with open(archivo, 'w', encoding='UTF-8') as d:
        json.dump(contenido, d, ensure_ascii=False)


def guardar_mstr_paráms(muestrear_paráms, archivo):
    mstr_json = {c: [p.tolist() for p in d] for c, d in muestrear_paráms.items()}
    guardar(mstr_json, archivo)


def cargar(archivo):
    arch, ext = os.path.splitext(archivo)
    if ext != '.json':
        archivo = arch + '.json'
    with open(archivo, encoding='UTF-8') as d:
        contenido_json = json.load(d)
    return contenido_json


def cargar_mstr_paráms(archivo):
    mstr_json = cargar(archivo)
    mstr = {c: [np.array(p) for p in d] for c, d in mstr_json.items()}
    return mstr
