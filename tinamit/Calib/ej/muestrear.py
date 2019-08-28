import os
import platform
from tinamit.Análisis.Sens.muestr import muestrear_paráms, guardar_mstr_paráms, cargar_mstr_paráms
from tinamit.Calib.ej.info_paráms import líms_paráms, mapa_paráms

método = 'morris'

if platform.release() == '7':
    archivo = "D:\Thesis\pythonProject\localuse\Dt\\"
else:
    archivo = ' '

def cargar_mstr_dt(método='morris'):
    if método == 'morris' and os.path.isfile(archivo + 'Mor\Mor_home\sampled_data\\muestra_morris_625.json'):
        mstr = cargar_mstr_paráms(archivo + 'Mor\Mor_home\sampled_data\\muestra_morris_625.json')

    elif método == 'fast' and os.path.isfile(archivo + 'Fast\sampled data\\muestra_fast_23params.json'):
        mstr = cargar_mstr_paráms(archivo + 'Fast\sampled data\\muestra_fast_23params.json')

    elif método == 'morris':
        mstr = muestrear_paráms(líms_paráms, 'morris', mapa_paráms=mapa_paráms,
                                ops_método={'N': 25, 'num_levels': 16, 'grid_jump': 8})
        guardar_mstr_paráms(mstr, archivo + 'Mor\sampled data\\')

    elif método == 'fast':
        mstr = muestrear_paráms(líms_paráms, 'fast', mapa_paráms=mapa_paráms, ops_método={'N': 5000})
        guardar_mstr_paráms(mstr, archivo + 'Fast\sampled data\\')
    return mstr

