import csv
import datetime as ft
import os
import shutil
from collections import OrderedDict
from warnings import warn as avisar

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import shapefile as sf
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as TelaFigura
from matplotlib.figure import Figure as Figura
from تقدیر.ذرائع.مشاہدات import دن_مشا, مہنہ_مشا, سال_مشا
from تقدیر.مقام import مقام

from tinamit.config import _
from tinamit.cositas import detectar_codif, valid_nombre_arch

# Ofrecemos la oportunidad de utilizar taqdir, تقدیر, en español

conv_vars = {
    'Precipitación': 'بارش',
    'Radiación solar': 'شمسی_تابکاری',
    'Temperatura máxima': 'درجہ_حرارت_زیادہ',
    'Temperatura mínima': 'درجہ_حرارت_کم',
    'Temperatura promedia': 'درجہ_حرارت_اوسط'
}


# Una subclase traducida, mientras que Lassi esté en desarrollo
class Lugar(مقام):
    """
    Esta clase conecta con la clase مقام, o Lugar, del paquete تقدیر.
    """

    def __init__(símismo, lat, long, elev):
        """
        Inciamos el :class:`Lugar` con sus coordenadas.

        :param lat: La latitud del lugares.
        :type lat: float | int
        :param long: La longitud del lugares.
        :type long: float | int
        :param elev: La elevación del lugares, en metros.
        :type elev: float | int
        """

        # Iniciamos como مقام
        super().__init__(چوڑائی=lat, طول=long, بلندی=elev)

    def observar_diarios(símismo, archivo, cols_datos, conv, c_fecha):
        """
        Esta función permite conectar observaciones diarias de datos climáticos.

        :param archivo: El fuente con la base de datos.
        :type archivo: str

        :param cols_datos: Un diccionario, donde cada llave es el nombre oficial del variable climático y
          el valor es el nombre de la columna en la base de datos.
        :type cols_datos: dict[str, str]

        :param conv: Diccionario de factores de conversión para cada variable. Las llaves deben ser el nombre del
          variable **en Tinamït**.
        :type conv: dict[str, int | float]

        :param c_fecha: El nombre de la columna con las fechas de las observaciones.
        :type c_fecha: str

        """

        for v in cols_datos:
            if v not in conv_vars:
                raise KeyError(_('Error en observaciones diarias: "{}" no es variable climático reconocido en Tinamït. '
                                 'Debe ser uno de: {}').format(v, ', '.join(conv_vars)))
        for v in conv:
            if v not in conv_vars:
                raise KeyError(_('Error en factores de conversión: "{}" no es variable climático reconocido en '
                                 'Tinamït. Debe ser uno de: {}').format(v, ', '.join(conv_vars)))

        d_cols = {conv_vars[x]: cols_datos[x] for x in cols_datos}
        d_conv = {conv_vars[v]: c for v, c in conv.items()}

        obs = دن_مشا(مسل=archivo, تبادلوں=d_conv, س_تاریخ=c_fecha, س_اعداد=d_cols)

        símismo.مشاہدہ_کرنا(مشاہد=obs)

    def observar_mensuales(símismo, archivo, cols_datos, conv, meses, años):
        """
        Esta función permite conectar observaciones mensuales de datos climáticos.

        :param archivo: El fuente con la base de datos.
        :type archivo: str

        :param cols_datos: Un diccionario, donde cada llave es el nombre oficial del variable climático y el valor es
          el nombre de la columna en la base de datos.
        :type cols_datos: dict[str, str]

        :param conv: Diccionario de factores de conversión para cada variable. Las llaves deben ser el nombre del
          variable **en Tinamït**.
        :type conv: dict[str, int | float]

        :param meses: El nombre de la columna con los meses de las observaciones.
        :type meses: str

        :param años: El nombre de la columna con los años de las observaciones.
        :type años: str

        """

        for v in cols_datos:
            if v not in conv_vars:
                raise KeyError(_('Error en observaciones mensuales: "{}" no es variable climático reconocido en '
                                 'Tinamït. Debe ser uno de: {}').format(v, ', '.join(conv_vars)))
        for v in conv:
            if v not in conv_vars:
                raise KeyError(_('Error en factores de conversión: "{}" no es variable climático reconocido en '
                                 'Tinamït. Debe ser uno de:\t\n{}').format(v, ', '.join(conv_vars)))

        d_cols = {conv_vars[x]: cols_datos[x] for x in cols_datos}
        d_conv = {conv_vars[v]: c for v, c in conv.items()}

        obs = مہنہ_مشا(مسل=archivo, س_اعداد=d_cols, تبادلوں=d_conv, س_مہینہ=meses, س_سال=años)

        símismo.مشاہدہ_کرنا(obs)

    def observar_anuales(símismo, archivo, cols_datos, conv, años):
        """
        Esta función permite conectar observaciones anuales de datos climáticos.

        :param archivo: El fuente con la base de datos.
        :type archivo: str

        :param cols_datos: Un diccionario, donde cada llave es el nombre oficial del variable climático y el valor es
          el nombre de la columna en la base de datos.
        :type cols_datos: dict[str, str]

        :param conv: Diccionario de factores de conversión para cada variable. Las llaves deben ser el nombre del
          variable **en Tinamït**.
        :type conv: dict[str, int | float]

        :param años: El nombre de la columna con los años de las observaciones.
        :type años: str

        """

        for v in cols_datos:
            if v not in conv_vars:
                raise KeyError(_('Error en observaciones anuales: "{}" no es variable climático reconocido en Tinamït. '
                                 'Debe ser uno de: {}').format(v, ', '.join(conv_vars)))
        for v in conv:
            if v not in conv_vars:
                raise KeyError(_('Error en factores de conversión: "{}" no es variable climático reconocido en '
                                 'Tinamït. Debe ser uno de: {}').format(v, ', '.join(conv_vars)))

        d_cols = {conv_vars[x]: cols_datos[x] for x in cols_datos}
        d_conv = {conv_vars[v]: c for v, c in conv.items()}

        obs = سال_مشا(مسل=archivo, س_اعداد=d_cols, تبادلوں=d_conv, س_سال=años)
        símismo.مشاہدہ_کرنا(obs)

    def prep_datos(símismo, fecha_inic, fecha_final, tcr=0, prefs=None, lím_prefs=False):
        """
        Esta función actualiza el diccionario interno de datos climáticos del Lugar, listo para una simulación.
        Intentará obtener datos de todas fuentes posibles, incluso observaciones y modelos de predicciones climáticas.

        :param fecha_inic: La fecha inicial.
        :type fecha_inic: ft.date | ft.datetime
        :param fecha_final: La fecha final.
        :type fecha_final: ft.date | ft.datetime
        :param tcr: El escenario climático, o trayectorio de concentración relativa (tcr), de la IPCC
          (puede ser uno de 2.6, 4.5, 6.0, o 8.5).
        :type tcr: str | float
        :param prefs: Una lista opcional de fuentes potenciales de datos, en orden de preferencia.
        :type prefs: list
        :param lím_prefs: Si hay que limitar las fuentes de datos
        :type lím_prefs: bool

        """
        super().اعداد_تیاری(
            fecha_inic, fecha_final, tcr, ش_ترکار=1, ترجیحات=prefs, ترجیحات_محدود=lím_prefs, دوبارہ_پیدا=True
        )

    def devolver_datos(símismo, vars_clima, f_inic, f_final):
        """
        Esta función devuelve datos ya calculados por :func:`~tinamit.Geog.Geog.Lugar.prep_datos`.

        :param vars_clima: Una lista de variables climáticos de interés.
        :type vars_clima: list[str] | str
        :param f_inic: La fecha inicial.
        :type f_inic: ft.datetime | ft.date
        :param f_final: La fecha final.
        :type f_final: ft.datetime | ft.date
        :return: Los datos pedidos.
        :rtype: pd.DataFrame
        """
        bd = símismo.اعداد_دن  # type: pd.DataFrame
        datos_interés = bd.loc[f_inic:f_final]

        if not isinstance(vars_clima, list):
            vars_clima = [vars_clima]

        for v in vars_clima:
            if v not in conv_vars:
                raise ValueError(_('El variable "{}" está erróneo. Debe ser uno de:\n'
                                   '\t{}').format(v, ', '.join(conv_vars)))

        v_conv = [conv_vars[v] for v in vars_clima]

        return datos_interés[v_conv].rename(index=str, columns={conv_vars[v]: v for v in vars_clima})

    def comb_datos(símismo, vars_clima, combin, f_inic, f_final):
        """
        Esta función combina datos climáticos entre dos fechas.

        :param vars_clima: Los variables de clima de interés.
        :type vars_clima: list[str] | str
        :param combin: Cómo hay que combinar (promedio o total)
        :type combin: list[str] | str
        :param f_inic: La fecha inicial
        :type f_inic: ft.datetime | ft.date
        :param f_final: La fecha final
        :type f_final: ft.datetime | ft.date
        :return: Un diccionario con los resultados.
        :rtype: dict[float]
        """

        bd = símismo.اعداد_دن  # type: pd.DataFrame
        datos_interés = bd.loc[f_inic:f_final]

        if isinstance(combin, str):
            combin = [combin]
        if isinstance(vars_clima, str):
            vars_clima = [vars_clima]

        resultados = {}
        for v, c in zip(vars_clima, combin):
            try:
                v_conv = conv_vars[v]
            except KeyError:
                raise ValueError(_('El variable "{}" está erróneo. Debe ser uno de:\n'
                                   '\t{}').format(v, ', '.join(conv_vars)))
            if c is None:
                if v in ['Temperatura máxima', 'Temperatura mínima', 'Temperatura promedia']:
                    c = 'prom'
                else:
                    c = 'total'

            if c == 'prom':
                resultados[v] = datos_interés[v_conv].mean()
            elif c == 'total':
                resultados[v] = datos_interés[v_conv].sum()
            else:
                raise ValueError(_('El método de combinación de datos debe ser "prom" o "total".'))

        return resultados


class Geografía(object):
    """
    Esta clase representa la geografía de un lugar.
    """

    def __init__(símismo, nombre, archivo=None):

        símismo.nombre = nombre
        símismo.formas_reg = {}
        símismo.formas = {}

        símismo.orden_jerárquico = []
        símismo.info_geog = {}
        símismo.cód_a_lugar = {}

        símismo.escalas = []

        if archivo is not None:
            símismo.espec_estruct_geog(archivo)

    def agregar_forma(símismo, archivo, nombre=None, tipo=None, alpha=None, color=None, llenar=None):
        if nombre is None:
            nombre = os.path.splitext(os.path.split(archivo)[1])[0]

        símismo.formas[nombre] = {'obj': sf.Reader(archivo),
                                  'color': color,
                                  'llenar': llenar,
                                  'alpha': alpha,
                                  'tipo': tipo}

    def agregar_frm_regiones(símismo, archivo, col_id, escala_geog=None):
        """

        :param archivo:
        :type archivo:
        :param col_id:
        :type col_id:
        :return:
        :rtype:
        """
        af = sf.Reader(archivo)

        attrs = af.fields[1:]
        nombres_attr = [field[0] for field in attrs]

        try:
            ids = np.array([x.record[nombres_attr.index(col_id)] for x in af.shapeRecords()], dtype=str)
        except ValueError:
            raise ValueError(_('La columna "{}" no existe en la base de datos.').format(col_id))
        if escala_geog is None:

            ids_no_existen = [id_ for id_ in ids if str(id_) not in símismo.cód_a_lugar]
            if len(ids_no_existen):
                avisar(_('Las formas con id "{}" no se encuentran en la geografía actual.').format(ids_no_existen))
            escls_ids = set(símismo.obt_escala_región(id_) for id_ in ids if id_ not in ids_no_existen)
            if len(escls_ids) > 1:
                raise ValueError
            elif len(escls_ids) == 1:
                if escala_geog is None:
                    escala_geog = list(escls_ids)[0]
            else:
                if escala_geog is None:
                    escala_geog = 'Automática'

        símismo.formas_reg[escala_geog] = {'af': af, 'ids': ids}

    def quitar_forma(símismo, forma):
        try:
            símismo.formas.pop(forma)
        except KeyError:
            símismo.formas_reg.pop(forma)

    def espec_estruct_geog(símismo, archivo, col_cód='Código'):

        símismo.cód_a_lugar.clear()
        símismo.info_geog.clear()
        símismo.orden_jerárquico.clear()
        símismo.escalas.clear()

        col_cód = col_cód.lower()

        codif_csv = detectar_codif(archivo)
        with open(archivo, newline='', encoding=codif_csv) as d:

            l = csv.DictReader(d)  # El lector de csv

            # Guardar la primera fila como nombres de columnas
            cols = [x.lower().strip() for x in l.fieldnames]

            if col_cód not in cols:
                raise ValueError(_('La columna de código de región especificada ({}) no concuerda con los nombres de '
                                   'columnas del fuente ({}).').format(col_cód, ', '.join(cols)))

            doc = [OrderedDict((ll.lower().strip(), v.strip()) for ll, v in f.items()) for f in l]

        # Inferir el orden de la jerarquía
        órden = []

        escalas = [x for x in cols if x != col_cód]
        símismo.escalas.extend(escalas)

        símismo.info_geog.update({esc: [] for esc in escalas})

        coescalas = []
        for f in doc:
            coescalas_f = {ll for ll, v in f.items() if len(v) and ll in escalas}
            if not any(x == coescalas_f for x in coescalas):
                coescalas.append(coescalas_f)

        coescalas.sort(key=lambda x: len(x))

        while len(coescalas):
            siguientes = {x.pop() for x in coescalas if len(x) == 1}
            if not len(siguientes):
                raise ValueError(_('Parece que hay un error con el fuente de información regional.'))
            órden.append(sorted(list(siguientes)) if len(siguientes) > 1 else siguientes.pop())
            for cn in coescalas.copy():
                cn.difference_update(siguientes)
                if not len(cn):
                    coescalas.remove(cn)

        símismo.orden_jerárquico.extend(órden)

        for f in doc:
            cód = f[col_cód]
            escala = None
            for esc in órden[::-1]:
                if isinstance(esc, str):
                    if len(f[esc]):
                        escala = esc
                else:
                    for nv_ig in esc:
                        if len(f[nv_ig]):
                            escala = nv_ig
                if escala is not None:
                    break

            nombre = f[escala]

            símismo.cód_a_lugar[cód] = {
                'escala': escala,
                'nombre': nombre,
                'en': {ll: v for ll, v in f.items() if ll in escalas and len(v) and ll != escala}
            }
            símismo.info_geog[escala].append(cód)

        for cód, dic in símismo.cód_a_lugar.items():
            for esc, lg_en in dic['en'].items():
                if lg_en not in símismo.cód_a_lugar:
                    dic['en'][esc] = símismo.nombre_a_cód(nombre=lg_en, escala=esc)

    def obt_lugares_en(símismo, en=None, escala=None):
        """

        Parameters
        ----------
        en : str | int
        escala : str

        Returns
        -------

        """

        en = símismo._validar_código_lugar(en)
        if escala is None and en is not None:
            return [en]
        escala = símismo._validar_escala(escala, en)

        if en is not None:
            escl_en = símismo.cód_a_lugar[en]['escala']
            return [x for x, d in símismo.cód_a_lugar.items()
                    if d['escala'].lower() == escala
                    and escl_en in d['en'] and d['en'][escl_en] == en]
        else:
            return [x for x, d in símismo.cód_a_lugar.items()
                    if d['escala'] == escala]

    def obt_todos_lugares_en(símismo, en=None):
        en = símismo._validar_código_lugar(en)
        if en is not None:
            escl_en = símismo.cód_a_lugar[en]['escala']
            sub_lgs = [x for x, d in símismo.cód_a_lugar.items() if escl_en in d['en'] and d['en'][escl_en] == en]
            for s_lg in sub_lgs.copy():
                sub_lgs += símismo.obt_todos_lugares_en(en=s_lg)
            return list(set(sub_lgs))
        else:
            return list(símismo.cód_a_lugar)

    def _validar_orden_jerárquico(símismo, escala, orden):

        if orden is None:
            orden = [x if isinstance(x, str) else escala if escala in x else x[0] for x in símismo.orden_jerárquico]
            hasta = orden.index(escala)
            orden = orden[:hasta + 1]

        elif isinstance(orden, str):
            orden = [orden]

        if orden[0] != escala:
            orden = [escala] + orden

        return orden

    def obt_jerarquía(símismo, en=None, escala=None, orden_jerárquico=None):
        escala = símismo._validar_escala(escala, en)
        en = símismo._validar_código_lugar(en)

        orden_jerárquico = símismo._validar_orden_jerárquico(escala, orden_jerárquico)

        lugares = símismo.obt_lugares_en(en=en, escala=escala)

        jerarquía = {}

        cód_a_lugar = símismo.cód_a_lugar

        def agr_a_jrq(lgr):
            if lgr not in jerarquía:
                for esc in orden_jerárquico[::-1]:
                    try:
                        pariente = cód_a_lugar[lgr]['en'][esc]
                        jerarquía[lgr] = pariente
                        agr_a_jrq(pariente)
                        return
                    except KeyError:
                        pass

                jerarquía[lgr] = None

        for lg in lugares:
            agr_a_jrq(lg)

        return jerarquía

    def obt_escala_región(símismo, cód):

        cód = símismo._validar_código_lugar(cód)
        return símismo.cód_a_lugar[cód]['escala']

    def nombre_a_cód(símismo, nombre, escala=None):

        nombre = nombre.lower()

        if escala is not None:
            escala = símismo._validar_escala(escala)

        for cód, dic in símismo.cód_a_lugar.items():
            if dic['nombre'].lower() == nombre:
                if escala is None:
                    return cód
                elif dic['escala'].lower() == escala.lower():
                    return cód

        raise ValueError(_('No encontramos región correspondiendo a "{}"').format(nombre))

    def _validar_escala(símismo, escala, en=None):
        if escala is None:
            if en is None:
                escala = símismo.orden_jerárquico[-1]
                if isinstance(escala, list):
                    escala = escala[0]
            else:
                escala = símismo.obt_escala_región(en)
        else:
            if escala.lower() not in [x.lower() for x in símismo.escalas]:
                raise ValueError(_('La escala "{esc}" no existe en la geografía de "{geog}"')
                                 .format(esc=escala, geog=símismo.nombre))

        return escala.lower()

    def _validar_código_lugar(símismo, cód):

        if cód is None:
            return None

        cód = str(cód)

        if cód not in símismo.cód_a_lugar:
            raise ValueError(_('El código "{cód}" no existe en la geografía actual de "{geog}".')
                             .format(cód=cód, geog=símismo.nombre))
        return cód

    def __str__(símismo):
        return símismo.nombre

    def dibujar(símismo, archivo, valores=None, ids=None, alpha=1, título=None, unidades=None, colores=None,
                escala_num=None, clr_bar_dic=None, fst_cut=None, snd_cut=None, n_bin=None):
        """
        Dibuja la Geografía.

        :param archivo: Dónde hay que guardar el mapa.
        :type archivo: str
        :param valores: Los valores para dibujar.
        :type valores: np.ndarray | dict
        :param título: El título del mapa.
        :type título: str
        :param unidades: Las unidades de los valores.
        :type unidades: str
        :param colores: Los colores para dibujar.
        :type colores: str | list | tuple | int
        :param escala_num: La escala numérica para los colores.
        :type escala_num: tuple

        """

        if colores is None:
            colores = ['#FF6666', '#FFCC66', '#00CC66']
        if colores == -1:
            colores = ['#00CC66', '#FFCC66', '#FF6666']

        if isinstance(colores, str):
            colores = ['#FFFFFF', colores]
        elif isinstance(colores, tuple):
            colores = list(colores)

        fig = Figura()
        TelaFigura(fig)
        ejes = fig.add_subplot(111)
        ejes.set_aspect('equal')

        if valores is not None:

            if isinstance(valores, np.ndarray):
                if ids is None:
                    try:
                        escls_ids = next(x for x, v in símismo.info_geog.items() if len(v) == len(valores))
                        ids = símismo.info_geog[escls_ids]

                    except StopIteration:
                        escls_ids = next(x for x, d in símismo.formas_reg.items() if len(d['ids']) == len(valores))
                        ids = [str(x) for x in símismo.formas_reg[escls_ids]['ids']]

                    escls_ids = [escls_ids]

                else:
                    try:
                        escls_ids = set(símismo.obt_escala_región(id_) for id_ in ids)
                    except ValueError:
                        escls_ids = [next(
                            x for x, d in símismo.formas_reg.items() if all(str(i) in d['ids'] for i in ids)
                        )]
                    if valores.shape[0] != len(ids):
                        raise ValueError
                dic_valores = {id_: valores[í] for í, id_ in enumerate(ids)}

            else:
                escls_ids = set(símismo.obt_escala_región(id_) for id_ in valores)
                dic_valores = valores

            if len(escls_ids) != 1:
                raise ValueError(
                    _('Todos los códigos de lugares en `ids` deben corresponder a la misma escala'
                      '\nespacial de la geografía, pero los tuyos tienen una mezcla de las escalas'
                      '\nsiguientes:'
                      '\n\t{}').format(', '.join(escls_ids))
                )
            d_frm_reg = símismo.formas_reg[list(escls_ids)[0]]

            #    for escala, d_reg in símismo.formas_reg.items():
            #        if len(d_reg['orden_regs']) == valores.shape[0]:
            #            d_frm_reg = d_reg
            #            continue
            #
            #    if d_frm_reg is None:
            #        raise ValueError(_('El número de regiones en los datos no concuerdan con la geografía del lugar.'))

            regiones = d_frm_reg['af']
            ids_frm = d_frm_reg['ids'].astype(str)

            ids_faltan_frm = [x for x in dic_valores if x not in ids_frm]
            if len(ids_faltan_frm):
                avisar(_('Las regiones siguientes no se encuentran en el fuente de forma y por lo tanto'
                         '\nno se podrán dibujar: "{}"').format(', '.join(ids_faltan_frm)))
                dic_valores = {ll: v for ll, v in dic_valores.items() if ll not in ids_faltan_frm}

            vec_valores = np.array([dic_valores[x] if x in dic_valores else np.nan for x in ids_frm])
            if isinstance(alpha, dict):
                alphas = np.array([alpha[x] if x in alpha else 0 for x in ids_frm])
            elif isinstance(alpha, (list, np.ndarray)):
                alphas = np.array([alpha[ids.index(id_)] for id_ in ids_frm])
            else:
                alphas = np.full(ids_frm.shape, alpha)

            # n_regiones = len(regiones.shapes())
            # if len(valores) != n_regiones:
            #     raise ValueError(_('El número de regiones no corresponde con el tamaño de los valores.'))

            if escala_num is None:
                escala_num = (np.nanmin(vec_valores), np.nanmax(vec_valores))

            if escala_num[1] - escala_num[0] == 0:
                vals_norm = vec_valores - escala_num[0]
            else:
                vals_norm = (vec_valores - escala_num[0]) / (escala_num[1] - escala_num[0])

            if escala_num[0] == escala_num[1]:
                norm = colors.Normalize(vmin=0, vmax=0.01)
            else:
                norm = colors.Normalize(vmin=escala_num[0], vmax=escala_num[-1])

            # d_clrs = _gen_d_mapacolores(colores=colores)

            # mapa_color = colors.LinearSegmentedColormap('mapa_color', d_clrs)
            # cpick = cm.ScalarMappable(norm=norm, cmap=mapa_color)
            # cpick.set_array([])
            #
            # v_cols = mapa_color(vals_norm)
            # v_cols[np.isnan(vals_norm)] = 1
            # _dibujar_shp(ejes=ejes, frm=regiones, colores=v_cols, alpha=alphas)

            # if unidades is not None:
            #     fig.colorbar(cpick, label=unidades)
            # else:
            #     fig.colorbar(cpick)

            def _cpick(norm, d_clrs=None, colores=colores):
                if d_clrs is None:
                    d_clrs = _gen_d_mapacolores(colores, maxi=None)
                    mapa_color = colors.LinearSegmentedColormap('mapa_color', d_clrs, N=n_bin)
                else:
                    mapa_color = colors.LinearSegmentedColormap('mapa_color', d_clrs)

                v_cols = mapa_color(vals_norm)
                v_cols[np.isnan(vals_norm)] = 1

                _dibujar_shp(ejes=ejes, frm=regiones, colores=v_cols, alpha=alphas)

                # if midpoint and top is not None:
                #     maxi=escala_num[-1]
                #     bottom_norm = (0, int(midpoint / norm[-1] * 255))
                #     top_norm = (int(top / norm[-1] * 255), 1)
                #     bottom = cm.get_cmap('Greens', bottom_norm)
                #     top = cm.get_cmap('Reds', top_norm)
                #     middle = cm.ScalarMappable(norm=[int(midpoint / norm[-1] * 255), int(top / norm[-1] * 255)],
                #                                cmap=mapa_color)
                #     cpick=np.vstack((top(np.linspace(0, 1, bottom_norm[1])),
                #                middle(np.linspace(0, 1, 255-bottom_norm[1]-top_norm[1])),
                #                bottom(np.linspace(0, 1, top_norm[1]))))
                # else:
                cpick = cm.ScalarMappable(norm=norm, cmap=mapa_color)
                cpick.set_array([])

                return cpick, v_cols

            if unidades is not None and fst_cut is not None:
                if np.round(escala_num[1], 3) == 0:
                    lst = list(escala_num)
                    lst[1] = 0
                    escala_num = tuple(lst)
                if snd_cut > escala_num[1] > fst_cut:
                    d_clrs = _gen_d_mapacolores(colores, clr_bar_dic=clr_bar_dic, fst_cut=fst_cut, snd_cut=None,
                                                maxi=escala_num[-1])
                    cbar = fig.colorbar(_cpick(norm, d_clrs)[0], label=unidades,
                                        ticks=[0, fst_cut, escala_num[1]])
                    cbar.set_label(label=unidades, labelpad=-50, y=0.45)
                    cbar.ax.set_yticklabels(['0', f'Screening Threshold, {fst_cut}',
                                             f'maxi_val, {np.round(escala_num[1], 6)}'], fontsize=5)

                if escala_num[1] > snd_cut:
                    d_clrs = _gen_d_mapacolores(colores, clr_bar_dic=clr_bar_dic, fst_cut=fst_cut, snd_cut=snd_cut,
                                                maxi=escala_num[-1])
                    cbar = fig.colorbar(_cpick(norm, d_clrs)[0], label=unidades,
                                        ticks=[0, fst_cut, snd_cut-0.1, escala_num[1]])
                    cbar.set_label(label=unidades, labelpad=-50, y=0.45)
                    cbar.ax.set_yticklabels(
                        ['0', f'Screnning Threshold, 0.1', '+8',
                         f'High sensitivity zone'], fontsize=5)

                elif escala_num[1] == escala_num[0]:
                    d_clrs = _gen_d_mapacolores(colores=colores[:2], maxi=None)
                    fig.colorbar(_cpick(norm, d_clrs)[0], label=unidades, ticks=[0])

                elif escala_num[1] < fst_cut:
                    d_clrs = _gen_d_mapacolores(colores, clr_bar_dic=clr_bar_dic, fst_cut=None, snd_cut=None,
                                                maxi=escala_num[-1])
                    cbar = fig.colorbar(_cpick(norm, d_clrs)[0], label=unidades,
                                        ticks=[0, escala_num[1], fst_cut])
                    cbar.set_label(label=unidades, labelpad=-45, y=0.45)
                    cbar.ax.set_yticklabels(
                        ['0', f'maxi_val, {np.round(escala_num[1], 3)}', f'Screening Threshold, {fst_cut}'],
                        fontsize=6)
            elif n_bin is not None:
                cbar = fig.colorbar(_cpick(norm, d_clrs=None)[0], label=unidades,
                                    ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                cbar.ax.set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
                                        fontsize=6)
            else:
                fig.colorbar(_cpick(norm, d_clrs=None)[0])

        for nombre, d_obj in símismo.formas.items():

            tipo = d_obj['tipo']

            for a in ['color', 'llenar', 'alpha']:
                if d_obj[a] is None:
                    d_obj[a] = _formatos_auto(a, tipo)

            color = d_obj['color']
            llenar = d_obj['llenar']
            alpha = d_obj['alpha']

            _dibujar_shp(ejes=ejes, frm=d_obj['obj'], colores=color, alpha=alpha, llenar=llenar)

        if título is not None:
            ejes.set_title(título)

        fig.savefig(archivo, dpi=500)

    def en_región(símismo, cód_lugar, cód_región):
        """
        Verifica si un lugar está en una región especificada.

        Parameters
        ----------
        cód_lugar : str
            El lugar de interés.
        cód_región : str
            La región en la cual queremos saber si se encuentra el lugar.

        Returns
        -------
        bool
            Si el lugar está en la región especificada.
        """
        if cód_lugar == cód_región:
            return True
        escala_reg = símismo.obt_escala_región(cód_región)
        dic_en = símismo.cód_a_lugar[cód_lugar]['en']
        return escala_reg in dic_en and dic_en[escala_reg] == cód_región

    def dibujar_corrida(símismo, resultados, l_vars=None, directorio='./', i_paso=None, colores=None, escala=None):

        if all(v in resultados for v in l_vars):
            lugares = None
        else:
            lugares = list(resultados)
            faltan_de_geog = [lg for lg in lugares if lg not in símismo.cód_a_lugar]
            if len(faltan_de_geog):
                avisar(_('\nLos lugares siguientes no están en la geografía y por tanto no se podrán dibujar:'
                         '\n\t{}').format(', '.join(faltan_de_geog)))

        if isinstance(l_vars, str):
            l_vars = [l_vars]

        for var in l_vars:

            if lugares is None:
                res_var = resultados[var]
                n_obs = res_var.shape[-1]
                if escala is None:
                    escala = np.min(res_var), np.max(res_var)
            else:
                res_var = {lg: resultados[lg][var] for lg in lugares}
                n_obs = len(list(resultados.values())[0]['n']) + 1
                if escala is None:
                    escala = res_var[var].min(), res_var[var].max()

            if isinstance(i_paso, tuple):
                i_paso = list(i_paso)

            if i_paso is None:
                i_paso = [0, n_obs]
            if isinstance(i_paso, int):
                i_paso = [i_paso, i_paso + 1]
            if i_paso[0] is None:
                i_paso[0] = 0
            if i_paso[1] is None:
                i_paso[1] = n_obs

            # Incluir el nombre del variable en el directorio, si no es que ya esté allí.
            nombre_var = valid_nombre_arch(var)
            if os.path.split(directorio)[1] != nombre_var:
                directorio = os.path.join(directorio, nombre_var)

            # Crear el directorio, si no existe ya. Sino borrarlo.
            if os.path.exists(directorio):
                shutil.rmtree(directorio)
            os.makedirs(directorio)

            for i in range(*i_paso):
                if lugares is None:
                    valores = res_var[..., i]
                else:
                    valores = {lg: res_var[i] for lg in res_var}
                archivo_í = os.path.join(directorio, '{}, {}'.format(nombre_var, i))
                símismo.dibujar(archivo=archivo_í, valores=valores, título=var, colores=colores, escala_num=escala)


def _dibujar_shp(ejes, frm, colores, orden=None, alpha=1.0, llenar=True):
    """
    Dibujar una forma geográfica.

    Basado en código de Chris Halvin: https://github.com/chrishavlin/learning_shapefiles/tree/master/src

    :param ejes: Los ejes de Matplotlib.
    :type ejes: matplotlib.axes._subplots.Axes

    :param frm: La forma.
    :type frm: sf.Reader

    :param colores: Los colores para dibujar.
    :type colores: str | list[str] | np.ndarray

    :param orden: El orden de las partes de la forma, relativo al orden de las colores.
    :type orden: np.ndarray | list

    :param alpha: La transparencia
    :type alpha: float | int | np.ndarray

    :param llenar: Si hay que llenar la forma, o simplemente dibujar los contornos.
    :type llenar: bool
    """

    if orden is not None:
        regiones = [
            x for o, x in sorted([t for t in zip(orden, frm.shapes()) if not np.isnan(t[0])], key=lambda x: x[0])
        ]
    else:
        regiones = frm.shapes()

    n_formas = len(regiones)

    if isinstance(colores, str) or isinstance(colores, tuple):
        colores = [colores] * n_formas
    else:
        if len(colores) != n_formas:
            raise ValueError

    if not isinstance(alpha, (list, np.ndarray)):
        alpha = np.full_like(regiones, alpha)

    for forma, col, alph in zip(regiones, colores, alpha):
        n_puntos = len(forma.points)
        n_partes = len(forma.parts)

        if n_partes == 1:
            x_lon = np.zeros((len(forma.points), 1))
            y_lat = np.zeros((len(forma.points), 1))
            for ip in range(len(forma.points)):
                x_lon[ip] = forma.points[ip][0]
                y_lat[ip] = forma.points[ip][1]
            if llenar:
                ejes.fill(x_lon, y_lat, color=col, alpha=alph)
            else:
                ejes.plot(x_lon, y_lat, color=col, alpha=alph)

        else:
            for ip in range(n_partes):  # Para cada parte del imagen
                i0 = forma.parts[ip]
                if ip < n_partes - 1:
                    i1 = forma.parts[ip + 1] - 1
                else:
                    i1 = n_puntos

                seg = forma.points[i0:i1 + 1]
                x_lon = np.zeros((len(seg), 1))
                y_lat = np.zeros((len(seg), 1))
                for i in range(len(seg)):
                    x_lon[i] = seg[i][0]
                    y_lat[i] = seg[i][1]

                if llenar:
                    ejes.fill(x_lon, y_lat, color=col, alpha=alph)
                else:
                    ejes.plot(x_lon, y_lat, color=col, alpha=alph)


def _hex_a_rva(hx):
    """
    Convierte colores RVA a Hex.

    :param hx: El valor hex.
    :type hx: str
    :return: El valor rva.
    :rtype: tuple
    """
    return tuple(int(hx.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def _gen_clrbar_dic(fst_cut, snd_cut, maxi, first_set, second_set, third_set):
    red = []
    green = []
    blue = []

    first_rva = [_hex_a_rva(x) for x in first_set]
    second_rva = [_hex_a_rva(x) for x in second_set]
    third_rva = [_hex_a_rva(x) for x in third_set]

    if fst_cut is None and snd_cut is None:
        f_cut = 1
        for i in range(len(first_rva)):
            red.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][0] / 255, first_rva[i][0] / 255))
            green.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][1] / 255, first_rva[i][1] / 255))
            blue.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][2] / 255, first_rva[i][2] / 255))

    elif snd_cut is None and fst_cut is not None:
        f_cut = fst_cut / maxi
        for i in range(len(first_rva)):
            red.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][0] / 255, first_rva[i][0] / 255))
            green.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][1] / 255, first_rva[i][1] / 255))
            blue.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][2] / 255, first_rva[i][2] / 255))
            last_r = first_rva[i][0] / 255
            last_g = first_rva[i][1] / 255
            last_b = first_rva[i][2] / 255  ## check it

        for i in range(len(second_rva)):
            if i == 0:
                red.append(
                    (f_cut + (i) * (1 - f_cut) / (len(second_rva) - 1), last_r, second_rva[i][0] / 255))
                green.append(
                    (f_cut + (i) * (1 - f_cut) / (len(second_rva) - 1), last_g, second_rva[i][1] / 255))
                blue.append(
                    (f_cut + (i) * (1 - f_cut) / (len(second_rva) - 1), last_b, second_rva[i][2] / 255))
            elif i == len(second_rva) - 1:
                red.append((1, second_rva[i][0] / 255,second_rva[i][0] / 255))
                green.append((1, second_rva[i][1] / 255,second_rva[i][1] / 255))
                blue.append((1, second_rva[i][2] / 255,second_rva[i][2] / 255))
            else:
                red.append(
                    (f_cut + (i) * (1 - f_cut) / (len(second_rva) - 1), second_rva[i][0] / 255,
                     second_rva[i][0] / 255))
                green.append(
                    (f_cut + (i) * (1 - f_cut) / (len(second_rva) - 1), second_rva[i][1] / 255,
                     second_rva[i][1] / 255))
                blue.append(
                    (f_cut + (i) * (1 - f_cut) / (len(second_rva) - 1), second_rva[i][2] / 255,
                     second_rva[i][2] / 255))

    elif snd_cut is not None:
        f_cut = fst_cut / maxi
        s_cut = snd_cut / maxi
        for i in range(len(first_rva)):
            red.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][0] / 255, first_rva[i][0] / 255))
            green.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][1] / 255, first_rva[i][1] / 255))
            blue.append(((i) * f_cut / (len(first_rva) - 1), first_rva[i][2] / 255, first_rva[i][2] / 255))
            last_r = first_rva[i][0] / 255
            last_g = first_rva[i][1] / 255
            last_b = first_rva[i][2] / 255

        for i in range(len(second_rva)):
            if i == 0:
                red.append((f_cut + (i) * (s_cut - f_cut) / (len(second_rva) - 1), last_r,
                            second_rva[i][0] / 255))
                green.append((f_cut + (i) * (s_cut - f_cut) / (len(second_rva) - 1), last_g,
                              second_rva[i][1] / 255))
                blue.append((f_cut + (i) * (s_cut - f_cut) / (len(second_rva) - 1), last_b,
                             second_rva[i][2] / 255))
            else:
                red.append((f_cut + (i) * (s_cut - f_cut) / (len(second_rva) - 1), second_rva[i][0] / 255,
                            second_rva[i][0] / 255))
                green.append((f_cut + (i) * (s_cut - f_cut) / (len(second_rva) - 1), second_rva[i][1] / 255,
                              second_rva[i][1] / 255))
                blue.append((f_cut + (i) * (s_cut - f_cut) / (len(second_rva) - 1), second_rva[i][2] / 255,
                             second_rva[i][2] / 255))
                last_r = second_rva[i][0] / 255
                last_g = second_rva[i][1] / 255
                last_b = second_rva[i][2] / 255

        for i in range(len(third_rva)):
            if i == 0:
                red.append(
                    (s_cut + (i) * (1 - s_cut) / (len(third_rva) - 1), last_r, third_rva[i][0] / 255))
                green.append(
                    (s_cut + (i) * (1 - s_cut) / (len(third_rva) - 1), last_g, third_rva[i][1] / 255))
                blue.append(
                    (s_cut + (i) * (1 - s_cut) / (len(third_rva) - 1), last_b, third_rva[i][2] / 255))
            elif i == len(third_rva) - 1:
                red.append((1, third_rva[i][0] / 255,third_rva[i][0] / 255))
                green.append((1, third_rva[i][1] / 255,third_rva[i][1] / 255))
                blue.append((1, third_rva[i][2] / 255,third_rva[i][2] / 255))
            else:
                red.append(
                    (s_cut + (i) * (1 - s_cut) / (len(third_rva) - 1), third_rva[i][0] / 255,
                     third_rva[i][0] / 255))
                green.append(
                    (s_cut + (i) * (1 - s_cut) / (len(third_rva) - 1), third_rva[i][1] / 255,
                     third_rva[i][1] / 255))
                blue.append(
                    (s_cut + (i) * (1 - s_cut) / (len(third_rva) - 1), third_rva[i][2] / 255,
                     third_rva[i][2] / 255))

    return {
        'red': tuple(red),
        'green': tuple(green),
        'blue': tuple(blue)
    }


def _gen_d_mapacolores(colores, maxi, clr_bar_dic=None, fst_cut=None, snd_cut=None):
    """
    Genera un diccionario de mapa de color para MatPlotLib.

    :param colores: Una lista de colores
    :type colores: list
    :return: Un diccionario para MatPlotLib
    :rtype: dict
    """

    clrs_rva = [_hex_a_rva(x) for x in colores]
    n_colores = len(colores)

    if clr_bar_dic is None:
        dic_c = {'red': tuple((round(i / (n_colores - 1), 2), clrs_rva[i][0] / 255, clrs_rva[i][0] / 255) for i in
                              range(0, n_colores)),
                 'green': tuple(
                     (round(i / (n_colores - 1), 2), clrs_rva[i][1] / 255, clrs_rva[i][1] / 255) for i in
                     range(0, n_colores)),
                 'blue': tuple(
                     (round(i / (n_colores - 1), 2), clrs_rva[i][2] / 255, clrs_rva[i][2] / 255) for i in
                     range(0, n_colores))}
    else:
        dic_c = _gen_clrbar_dic(fst_cut=fst_cut, snd_cut=snd_cut, maxi=maxi, first_set=clr_bar_dic['green'],
                                second_set=clr_bar_dic['blue'], third_set=clr_bar_dic['red'])

    return dic_c


def _formatos_auto(a, tipo):
    """
    Formatos automáticos para objetos en mapas.

    :param a: El atributo.
    :type a: str
    :param tipo: El tipo_mod de objeto geográfico.
    :type tipo: str
    :return: El atributo automático.
    :rtype: str | bool | float
    """

    d_formatos = {
        'agua': {'color': '#A5BFDD',
                 'llenar': True,
                 'alpha': 0.5},
        'calle': {'color': '#7B7B7B',
                  'llenar': False,
                  'alpha': 1.0},
        'ciudad': {'color': '#FB9A99',
                   'llenar': True,
                   'alpha': 1.0},
        'bosque': {'color': '#33A02C',
                   'llenar': True,
                   'alpha': 0.7},
        'otro': {'color': '#FFECB3',
                 'llenar': False,
                 'alpha': 1.0}
    }

    if tipo is None or tipo not in d_formatos:
        tipo = 'otro'

    return d_formatos[tipo.lower()][a.lower()]
