import os
import re
import tempfile
from warnings import warn as avisar

import numpy as np
import scipy.stats as estad
import spotpy
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from xarray import Dataset

from tinamit.Análisis.Datos import BDtexto, gen_SuperBD, SuperBD
from tinamit.Análisis.Sens.behavior import compute_rmse, nse, superposition, predict
from tinamit.Análisis.sintaxis import Ecuación
from tinamit.Calib.ej.obs_patrón import compute_patron
from tinamit.config import _
from tinamit.cositas import cargar_json

try:
    import pymc3 as pm
except ImportError:  # pragma: sin cobertura
    pm = None

if pm is None:
    dists = None  # pragma: sin cobertura
else:
    dists = {'Beta': {'sp': estad.beta, 'pm': pm.Beta,
                      'sp_a_pm': lambda p: {'alpha': p[0], 'beta': p[1]},
                      'líms': (0, 1)},
             'Cauchy': {'sp': estad.cauchy, 'pm': pm.Cauchy,
                        'sp_a_pm': lambda p: {'alpha': p[0], 'beta': p[1]},
                        'líms': (None, None)},
             'Chi2': {'sp': estad.chi2, 'pm': pm.ChiSquared,
                      'sp_a_pm': lambda p: {'df': p[0]},
                      'líms': (0, None)},
             'Exponencial': {'sp': estad.expon, 'pm': pm.Exponential,
                             'sp_a_pm': lambda p: {'lam': 1 / p[1]},
                             'líms': (0, None)},
             'Gamma': {'sp': estad.gamma, 'pm': pm.Gamma,
                       'sp_a_pm': lambda p: {'alpha': p[0], 'beta': 1 / p[2]},
                       'líms': (0, None)},
             'Laplace': {'sp': estad.laplace, 'pm': pm.Laplace,
                         'sp_a_pm': lambda p: {'mu': p[0], 'b': p[1]},
                         'líms': (None, None)},
             'LogNormal': {'sp': estad.lognorm, 'pm': pm.Lognormal,
                           'sp_a_pm': lambda p: {'mu': p[1], 'sd': p[2]},
                           'líms': (0, None)},
             'MitadCauchy': {'sp': estad.halfcauchy, 'pm': pm.HalfCauchy,
                             'sp_a_pm': lambda p: {'beta': p[1]},
                             'líms': (0, None)},
             'MitadNormal': {'sp': estad.halfnorm, 'pm': pm.HalfNormal,
                             'sp_a_pm': lambda p: {'sd': p[1]},
                             'líms': (0, None)},
             'Normal': {'sp': estad.norm, 'pm': pm.Normal,
                        'sp_a_pm': lambda p: {'mu': p[0], 'sd': p[1]},
                        'líms': (None, None)},
             'T': {'sp': estad.t, 'pm': pm.StudentT,
                   'sp_a_pm': lambda p: {'nu': p[0], 'mu': p[1], 'sd': p[2]},
                   'líms': (None, None)},
             'Uniforme': {'sp': estad.uniform, 'pm': pm.Uniform,
                          'sp_a_pm': lambda p: {'lower': p[0], 'upper': p[0] + p[1]},
                          'líms': (0, 1)},
             'Weibull': {'sp': estad.weibull_min, 'pm': pm.Weibull,
                         'sp_a_pm': lambda p: {'alpha': p[0], 'beta': p[2]},
                         'líms': (0, None)},
             }


class CalibradorEc(object):
    """
    Un objeto para manejar la calibración de ecuaciones.
    """

    def __init__(símismo, ec, var_y=None, otras_ecs=None, corresp_vars=None, dialecto=None):
        """
        Inicializa el `Calibrador`.

        Parameters
        ----------
        ec: str
            La ecuación, en formato texto, para calibrar.
        var_y: str
            El nombre del variable y. Si no se especifica aquí, debe estar en la ecuación sí misma (por ejemplo,
            "y = a*x + b" en vez de simplemente "a*x + b").
        otras_ecs: dict[str, str]
            Un diccionario de otras ecuaciones para substituir variables en la ecuación principal.
        corresp_vars: dict[str, str]
            Un diccionario de equivalencias de nombres entre ecuación y eventual base de datos.
        """

        # Crear la ecuación.
        símismo.ec = Ecuación(ec, nombre=var_y, otras_ecs=otras_ecs, nombres_equiv=corresp_vars, dialecto=dialecto)

        # Asegurarse que se especificó el variable y.
        if símismo.ec.nombre is None:
            raise ValueError(_('Debes especificar el nombre del variable dependiente, o con el parámetro'
                               '`var_y`, o directamente en la ecuación, por ejemplo: "y = a*x ..."'))

    def calibrar(símismo, bd_datos, paráms=None, líms_paráms=None, método=None, binario=False, geog=None, en=None,
                 escala=None, jrq=None, ops_método=None, no_recalc=None):
        """
        Calibra la ecuación, según datos, y, opcionalmente, una geografía espacial.

        Parameters
        ----------
        bd_datos : SuperBD
            La base de datos. Los nombres de las columnas deben coincidir con los nombres de los variables en la
            ecuación.
        paráms : list[str]
            La lista de los parámetros para calibrar. Si es ``None``, se tomarán los variables que no
            están en `bd_datos`.
        líms_paráms : dict[str, tuple]
            Un diccionario con los límites teoréticos de los parámetros.
        método : {'optimizar', 'inferencia bayesiana'}
            El método para emplear para la calibración.
        binario : bool
            Si la ecuación es binaria o no.
        en : str | int
            El código del lugar donde quieres calibrar esta ecuación. Aplica únicamente si `bd_datos` está
            conectada a un objeto :class:`Geografía`.
        escala : str
            La escala geográfica a la cual quieres calibrar el modelo. Aplica únicamente si `bd_datos` está
            conectada a un objeto :class:`Geografía`.
        ops_método : dict
            Un diccionario de opciones adicionales a pasar directamente al algoritmo de calibración. Por ejemplo,
            puedes pasar argumentos para pymc3.Model.sample() aquí. También puedes especificar opciones
            específicas a Tinamït:
                - 'mod_jerárquico': bool. Determina si empleamos un modelo bayesiano jerárquico para la geografía.

        Returns
        -------
        dict
            La calibración completada.
        """

        # Poner diccionario vacío si `ops_método` no existe.
        if ops_método is None:
            ops_método = {}
        if no_recalc is None:
            no_recalc = {}

        # Leer los variables de la ecuación.
        vars_ec = símismo.ec.variables()
        var_y = símismo.ec.nombre

        # Asegurarse que tenemos observaciones para el variable y.
        if var_y not in bd_datos.variables:
            raise ValueError(_('El variable "{}" no parece existir en la base de datos.').format(var_y))

        # Adivinar los parámetros si ne se especificaron.
        if paráms is None:
            # Todos los variables para cuales no tenemos observaciones.
            paráms = [x for x in vars_ec if x not in bd_datos.variables]

        # Los variables x, por definición, son los que no son parámetros.
        vars_x = [v for v in vars_ec if v not in paráms]

        # Generar los límites de parámetros, si necesario.
        if líms_paráms is None:
            líms_paráms = {}

        líms_paráms_final = {}
        for p in paráms:
            # Para cada parámetro...
            if p not in líms_paráms or líms_paráms[p] is None:
                # Aplicar y formatear límites para parámetros sin límites
                líms_paráms_final[p] = (None, None)
            else:
                # Pasar los límites que sí se especificaron.
                if len(líms_paráms[p]) == 2:
                    líms_paráms_final[p] = líms_paráms[p]
                else:
                    # Límites deben se tuplas con únicamente 2 elementos.
                    raise ValueError(
                        _('Límite "{}" inválido. Los límites de parámetros deben tener dos elementos: '
                          '(mínimo, máximo). Utilizar ``None`` para ± infinidad: (None, 10); (0, None).'
                          ).format(líms_paráms[p])
                    )

        # Aplicar el método automáticamente, si necesario.
        if método is None:
            if pm is not None:
                método = 'inferencia bayesiana'
            else:
                método = 'optimizar'
        else:
            método = método.lower()

        # Para calibración bayesiana, emplear modelos jerárquicos si no se especificó el contrario.
        if método == 'inferencia bayesiana':
            if 'mod_jerárquico' not in ops_método or ops_método['mod_jerárquico'] is None:
                ops_método['mod_jerárquico'] = True
            mod_jerárquico = ops_método.pop('mod_jerárquico')
        else:
            mod_jerárquico = False

        # Intentar obtener información geográfica, si posible.
        if geog is not None:
            lugares = geog.obt_lugares_en(en, escala=escala)
            jerarquía = geog.obt_jerarquía(en, escala=escala, orden_jerárquico=jrq)
        else:
            if en is not None or escala is not None:
                raise ValueError(_('Debes especificar una geografía en `geog` para emplear `en` o `escala`.'))
            lugares = jerarquía = None

        if all(p in no_recalc and all(lg in no_recalc[p] for lg in lugares) for p in paráms):
            return

        # Ahora, por fin hacer la calibración.
        if método == 'inferencia bayesiana':
            return símismo._calibrar_bayesiana(
                ec=símismo.ec, var_y=var_y, vars_x=vars_x, líms_paráms=líms_paráms_final, binario=binario,
                ops_método=ops_método, bd_datos=bd_datos, lugares=lugares, jerarquía=jerarquía, geog=geog,
                mod_jerárquico=mod_jerárquico
            )
        elif método == 'optimizar':
            return símismo._calibrar_optim(
                ec=símismo.ec, var_y=var_y, vars_x=vars_x, líms_paráms=líms_paráms_final,
                ops_método=ops_método, bd_datos=bd_datos, lugares=lugares, jerarquía=jerarquía, geog=geog
            )
        else:
            raise ValueError(_('Método de calibración "{}" no reconocido.').format(método))

    @staticmethod
    def _calibrar_bayesiana(ec, var_y, vars_x, líms_paráms, binario, bd_datos, lugares, jerarquía, geog,
                            mod_jerárquico, ops_método):
        """
        Efectua la calibración bayesiana.

        Parameters
        ----------
        ec: Ecuación
            La ecuación para calibrar.
        var_y: str
            El nombre del variable y.
        vars_x: list[str]
            Los nombres de los variables x.
        líms_paráms: dict[str, tuple]
            El diccionario con límites de parámetros.
        binario: bool
            Si el modelo es binario o no.
        bd_datos: SuperBD
            La base de datos observados.
        lugares: list
            La lista de lugares en los cuales calibrar.
        jerarquía: dict
            La jerarquía de los lugares.
        mod_jerárquico: bool
            Si implementamos un único modelo jerárquico o no.
        ops_método: dict
            Opciones adicionales a pasar a la función de calibración bayesiana de PyMC3.

        Returns
        -------
        dict
            La calibración de los parámetros.

        """

        # Si no tenemos PyMC3, no podemos hacer inferencia bayesiana.
        if pm is None:
            raise ImportError(_('Debes instalar PyMC3 para poder hacer calibraciones con inferencia bayesiana.'))

        # Generar los datos
        paráms = list(líms_paráms)
        l_vars = vars_x + [var_y]
        obs = bd_datos.obt_datos(l_vars=l_vars, excl_faltan=True, interpolar=False)

        # Calibrar según la situación
        if lugares is None:
            # Si no hay lugares, generar y calibrar el modelo de una vez.
            mod_bayes = ec.gen_mod_bayes(
                líms_paráms=líms_paráms, obs_x=obs[vars_x], obs_y=obs[var_y],
                binario=binario, aprioris=None, nv_jerarquía=None
            )

            resultados = _calibrar_mod_bayes(mod_bayes, paráms=paráms, ops=ops_método)

        else:
            # Si hay distribución geográfica, es un poco más complicado.
            if mod_jerárquico:
                # Si estamos implementando un modelo jerárquico...

                # Primero, crear una lista de las relaciones jerárquicas, el cual se necesita para crear el modelo
                # jerárquico bayes.

                def _gen_nv_jerarquía(jrq, egr=None, nv_ant=None):
                    """

                    Parameters
                    ----------
                    jrq: dict
                        La jerarquía.
                    egr: list
                        Parámetro para la recursión.
                    nv_ant: list
                        Un lista de los nombres del nivel superior en la jerarquía. Parámetro de recursión.

                    Returns
                    -------
                    list:
                    """

                    # Empezar con el primer nivel
                    if nv_ant is None:
                        nv_ant = [None]

                    # Empezar con egresos vacíos
                    if egr is None:
                        egr = []

                    nv_act = [x for x in jrq if jrq[x] in nv_ant]

                    if len(nv_act):
                        nv_act += [x for x in nv_ant if x in lugares and x not in nv_act]

                        # Agregar a los egresos
                        egr.append(nv_act)

                        # Recursarr en los niveles inferiores
                        _gen_nv_jerarquía(jrq, egr=egr, nv_ant=nv_act)

                    # Devolver el resultado
                    return egr

                # Generar la lista de relaciones jerárquicas
                nv_jerarquía = _gen_nv_jerarquía(jerarquía)

                nv_jerarquía.insert(0, [None])

                for í, nv in list(enumerate(nv_jerarquía))[::-1]:

                    if í == (len(nv_jerarquía) - 1):

                        nv[:] = [x for x in nv
                                 if len(obs.where(obs['lugar'].isin(geog.obt_lugares_en(x) + [x]), drop=True)['n'])
                                 ]  # para hacer: accelerar con .values()
                    else:

                        nv[:] = [x for x in nv if x in [jerarquía[y] for y in nv_jerarquía[í + 1]]]

                í_nv_jerarquía = [np.array([nv_jerarquía[í - 1].index(jerarquía[x]) for x in y])
                                  for í, y in list(enumerate(nv_jerarquía))[:0:-1]]
                obs = obs.where(obs['lugar'].isin(nv_jerarquía[-1]), drop=True)
                í_nv_jerarquía.insert(0, np.array([nv_jerarquía[-1].index(x) for x in obs['lugar'].values.tolist()]))

                # Generar el modelo bayes
                mod_bayes_jrq = ec.gen_mod_bayes(
                    líms_paráms=líms_paráms, obs_x=obs[vars_x], obs_y=obs[var_y],
                    aprioris=None, binario=binario, nv_jerarquía=í_nv_jerarquía[::-1],
                )
                var_res_lugares = {}
                for lg in lugares:
                    if lg in nv_jerarquía[-1]:
                        var_res_lugares[lg] = nv_jerarquía[-1].index(lg)
                    else:
                        for í, nv in enumerate(nv_jerarquía[1::-1]):
                            id_nv = lg
                            while id_nv is not None:
                                id_nv = jerarquía[id_nv]
                                if id_nv in nv:
                                    var_res_lugares[lg] = (í + 1, nv.index(id_nv))
                                    break
                            if lg in var_res_lugares:
                                break

                prms_extras = list({
                    'mu_{p}_nv_{í}'.format(p=p, í=x[0]) for x in set(var_res_lugares.values()) if isinstance(x, tuple)
                    for p in paráms
                })

                # Calibrar
                res_calib = _calibrar_mod_bayes(mod_bayes_jrq, paráms=paráms + prms_extras, ops=ops_método)

                # Formatear los resultados
                resultados = {}
                for lg in lugares:
                    ubic_res = var_res_lugares[lg]

                    if isinstance(ubic_res, int):
                        resultados[lg] = {p: {ll: v[..., ubic_res] for ll, v in res_calib[p].items()} for p in paráms}
                    else:
                        nv, í = ubic_res
                        resultados[lg] = {
                            p: {ll: v[..., í] for ll, v in res_calib['mu_{}_nv_{}'.format(p, nv)].items()}
                            for p in paráms
                        }

            else:
                # Si no estamos haciendo un único modelo jerárquico, hay que calibrar cada lugar individualmente.

                # Efectuar las calibraciones para todos los lugares.
                resultados = {}
                for lg in lugares:
                    lgs_potenciales = geog.obt_todos_lugares_en(lg)
                    obs_lg = obs.where(obs['lugar'].isin(lgs_potenciales + [lg]), drop=True)
                    if len(obs_lg['n']):
                        mod_bayes = ec.gen_mod_bayes(
                            líms_paráms=líms_paráms, obs_x=obs_lg[vars_x], obs_y=obs_lg[var_y],
                            binario=binario, aprioris=None, nv_jerarquía=None
                        )
                        resultados[lg] = _calibrar_mod_bayes(
                            mod_bayes=mod_bayes, paráms=paráms, ops=ops_método, obs=obs_lg
                        )
                    else:
                        resultados[lg] = None

        # Devolver únicamente los lugares de interés (y no lugares de más arriba en la jerarquía).
        if lugares is not None:
            return {ll: v for ll, v in resultados.items() if ll in lugares}
        else:
            return resultados

    @staticmethod
    def _calibrar_optim(ec, var_y, vars_x, líms_paráms, bd_datos, lugares, jerarquía, geog, ops_método):
        """
        Calibra la ecuación con un método de optimización.

        Parameters
        ----------
        ec : Ecuación
            La ecuación para calibrar.
        var_y : str
            El variable y.
        vars_x : list[str]
            Los variables x.
        líms_paráms : dict[str, tuple]
            El diccionario de los límites de los parámetros.
        bd_datos : SuperBD
            La base de datos.
        lugares : list[str]
            La lista de lugares de interés.
        jerarquía : dict
            La jerarquía de los lugares.
        ops_método : dict
            Opciones para la optimización.

        Returns
        -------
        dict:
            Los parámetros calibrados.
        """

        # Generar la función dinámica Python
        paráms = list(líms_paráms)
        f_python = ec.gen_func_python(paráms=paráms)

        # Todos los variables
        l_vars = vars_x + [var_y]

        # Todas las observaciones
        obs = bd_datos.obt_datos(l_vars=l_vars, excl_faltan=True)

        # Calibrar según la situación
        if lugares is None:
            # Si no hay lugares, optimizar de una vez con todas las observaciones.
            resultados = _optimizar(f_python, líms_paráms=líms_paráms, obs_x=obs[vars_x], obs_y=obs[var_y],
                                    **ops_método)
        else:
            # Si hay lugares...

            # Una función recursiva para calibrar según la jerarquía
            def _calibrar_jerárchico_manual(lugar, jrq, clbs=None):
                """
                Una función recursiva que permite calibrar en una jerarquía, tomando optimizaciones de niveles más
                altos si no consigue datos para niveles más bajos.

                Parameters
                ----------
                lugar: str
                    El lugar en cual calibrar.
                jrq: dict
                    La jerarquía.
                clbs: dict
                    El diccionario de los resultados de la calibración. (Parámetro recursivo.)

                """

                # Para la recursión
                if clbs is None:
                    clbs = {}

                if lugar is None:
                    # Si estamos al nivel más alto de la jerarquía, tomar todos los datos.
                    obs_lg = obs
                    inic = pariente = None  # Y no tenemos ni estimos iniciales, ni región pariente
                else:
                    # Sino, tomar los datos de esta región únicamente.
                    lgs_potenciales = geog.obt_todos_lugares_en(lugar)
                    obs_lg = obs.where(obs['lugar'].isin(lgs_potenciales + [lugar]), drop=True)

                    # Intentar sacar información del nivel superior en la jerarquía
                    try:
                        pariente = jrq[lugar]  # El nivel inmediatemente superior

                        # Calibrar recursivamente si necesario
                        if pariente not in clbs:
                            _calibrar_jerárchico_manual(lugar=pariente, jrq=jrq, clbs=clbs)

                        # Tomar la calibración superior como punto inicial para facilitar la búsqueda
                        inic = [clbs[pariente][p]['val'] for p in paráms]

                    except KeyError:
                        # Error improbable con la jerarquía.
                        avisar(_('No encontramos datos para el lugar "{}", ni siguiera en su jerarquía, y por eso'
                                 'no pudimos calibrarlo.').format(lugar))
                        resultados[lugar] = {}  # Calibración vacía
                        return  # Si hubo error en la jerarquía, no hay nada más que hacer para este lugar.

                # Ahora, calibrar.
                if len(obs_lg['n']):
                    # Si tenemos observaciones, calibrar con esto.
                    resultados[lugar] = _optimizar(
                        f_python, líms_paráms=líms_paráms,
                        obs_x=obs_lg[vars_x], obs_y=obs_lg[var_y], inic=inic, **ops_método
                    )
                else:
                    # Si no tenemos observaciones, copiar la calibración del pariente
                    resultados[lugar] = clbs[pariente]

            # Calibrar para cada lugar
            resultados = {}
            for lg in lugares:
                _calibrar_jerárchico_manual(lugar=lg, jrq=jerarquía, clbs=resultados)

        # Devolver únicamente los lugares de interés.
        if lugares is not None:
            return {ll: v for ll, v in resultados.items() if ll in lugares}
        else:
            return resultados


class CalibradorMod(object):
    def __init__(símismo, mod):
        """

        Parameters
        ----------
        mod : Modelo.Modelo
        """
        símismo.mod = mod

    def calibrar(símismo, paráms, líms_paráms, bd, método, vars_obs, n_iter, guardar, corresp_vars=None, tipo_proc=None,
                 mapa_paráms=None, final_líms_paráms=None, guar_sim=False, egr_spotpy=False, warmup_period=None,
                 cls=None, obj_func=None, simluation_res=None, ind_simul=None):

        método = método.lower()
        mod = símismo.mod
        if isinstance(bd, xr.Dataset):
            obs = gen_SuperBD(bd)
        else:
            obs = _conv_xr(bd, vars_obs, warmup_period)
            t_final = len(obs['n'])

        if corresp_vars is None:
            corresp_vars = {}

        if vars_obs is None:
            vars_obs = list(obs.variables)
        vars_obs = [corresp_vars[v] if v in corresp_vars else v for v in vars_obs]

        if isinstance(bd, xr.Dataset) and tipo_proc is not None:
            obs = obs.obt_datos(vars_obs, tipo='datos', interpolar=False)[vars_obs]
            t_final = len(obs['n']) - 1
        elif tipo_proc is None:
            obs = obs.obt_datos(vars_obs, tipo='datos')[vars_obs]
            t_final = len(obs['n']) - 1

        if egr_spotpy is None:
            if método in _algs_spotpy:
                temp = tempfile.NamedTemporaryFile('w', encoding='UTF-8', prefix='CalibTinamït_')
                if tipo_proc is None:
                    mod_spotpy = ModSpotPy(mod=mod, líms_paráms=líms_paráms, obs=obs)
                else:
                    mod_spotpy = PatrónProc(mod=mod, líms_paráms=líms_paráms, obs=obs, tipo_proc=tipo_proc,
                                            mapa_paráms=mapa_paráms, comp_final_líms=final_líms_paráms,
                                            t_final=t_final, guar_sim=guar_sim, warmup_period=warmup_period, cls=cls,
                                            obj_func=obj_func, método=método, simluation_res=simluation_res, ind_simul=ind_simul)

                muestreador = _algs_spotpy[método](mod_spotpy, dbname=temp.name, dbformat='csv',
                                                   save_sim=False)  # , alt_objfun=None)

                # if final_líms_paráms is not None and método in ['dream', 'demcz', 'sceua']:
                # if método == 'dream':
                #     muestreador.sample(repetitions=n_iter, nChains=len(final_líms_paráms))
                # if método == 'sceua':
                #     muestreador.sample(n_iter, ngs=len(final_líms_paráms) * 3)
                # elif método == 'demcz':
                #     muestreador.sample(n_iter, nChains=len(final_líms_paráms))

                if método == 'fscabc':
                    if obj_func == 'aic':
                        muestreador.sample(n_iter, peps=500)
                    elif obj_func == 'mic':
                        muestreador.sample(n_iter, peps=0.5)
                    elif obj_func == 'rmse':
                        muestreador.sample(n_iter, peps=-0.6)
                    elif obj_func == 'nse':
                        muestreador.sample(n_iter, peps=-0.5)

                else:
                    muestreador.sample(n_iter)

                egr_spotpy = BDtexto(temp.name + '.csv')
            else:
                raise ValueError(_('Método de calibración "{}" no reconocido.').format(método))
        else:
            egr_spotpy = BDtexto(egr_spotpy)

        cols_prm = [c for c in egr_spotpy.obt_nombres_cols() if c.startswith('par')]
        trzs = egr_spotpy.obt_datos(cols_prm)
        probs = egr_spotpy.obt_datos('like1')['like1']
        chains = egr_spotpy.obt_datos('chain')['chain']
        # if os.path.isfile(temp.name + '.csv'):
        #     os.remove(temp.name + '.csv')
        if obj_func in ['rmse', 'aic']:
            probs = np.negative(probs)
            buenas = (probs <= np.max(np.sort(probs)[:int(len(probs) * 0.2)]))
        elif obj_func in ['mic', 'nse']:
            buenas = (probs >= np.min(np.sort(probs)[int(len(probs) * 0.8):]))

        if tipo_proc is None:
            trzs = trzs[-n_iter:]
            prob = probs[-n_iter:]

            pesos = (prob - np.nanmin(probs)) / np.ptp(probs)  # those > 0.8 like, weight distribution

            res = {'buenas': np.where(buenas)[0], 'peso': pesos, 'máx_prob': np.nanmax(probs), 'prob': prob}

            for i, p in enumerate(paráms):
                col_p = 'par' + str(i)
                res[p] = {'dist': trzs[col_p], 'val': _calc_máx_trz(trzs[col_p])}
        else:
            par_spotpy = {k: str(i) for i, (k, v) in enumerate(final_líms_paráms.items())}
            res = {'buenas': np.where(buenas)[0], 'prob': probs, 'chains': chains,
                   'sampled_prm': {prm: trzs[f'par{par_spotpy[prm]}'] for prm in par_spotpy}}
            ####
            print(obj_func)
            for k in range(5):
                a = [j[k] for j in [v[-5:] for i, v in res['sampled_prm'].items()]]
                print('SD:', np.average(a[:3]), 'P:', np.average(a[-10:]))
            print(res['prob'][-5:])
            ###
            parameters = {}
            for i in range(len(list(trzs.values())[0])):
                x = np.asarray([val[i] for val in list(trzs.values())])
                val_inic = gen_val_inic(x, mapa_paráms, líms_paráms, final_líms_paráms)
                for k, val in val_inic.items():
                    if k in parameters:
                        parameters[k].append(val)
                    else:
                        parameters[k] = [val]
            res['parameters'] = {p: np.asarray(v) for p, v in parameters.items()}

        if guardar:
            prms = {list(final_líms_paráms)[i]: v for i, (p2, v) in enumerate(trzs.items())}
            prms.update({'likes': probs})

            np.save(guardar + '_PrmProb', prms)
            np.save(guardar, res)

        return res


# Unas funciones auxiliares
def _calibrar_mod_bayes(mod_bayes, paráms, obs=None, vars_compartidos=None, ops=None):
    """
    Esta función calibra un modelo bayes.

    Parameters
    ----------
    mod_bayes: pm.Modelo
        El modelo para calibrar.
    paráms: list[str]
        Una lista de los nombres de los parámetros de interés para sacar de la traza del modelo.
    obs: dict
        Base de datos observados.
    vars_compartidos: dict
        Un diccionario con los variables compartidos Theano en los cuales podemos poner nuevas observaciones.
    ops: dict
        Opciones adicionales para pasar a pm.Modelo.sample.

    Returns
    -------

    """

    # El diccionario de opciones adicionales.
    if ops is None:
        ops = {}

    # Si hay variables de datos compartidos, poner los nuevos datos.
    if vars_compartidos is not None:
        for var, var_pymc in vars_compartidos.items():
            var_pymc.set_value(obs[var])

    # Crear el diccionarion de argumentos
    ops_auto = {
        'tune': 1000,
        'cores': 1
    }
    ops_auto.update(ops)

    # Efectuar la calibración
    with mod_bayes:
        t = pm.sample(**ops_auto)

    # Devolver los datos procesados
    return _procesar_calib_bayes(t, paráms=paráms)


def _procesar_calib_bayes(traza, paráms):
    """
    Procesa los resultados de una calibración bayes. Con base en la traza PyMC3, calcula el punto de probabilidad más
    alta para cada parámetro de interés.

    Parameters
    ----------
    traza: pm.Trace
        La traza PyMC3.
    paráms: list
        La lista de parámetros de interés.

    Returns
    -------
    dict
        Los resultados procesados.
    """

    # El diccionario para los resultados
    d_máx = {}

    # Calcular el punto de probabilidad máxima
    for p in paráms:
        # Para cada parámetro...

        dims = traza[p].shape
        if len(dims) == 1:
            d_máx[p] = _calc_máx_trz(traza[p])
        elif len(dims) == 2:
            d_máx[p] = np.empty(dims[1])
            for e in range(dims[1]):
                d_máx[p][e] = _calc_máx_trz(traza[p][:, e])
        else:
            raise ValueError

    # Devolver los resultados procesados.
    return {p: {'val': d_máx[p], 'dist': traza[p]} for p in paráms}


def _calc_máx_trz(trz):
    if len(trz) == 1:
        return trz[0]

    # Ajustar el rango, si es muy grande (necesario para las funciones que siguen)
    escl = np.max(trz)
    rango = escl - np.min(trz)
    if escl < 10e10:
        escl = 1  # Si no es muy grande, no hay necesidad de ajustar

    # Intentar calcular la densidad máxima.
    try:
        # Se me olvidó cómo funciona esta parte.
        fdp = gaussian_kde(trz / escl)
        x = np.linspace(trz.min() / escl - 1 * rango, trz.max() / escl + 1 * rango, 1000)
        máx = x[np.argmax(fdp.evaluate(x))] * escl
        return máx

    except BaseException:
        return np.nan


def _conv_xr(datos, vars_obs, warmup_period=None):  # datos[0] time; datos[1] dict{obspoly: ndarray}
    if warmup_period is None:
        warmup_period = 0

    if isinstance(datos, tuple):
        datos = list(datos)
        matriz_vacía = np.empty([len(datos[0]) - warmup_period, len(datos[1])])  # 60, 38
        datos[0] = datos[0].values[warmup_period:]
    else:
        raise TypeError(_("Por favor agregue o seleccione el tipo correcto de los datos observados."))

    for poly, data in datos[1].items():
        matriz_vacía[:, list(datos[1]).index(poly)] = data[warmup_period:]

    return xr.Dataset(
        data_vars={vars_obs[0]: (('n', 'x0'), matriz_vacía)},
        coords={'n': datos[0],
                'x0': np.asarray(list(datos[1])),
                'tiempo': np.asarray(range(len(list(datos[0])))),
                }
    )


def _optimizar(func, líms_paráms, obs_x, obs_y, inic=None, **ops):
    """
    Optimiza una función basándose en observaciones.

    Parameters
    ----------
    func: Callable
        La función para optimizar.
    líms_paráms: dict[str, tuple]
        Un diccionario de los parámetros y de sus límites.
    obs_x: pd.DataFrame
        Las observaciones de los variables x.
    obs_y: pd.Series | np.array
        Las observaciones correspondientes del variable y.
    inic: list | np.array
        Los valores iniciales para la optimización.
    ops: dict
        Opciones para pasar a la función de optimización.

    Returns
    -------
    dict
        Los parámetros optimizados.
    """

    # Leer el método de ajusto.
    try:
        med_ajusto = ops.pop('med_ajusto')
    except KeyError:
        med_ajusto = 'rmec'

    # La lista de parámetros de interés.
    paráms = list(líms_paráms)

    # Crear la función objetiva que minimizaremos (SciPy solamente puede minimizar).
    def f(prm):
        """
        Una función objetiva que SciPy puede minimizar.

        Parameters
        ----------
        prm: list | np.ndarray
            Los parámetros para calibrar

        Returns
        -------
        np.ndarray
            El ajusto del modelo con los parámetros actuales.

        """

        # Definir la función de ajusto.
        if med_ajusto.lower() == 'rmec':
            # Raíz media del error cuadrado
            def f_ajusto(y, y_o):
                return np.sqrt(np.sum(np.square(y - y_o)) / len(y))

        else:
            # Implementar otras medidas de ajusto aquí.
            raise ValueError(_('La medida de ajusto "{}" no se reconoció.').format(med_ajusto))

        # Devolver el ajusto de la función con los parámetros actuales.
        return f_ajusto(func(prm, obs_x), obs_y)

    # Generar los estimos iniciales, si necesario
    if inic is not None:
        # No es necesario
        x0 = inic
    else:
        # Sí es necesario
        x0 = []
        for p in paráms:
            # Para cada parámetro...
            lp = líms_paráms[p]  # Sus límites

            # Calcular un punto razonable para empezar la búsqueda
            if lp[0] is None:
                if lp[1] is None:
                    # El caso (-inf, +inf): empezaremos en 0
                    x0.append(0)
                else:
                    # El caso (-inf, R]: empezaremos en R
                    x0.append(lp[1])
            else:
                if lp[1] is None:
                    # El caso [R, +inf): empezaremos en R
                    x0.append(lp[0])
                else:
                    # El caso [R1, R2]: empezaremos en el promedio de R1 y R2
                    x0.append((lp[0] + lp[1]) / 2)

    # Convertir a matriz NumPy
    x0 = np.array(x0)

    # Optimizar
    opt = minimize(f, x0=x0, bounds=[líms_paráms[p] for p in paráms], **ops)

    # Avisar si la optimización no funcionó tan bien como lo esperamos.
    if not opt.success:
        avisar(_('Es posible que haya un error de optimización. Mejor le eches un vistazo a los resultados.'))

    # Devolver los resultados en el formato correcto.
    return {p: {'val': opt.x[i]} for i, p in enumerate(paráms)}


_algs_spotpy = {
    'fast': spotpy.algorithms.fast,
    'dream': spotpy.algorithms.dream,
    'mc': spotpy.algorithms.mc,
    'mcmc': spotpy.algorithms.mcmc,
    'mle': spotpy.algorithms.mle,
    'lhs': spotpy.algorithms.lhs,
    'sa': spotpy.algorithms.sa,
    'sceua': spotpy.algorithms.sceua,
    'demcz': spotpy.algorithms.demcz,
    'rope': spotpy.algorithms.rope,
    'abc': spotpy.algorithms.abc,
    'fscabc': spotpy.algorithms.fscabc,
}


class ModSpotPy(object):
    def __init__(símismo, mod, líms_paráms, obs):
        """

        Parameters
        ----------
        mod : Modelo.Modelo
        líms_paráms : dict
        obs: xr.Dataset
        """

        símismo.paráms = [
            spotpy.parameter.Uniform(str(list(líms_paráms).index(p)), low=d[0], high=d[1], optguess=(d[0] + d[1]) / 2)
            for p, d in líms_paráms.items()
        ]
        símismo.nombres_paráms = list(líms_paráms)
        símismo.mod = mod
        símismo.vars_interés = sorted(list(obs.data_vars))
        símismo.t_final = len(obs['n']) - 1

        símismo.mu_obs = símismo._aplastar(obs.mean())  # nparrar(3,) -> 1; obs=xr.Dtset(21,3)
        símismo.sg_obs = símismo._aplastar(obs.std())  # nparrar(3,) -> 1
        símismo.obs_norm = símismo._aplastar((obs - obs.mean()) / obs.std())  # ??? (63,)

    def parameters(símismo):
        return spotpy.parameter.generate(símismo.paráms)

    def simulation(símismo, x):
        res = símismo.mod.simular(
            t_final=símismo.t_final, vars_interés=símismo.vars_interés, vals_inic=dict(zip(símismo.nombres_paráms, x))
        )
        m_res = np.array([res[v].values for v in símismo.vars_interés]).T  # Transpose the array shape why?

        return ((m_res - símismo.mu_obs) / símismo.sg_obs).T.ravel()  # (63,) 1*21*3 -> 3*21 -> flatten (??)

    def evaluation(símismo):
        return símismo.obs_norm

    def objectivefunction(símismo, simulation, evaluation, params=None):
        like = spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)  # should be ave()

        return like

    def _aplastar(símismo, datos):
        if isinstance(datos, xr.Dataset):
            return np.array([datos[v].values.ravel() for v in símismo.vars_interés]).ravel()
        elif isinstance(datos, dict):
            return np.array([datos[v].ravel() for v in sorted(datos)]).ravel()


class PatrónProc(object):
    itr = 0

    def __init__(símismo, mod, líms_paráms, obs, tipo_proc, mapa_paráms, comp_final_líms, t_final,
                 guar_sim, warmup_period, cls, obj_func, método, simluation_res, ind_simul):
        símismo.paráms = [spotpy.parameter.Uniform(str(list(comp_final_líms).index(p)), low=d[0], high=d[1],
                                                   optguess=(d[0] + d[1]) / 2) for p, d in comp_final_líms.items()]
        # spotpy.parameter.logNormal
        símismo.mapa_paráms = mapa_paráms
        símismo.líms_paráms = líms_paráms
        símismo.final_líms_paráms = comp_final_líms
        símismo.mod = mod
        símismo.guar_sim = guar_sim
        símismo.vars_interés = sorted(list(obs.data_vars))
        símismo.t_final = t_final
        símismo.tipo_proc = tipo_proc
        símismo.warmup_period = warmup_period
        símismo.poly = obs['x0'].values
        símismo.obs = obs[símismo.vars_interés[0]].values
        símismo.mu_obs, símismo.sg_obs, símismo.obs_norm = aplastar(símismo.poly, símismo.obs)
        símismo.obj_func = obj_func
        símismo.eval = patro_proces(símismo.tipo_proc, símismo.poly, símismo.obs_norm, obj_func=símismo.obj_func)
        símismo.cls = cls
        símismo.método = método
        símismo.simul_res = simluation_res
        símismo.ind_simul = ind_simul

    def parameters(símismo):
        return spotpy.parameter.generate(símismo.paráms)

    def simulation(símismo, x):
        if símismo.simul_res is None:
            vals_inic = {PatrónProc.itr:
                             gen_val_inic(x, símismo.mapa_paráms, símismo.líms_paráms, símismo.final_líms_paráms)}

            res = símismo.mod.simular_grupo(vars_interés=símismo.vars_interés[0], t_final=40,
                                            vals_inic=vals_inic, guardar=símismo.guar_sim)
        else:
            ind = np.arange(símismo.ind_simul[0], símismo.ind_simul[1])[PatrónProc.itr]
            res = {PatrónProc.itr: Dataset.from_dict(cargar_json(os.path.join(símismo.simul_res, f'{ind}')))}

        PatrónProc.itr += 1

        if símismo.warmup_period is None:
            símismo.warmup_period = 0

        if isinstance(res, dict):
            m_res = np.asarray(
                [list(res.values())[0][símismo.vars_interés[0]].values[símismo.warmup_period:, i - 1] for i in
                 símismo.poly]) #19*41

            return ((m_res.T - símismo.mu_obs) / símismo.sg_obs)

        else:
            m_res = np.array([res[v].values for v in símismo.vars_interés][0])[símismo.warmup_period:]  # 62*215

            return ((m_res - símismo.mu_obs) / símismo.sg_obs)  # 39*18

    def evaluation(símismo):
        return símismo.eval

    def objectivefunction(símismo, simulation, evaluation, params=None):
        gof = gen_gof(símismo.tipo_proc, simulation, evaluation, cls=símismo.cls, obj_func=símismo.obj_func,
                      método=símismo.método)
        return gof


def patro_proces(tipo_proc, npoly, norm_obs, valid=False, obj_func='aic'):
    if valid:
        best_behaviors, linear, all_bbehav_params = compute_patron(npoly, norm_obs, valid=valid, obj_func=obj_func,
                                                                   tipo_proc=tipo_proc)
        return best_behaviors, linear, all_bbehav_params

    else:
        if norm_obs.shape[0] == len(npoly):
            norm_obs = norm_obs.T

        if tipo_proc == 'multidim':  # {var: nparray[61, 38]}
            return norm_obs  # nparray[38, 61]

        elif tipo_proc == 'patrón':
            best_behaviors = compute_patron(npoly, norm_obs, obj_func=obj_func)  # norm_obs: 19*39
            return best_behaviors


def aplastar(polys, dato):
    dato = dato.astype(float)
    len_dato = len(polys)

    if dato.shape[1] != len_dato:
        dato = dato.T

    mu = [np.nanmean(dato[:, i]) for i in range(len_dato)]
    sg = np.array([np.nanstd(dato[:, i]) for i in range(len_dato)])  # 215

    norm = np.array([((dato[:, i] - mu[i]) / sg[i]) for i in range(len_dato)])  # 38*61
    return mu, sg, norm


def gen_val_inic(x, mapa_paráms, líms_paráms, final_líms_paráms):
    vals_inic = {p: np.array(x[list(final_líms_paráms).index(p)]) for p in líms_paráms if p in final_líms_paráms}

    for p, mapa in mapa_paráms.items():
        if isinstance(mapa, list):
            mapa = np.array(mapa)

        if isinstance(mapa, np.ndarray):
            vals_inic[p] = np.empty(len(mapa))
            for i, cof in enumerate(mapa):
                vals_inic[p][i] = x[list(final_líms_paráms).index(f'{p}_{cof}')]

        elif isinstance(mapa, dict):
            transf = mapa['transf'].lower()
            for var, mp_var in mapa['mapa'].items():
                vals_inic[var] = np.empty(len(mp_var))
                if transf == 'prom':
                    for i, t_índ in enumerate(mp_var):
                        vals_inic[var][i] = (x[list(final_líms_paráms).index(f'{p}_{t_índ[0]}')] +
                                             x[list(final_líms_paráms).index(f'{p}_{t_índ[1]}')]) / 2
                else:
                    raise ValueError(_('Transformación "{}" no reconocida.').format(transf))

    return vals_inic


def gen_gof(tipo_proc, sim, eval, valid=False, cls=False, obj_func=None, valid_like=False, método=None):
    if tipo_proc == 'patrón':
        likes = np.zeros([len(eval)])
        poly = list(eval.keys())

        if len(sim[0, :]) != len(poly):
            sim = sim.T

        if valid:
            linear, shps = {}, {}

        for i, (p, best_behav) in enumerate(eval.items()):
            if best_behav[:3] == 'spp':
                if 'spp_oscil_aten' in best_behav:
                    behaviour = best_behav[[x.start() for x in re.finditer("\_", best_behav)][2] + 1:]
                elif 'spp_oscil' in best_behav:
                    behaviour = best_behav[[x.start() for x in re.finditer("\_", best_behav)][1] + 1:]
            else:
                behaviour = best_behav

            behaviours = [['linear', behaviour] if valid else [behaviour]][0]
            shape = superposition(np.arange(1, sim.shape[0] + 1), sim[:, poly.index(p)], gof_type=[obj_func],
                                  behaviours=behaviours)[0]

            if valid:
                shps[p], linear[p] = shape[best_behav], shape['linear']

            likes[i] = shape[best_behav]['gof'][obj_func]

        if valid_like:
            return likes
        elif valid:
            return likes, linear, shps

        else:
            if método == 'abc' or método == 'sceua' or método == 'fscabc':
                return _cls_objfc(cls=cls, obj_func=obj_func, min_max='min', likes=likes)
            else:
                return _cls_objfc(cls=cls, obj_func=obj_func, min_max='max', likes=likes)

    elif tipo_proc == 'multidim':
        if eval.shape != sim.shape:  # 39*18
            eval = eval.T
        poly = eval.shape[1]
        likes = np.zeros(poly)

        for i in range(poly):
            if obj_func.lower() == 'nse':
                likes[i] = nse(eval[:, i], sim[:, i])
            elif obj_func.lower() == 'rmse':
                likes[i] = compute_rmse(eval[:, i], sim[:, i])
            else:
                raise ValueError(f"{obj_func} is not considered")
        if valid:
            return likes
        else:
            if método == 'abc' or método == 'sceua' or método == 'fscabc':
                return _cls_objfc(cls=cls, obj_func=obj_func, min_max='min', likes=likes)
            else:
                return _cls_objfc(cls=cls, obj_func=obj_func, min_max='max', likes=likes)


def _cls_objfc(cls, obj_func, min_max, likes):
    l_max = ['nse', 'mic']
    l_min = ['rmse', 'aic']
    if min_max == 'min':
        if cls:
            return [-_classify(likes) if obj_func in l_max else _classify(
                likes) if obj_func in l_min else f"{obj_func} is not considered"][0]
        else:
            return [-np.nanmean(likes) if obj_func in l_max else np.nanmean(
                likes) if obj_func in l_min else f"{obj_func} is not considered"][0]

    elif min_max == 'max':
        if cls:
            return [_classify(likes) if obj_func in l_max else -_classify(
                likes) if obj_func in l_min else f"{obj_func} is not considered"][0]
        else:
            return [np.nanmean(likes) if obj_func in l_max else -np.nanmean(
                likes) if obj_func in l_min else f"{obj_func} is not considered"][0]


def _classify(l_likes):
    maxi = np.nanmax(l_likes)
    mini = np.nanmin(l_likes)
    jump = (maxi - mini) / 3
    cls1 = [i for i in l_likes if int(i) in range(int(mini - 0.5), int(mini + jump))]
    cls2 = [i for i in l_likes if int(i) in range(int(mini + jump - 0.5), int(mini + 2 * jump))]
    cls3 = [i for i in l_likes if int(i) in range(int(mini + 2 * jump - 0.5), int(mini + 3 * jump + 0.5))]
    best = np.asarray(max([cls1, cls2, cls3], key=len))
    return np.nanmean(best)
