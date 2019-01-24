import os
import unittest

import numpy as np

from pruebas.recursos.prueba_calib import ModeloLogisticCalib
from tinamit.Análisis.Sens.muestr import gen_problema
from pruebas.test_mds import limpiar_mds
from tinamit.EnvolturasMDS import generar_mds
from tinamit.Geog import Geografía

dir_act = os.path.split(__file__)[0]
arch_mds = os.path.join(dir_act, 'recursos/MDS/mod_enferm.mdl')
arch_csv_geog = os.path.join(dir_act, 'recursos/datos/prueba_geog.csv')

líms_paráms = {
    'taza de contacto': (0, 100),
    'taza de infección': (0, 0.02),
    'número inicial infectado': (0, 50),
    'taza de recuperación': (0, 0.1)
}


class Test_CalibModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paráms = {
            'taza de contacto': 81.25,
            'taza de infección': 0.007,
            'número inicial infectado': 22.5,
            'taza de recuperación': 0.0375
        }
        cls.mod = generar_mds(arch_mds)

        cls.datos = cls.mod.simular(
            t_final=20,
            vals_inic=cls.paráms,
            vars_interés=['Individuos Suceptibles', 'Individuos Infectados', 'Individuos Resistentes']
        )

    def test_calibrar_validar(símismo):
        símismo.mod.calibrar(
            paráms=list(símismo.paráms),
            líms_paráms=líms_paráms,
            bd=símismo.datos
        )
        símismo.assertTrue(símismo.mod.validar(bd=símismo.datos)['éxito'])


class Test_CalibModeloEspacial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paráms = {
            'taza de contacto': {'708': 81.25, '1010': 50},
            'taza de infección': {'708': 0.007, '1010': 0.005},
            'número inicial infectado': {'708': 22.5, '1010': 40},
            'taza de recuperación': {'708': 0.0375, '1010': 0.050}
        }
        cls.mod = mod = generar_mds(arch_mds)
        mod.geog = Geografía('prueba', archivo=arch_csv_geog)
        mod.cargar_calibs(cls.paráms)
        cls.datos = mod.simular_en(
            t_final=25, en=['708', '1010'],
            vars_interés=['Individuos Suceptibles', 'Individuos Infectados', 'Individuos Resistentes']
        )
        mod.borrar_calibs()

    def test_calib_valid_espacial(símismo):
        símismo.mod.calibrar(paráms=list(símismo.paráms), bd=símismo.datos, líms_paráms=líms_paráms)
        valid = símismo.mod.validar(
            bd=símismo.datos,
            var=['Individuos Suceptibles', 'Individuos Infectados', 'Individuos Resistentes']
        )
        símismo.assertTrue(valid['éxito'])

    @classmethod
    def tearDownClass(cls):
        limpiar_mds()


class Test_CalibMultidim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.paráms = {
        #     'A': np.arange(1, 7),
        #     'B': 0.3 * np.arange(1, 7),
        #     'C': np.arange(1, 7)
        # }
        # cls.líms_paráms = {'A': (3, 10), 'B': (0.5, 2), 'C': (3, 5)}

        cls.paráms = {
            'A': np.arange(3, 5, 0.39),
            'B': np.arange(1, 2, 0.19),
            'C': np.arange(4, 6, 0.39)
        }

        cls.líms_paráms = {
            'A': [(3, 4), (4, 5)],
            'B': [(1, 1.5), (1.5, 2)],
            'C': [(4, 5), (5, 6)]
        }

        cls.mapa_paráms = {'A': [0, 0, 0, 1, 1, 1], 'B': [1, 0, 1, 0, 1, 0], 'C': [1, 1, 1, 0, 0, 0]}

        cls.mod = ModeloLogisticCalib()
        cls.mod.geog = Geografía('prueba', archivo=arch_csv_geog)
        cls.mod.cargar_calibs(cls.paráms)

        cls.datos = cls.mod.simular(
            t_final=25,
            vals_inic=cls.paráms,
            vars_interés=['y']
        )

    def test_multidim_calibrar_validar(símismo):
        líms_paráms_final = \
            gen_problema(líms_paráms=símismo.líms_paráms, mapa_paráms=símismo.mapa_paráms, ficticia=False)[1]

        símismo.mod.calibrar(paráms=list(símismo.paráms), bd=símismo.datos, líms_paráms=símismo.líms_paráms,
                             mapa_paráms=símismo.mapa_paráms, tipo_proc='multidim', final_líms_paráms=líms_paráms_final)

        valid = símismo.mod.validar(
            bd=símismo.datos,
            var=['y']
        )
        símismo.assertTrue(valid['éxito'])

    def test_patron_calibrar_validar(símismo):
        líms_paráms_final = \
            gen_problema(líms_paráms=símismo.líms_paráms, mapa_paráms=símismo.mapa_paráms, ficticia=False)[1]

        símismo.mod.calibrar(paráms=list(símismo.paráms), bd=símismo.datos, líms_paráms=símismo.líms_paráms,
                             mapa_paráms=símismo.mapa_paráms, tipo_proc='patrón', final_líms_paráms=líms_paráms_final,
                             obj_func='AIC')

        valid = símismo.mod.validar(bd=símismo.datos, var=['y'])
        símismo.assertTrue(valid['éxito'])
