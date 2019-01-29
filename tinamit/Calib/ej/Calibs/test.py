import spotpy
from spotpy.examples.hymod_python.hymod import hymod
import os

class setUpClass(object):
    def __init__(self):
        # Transform [mm/day] into [l s-1], where 1.783 is the catchment area
        self.Factor = 1.783 * 1000 * 1000 / (60 * 60 * 24)
        # Load Observation data from file
        self.PET, self.Precip = [], []
        self.date, self.trueObs = [], []
        self.owd = os.path.dirname(os.path.realpath(__file__))
        self.hymod_path = self.owd + os.sep + 'hymod_python'
        climatefile = open(self.hymod_path + os.sep + 'hymod_input.csv', 'r')
        headerline = climatefile.readline()[:-1]

        if ';' in headerline:
            self.delimiter = ';'
        else:
            self.delimiter = ','
        self.header = headerline.split(self.delimiter)
        for line in climatefile:
            values = line.strip().split(self.delimiter)
            self.date.append(str(values[0]))
            self.Precip.append(float(values[1]))
            self.PET.append(float(values[2]))
            self.trueObs.append(float(values[3]))

        climatefile.close()
        self.params = [spotpy.parameter.Uniform('cmax', low=1.0, high=500, optguess=412.33),
                       spotpy.parameter.Uniform('bexp', low=0.1, high=2.0, optguess=0.1725),
                       spotpy.parameter.Uniform('alpha', low=0.1, high=0.99, optguess=0.8127),
                       spotpy.parameter.Uniform('Ks', low=0.0, high=0.10, optguess=0.0404),
                       spotpy.parameter.Uniform('Kq', low=0.1, high=0.99, optguess=0.5592),
                       spotpy.parameter.Uniform('fake1', low=0.1, high=10, optguess=0.5592),
                       spotpy.parameter.Uniform('fake2', low=0.1, high=10, optguess=0.5592)]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, x):
        data = simular(
            t_final=100,
            vals_inic=cls.paráms, #import from the file
            vars_interés=['mds..']
        )
        sim = []
        for val in data:
            sim.append(val * self.Factor)
        return sim[366:]

    def evaluation(self): #import obs data
        return self.trueObs[366:]

    def objectivefunction(self, simulation, evaluation, params=None):
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation, simulation)
        return like

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
            t_final=100, en=['708', '1010'],
            vars_interés=['Individuos Suceptibles', 'Individuos Infectados', 'Individuos Resistentes']
        )
        mod.borrar_calibs()

    def test_calib_valid_espacial(símismo):
        símismo.mod.calibrar(paráms=list(símismo.paráms), bd=símismo.datos, líms_paráms=líms_paráms)
        valid = símismo.mod.validar(
            bd=símismo.datos,
            var=['Individuos Suceptibles', 'Individuos Infectados', 'Individuos Resistentes']
        )
