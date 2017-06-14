import numpy as np


# Clase CapaNeuronal
class CapaNeuronal(object):

    # Constructor de una Capa de la Red Neuronal
    def __init__(self, cantidad_neuronas_capa_anterior, cantidad_neuronas, distrubucion_pesos = 0):

        # Inicializo variables
        self.cantidad_neuronas = cantidad_neuronas
        self.pesos = []
        self.salida = []
        self.delta = []
        self.delta_anterior = []

        # Sumo uno por el BIAS a la cantidad de axones
        self.cantidad_neuronas_capa_anterior = (cantidad_neuronas_capa_anterior + 1)

        # Genero los pesos aleatorios
        self.generar_pesos_aleatorios(distrubucion_pesos)

        # Inicializo con valores en cero todas las variables
        for i in range(self.cantidad_neuronas):
            self.salida.append(0.0)
            self.delta.append(0.0)
            self.delta_anterior.append(0.0)

    # Generador de pesos aletorios segun la distrucccion selecionada
    def generar_pesos_aleatorios(self, distrubucion_pesos):

        # Ciclo para cada neurona
        for i in range(0, self.cantidad_neuronas):

            # Decide cual distrubuccion utilizar
            if distrubucion_pesos == 0:
                lista_pesos = np.random.rand(self.cantidad_neuronas_capa_anterior).tolist()
            else:
                lista_pesos = np.random.uniform(-0.1, 0.1, self.cantidad_neuronas_capa_anterior).tolist()

            # Remplazo el valor del bias ya que no lo quiero aleatorio
            lista_pesos[-1] = 1.0

            # Agrego a la lista de pesos los pesos de la i-esima neurona
            self.pesos.append(lista_pesos)
