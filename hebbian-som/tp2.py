import csv
import numpy as np

FILE_PATH_DATASET = 'tp2_training_dataset.csv'
NEURONAS_SALIDA = 9


# Parser dataset
def parsear_dataset():
    f = open(FILE_PATH_DATASET, 'rt')
    dataset = []
    categoria = []
    try:
        reader = csv.reader(f)
        for row in reader:
            categoria.append(row[0])
            dataset_row = []
            for i in range(1, len(row)):
                dataset_row.append(float(row[i]))
            dataset.append(dataset_row)
    finally:
        f.close()
    return categoria, dataset


class Hebbian(object):

    def __init__(self, n_entrada, n_salida):
        self.n_entrada = n_entrada
        self.n_salida = n_salida
        self.W = self.generar_matriz_pesos_aleatorios(n_entrada, n_salida)

    # Genero pesos aleatorios
    @staticmethod
    def generar_matriz_pesos_aleatorios(n_entrada, n_salida):
        ret = []
        for _ in range(n_salida):
            ret.append(np.random.uniform(-0.1, 0.1, n_entrada))
        return ret

    # Entreno red
    def entrenar(self, dataset_entrada, dataset_salida, epocas, eta, algoritmo):
        # for X en D:
        #     Y = X . W
        #     for j en [1..M]:
        #         for i en [1..N]:
        #             X~_i = 0
        #             for k en [1..Q]
        #                 X~_i += Y_k . W_ik
        #             DeltaW_ij = eta . (X_i - X~_i) . Y_j
        #     W += DeltaW
        for epoca in range(epocas):

            for X in dataset_entrada:
                y = []
                for neurona_salida_i in range(self.n_salida):
                    y_i = np.dot(X, self.W[neurona_salida_i])
                    y.append(y_i)

                for neurona_salida_i in range(self.n_salida):
                    w_delta = []
                    if algoritmo == "hebb":
                        intervalo = 0

                    elif algoritmo == "oja1":
                        intervalo = 1

                    elif algoritmo == "oja":
                        intervalo = len(self.W)

                    elif algoritmo == "sanger":
                        intervalo = neurona_salida_i + 1

                    for j, x_i in enumerate(X):
                        x = 0
                        for k in range(intervalo):
                            x += y[k] * self.W[k]
                        w_delta.append(eta * (x_i - x) * neurona_salida_i)
                    self.W[neurona_salida_i] = np.sum([self.W[neurona_salida_i], np.array(w_delta)], axis=0)
                    print(y_i)


def prueba_red():
    categoria, dataset = parsear_dataset()
    red = Hebbian(len(dataset[0]), NEURONAS_SALIDA)
    red.entrenar(dataset, [], 300, 0.05, 'sanger')


prueba_red()