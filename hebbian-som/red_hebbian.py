import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

FILE_PATH_DATASET = 'tp2_training_dataset.csv'


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


def centrar_matriz(matriz):
    matriz_t = np.array(matriz).T
    ret = []
    for columna in matriz_t:
        media = np.mean(columna)
        ret.append(columna-media)
    return np.array(ret).T


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

    @staticmethod
    def graficar(datos, columna_clases):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = []
        Y = []
        Z = []
        for elem in datos:
            X.append(elem[0])
            Y.append(elem[1])
            Z.append(elem[2])
        #cmap = plt.get_cmap('jet', 4)
        ax.scatter(X, Y, Z, c=columna_clases)
        #ax.view_init(30, 0)
        plt.draw()
        plt.show()

    # Entreno red
    def entrenar(self, dataset_entrada, dataset_salida, epocas, eta, algoritmo):
        # for X en D:
        #     Y = X . W
        #     for j en [1..M]: // cantidad outputs
        #         for i en [1..N]: // cantidad inputs
        #             X~_i = 0
        #             for k en [1..Q]  //cota algoritmo
        #                 X~_i += Y_k . W_ik //mini-delta
        #             DeltaW_ij = eta . (X_i - X~_i) . Y_j
        #     W += DeltaW
        for epoca in range(epocas):
            print('ENTRENANDO -> dataset: %d --  algoritmo: %s -- epoca: %d/%d -- eta: %f  -- neuronas salida: %d' % (len(dataset_entrada), algoritmo, epoca, epocas, eta, self.n_salida))
            # for X en D:
            for X in dataset_entrada:
                #Y = X.W
                y = []
                for output in range(self.n_salida):
                    y_i = np.dot(X, self.W[output])
                    y.append(y_i)

                #for j en[1..M]:
                for j in range(self.n_salida):
                    w_delta = []

                    if algoritmo == "oja":
                        intervalo = len(self.W)

                    elif algoritmo == "sanger":
                        intervalo = output + 1

                    #for i en[1..N]:
                    for i in range(len(X)):
                        x = 0
                        #for k en[1..Q]
                        for k in range(intervalo):
                            #X~_i += Y_k.W_ik // mini - delta
                            x += y[k] * self.W[k][i]
                        #DeltaW_ij = eta.(X_i - X~_i).Y_j
                        w_delta.append(eta * (X[i] - x) * y[j]) #como se ordenan aca?

                    # W += DeltaW
                    self.W[j] = np.sum([self.W[j], np.array(w_delta)], axis=0)

    def testear_red(self, dataset, clases):
        resultados = []
        for X in dataset:
            y = []
            for output in range(self.n_salida):
                y_i = np.dot(X, self.W[output])
                y.append(y_i)
            resultados.append(y)
        return resultados

    def salvar_matriz_pesos(self, file_name):
        np.savetxt(file_name, self.W, delimiter=",")

    def setear_matriz_pesos(self, nombre_archivo):
        f = open(nombre_archivo, 'rt')
        matriz_pesos = []
        try:
            reader = csv.reader(f)
            for row in reader:
                matriz_pesos_row = []
                for i in row:
                    matriz_pesos_row.append(float(i))
                matriz_pesos.append(matriz_pesos_row)
        finally:
            f.close()
        self.W = matriz_pesos

    def graficar_en_3_partes(self, datos, categorias):
        primera = []
        segunda = []
        tercera = []
        for elem in datos:
            primera.append(elem[0:3])
            segunda.append(elem[3:6])
            tercera.append(elem[6:9])

        self.graficar(primera, categorias)
        self.graficar(segunda, categorias)
        self.graficar(tercera, categorias)