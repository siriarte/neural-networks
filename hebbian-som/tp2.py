import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

FILE_PATH_DATASET = 'tp2_training_dataset.csv'
NEURONAS_SALIDA = 3


def grafico_3d(prediccion, file_name, columna_clases,ang_elev =  None ,ang_azim = None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    longitud_pred = len(prediccion)
    xs = []
    for i in range(0,longitud_pred):
        xs.append(prediccion[i][0])
    ys = []
    for i in range(0,longitud_pred):
        ys.append(prediccion[i][1])
    zs = []
    for i in range(0,longitud_pred):
        zs.append(prediccion[i][2])
    ax.view_init(elev=ang_elev, azim=ang_azim)
    ax.scatter(xs, ys, zs,c = columna_clases)
    plt.savefig(file_name)
    #plt.show()

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
                        w_delta.append(eta * (X[i] - x) * y[j])

                    # W += DeltaW
                    self.W[output] = np.sum([self.W[output], np.array(w_delta)], axis=0)

    def testear_red(self, dataset, clases):
        resultados = []
        for X in dataset:
            y = []
            for output in range(self.n_salida):
                y_i = np.dot(X, self.W[output])
                y.append(y_i)
            resultados.append(y[0:3])
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




def prueba_red():
    for algoritmo in ['oja', 'sanger']:
        for neuronas_salida in [3, 9]:
            for epocas in [100, 200, 300]:
                for eta in [0.0001, 0.001, 0.01]:
                    categorias, dataset = parsear_dataset()
                    data_centrado = centrar_matriz(dataset)
                    red = Hebbian(len(dataset[0]), neuronas_salida)
                    str_eta = str(eta)[2:]
                    red.entrenar(data_centrado, [], epocas, eta, algoritmo)
                    archivo_salida_cvs = r'matrices_de_pesos/m_%s_%d_%s_%d.csv' % (algoritmo, epocas, str_eta, neuronas_salida)
                    red.salvar_matriz_pesos(archivo_salida_cvs)


#red.setear_matriz_pesos('matriz_pesos.csv')
#resultados = red.testear_red(data_centrado, categorias)
#grafico_3d(resultados,'g_100_180_0_600', categorias,180,0)
#grafico_3d(resultados, 'g_100_0_0_600' ,categorias,)
#grafico_3d(resultados, 'g_100_90_90_600',categorias,90,90)
#grafico_3d(resultados, 'g_100_180_45_600', categorias, 180, 45)






prueba_red()