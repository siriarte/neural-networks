from math import e, log
import numpy as np
from numpy import linalg as LA
import red_hebbian as hebbian
import matplotlib.pyplot as plt
import csv
from matplotlib import colors

def norm_col(x):
    y = LA.norm(x, axis=1)
    return y

def flatten(x): #debe ser una matriz x
    y = np.reshape(x,x.shape[0]*x.shape[1])
    return y

class PerceptronSOM:
    def __init__(self, n_entrada, filas_salida, columnas_salida):
        self.n_entrada = n_entrada
        self.filas_output = filas_salida
        self.columnas_output = columnas_salida
        self.W = np.matrix(hebbian.Hebbian.generar_matriz_pesos_aleatorios(self.n_entrada, self.filas_output * self.columnas_output))
        
    def minimo_indice(self, x):
        y = norm_col(x - self.W)
        min_index = np.argmin(y)
        return min_index
    
    def coordena_ganadora(self, min_index):
        coord_col = min_index%self.columnas_output
        coord_fila = int(min_index/self.filas_output)
        return [coord_fila, coord_col]

    def cordenada_minima_distancia(self, x):
        min_index = self.minimo_indice(x)
        coordenada_minima_distancia = self.coordena_ganadora(min_index)
        return coordenada_minima_distancia

    def vecino(self, coordenada_actual, coordenada_ganadora, sigma):
        dist_ij2posicion = (coordenada_actual[0]-coordenada_ganadora[0])**2 + (coordenada_actual[1]-coordenada_ganadora[1])**2
        dist_ij2posicion = (dist_ij2posicion * 2) / (2*(sigma**2))
        return  np.e**(-dist_ij2posicion)

    def train_som(self, data_set, eta=0.1, sigma=5, epocas=1000):
        factor = 10**int(log(eta,10)-2)*0.25
        factor_sigma = sigma/3
        iterador_fase = 1
        tamano_fase = int(epocas / 3)
        eta_inicial = eta
        sigma_inicial = sigma
        for epoca in range(epocas):
            print('ENTRENANDO -> dataset: %d -- epoca: %d/%d -- iterador fase %d/%d -- eta: %f  -- inicial_eta: %f -- sigma: %f -- inical_sigma:%f -- filas: %d -- columnas %d' %
                (len(data_set), epoca, epocas, iterador_fase, tamano_fase, eta, eta_inicial, sigma, sigma_inicial, self.filas_output, self.columnas_output))

            coordenada_ganadora = []

            # Para cada documento
            for x in data_set:

                # Busco la coordenada de la distancia más corta
                x = np.matrix(x)
                min_index = self.minimo_indice(x)
                coordenada_ganadora = self.coordena_ganadora(min_index)

                for i in range(self.filas_output):
                    for j in range(self.columnas_output):
                        fx_vecindad = self.vecino([i, j], coordenada_ganadora, sigma)
                        posicion_matriz_pesos = i*self.filas_output+j
                        y = np.subtract(x, self.W[posicion_matriz_pesos])
                        delta = fx_vecindad * y
                        self.W[posicion_matriz_pesos] += eta * delta

            # Actualizo ETA y SIGMA de la epoca
            eta_nuevo = eta - factor
            sigma_nuevo = sigma - (sigma*0.005)
            if eta_nuevo > 0:
                    eta = eta_nuevo
            if sigma_nuevo > 0:
                    sigma = sigma_nuevo

            # Actualizo ETA y SIGMA de la fase
            if iterador_fase == tamano_fase:
                sigma_nuevo = sigma - (sigma_inicial/3.5)
                eta_nuevo = eta_inicial / 10
                if sigma_nuevo > 0:
                    sigma = sigma_nuevo
                if eta_nuevo > 0:
                    eta = eta_nuevo
                factor = 10 ** int(log(eta,10)-3) * 0.50
                iterador_fase = 0
            iterador_fase +=1

        return self

    def calcular_y_graficar(self, data_set, categorias, out_file_name):
        cantidad_entradas = len(data_set)
        matriz_neuronas = []
        for i in range(self.filas_output):
            matriz_neuronas.append([])
            for _ in range(self.columnas_output):
                matriz_neuronas[i].append([])

        for i in range(cantidad_entradas):
            x = np.matrix(data_set[i])
            min_index = self.minimo_indice(x)
            coordenada = self.coordena_ganadora(min_index)
            matriz_neuronas[coordenada[0]][coordenada[1]].append(categorias[i])

        resultados = np.zeros((self.filas_output, self.columnas_output))
        for i in range(self.filas_output):
            for j in range(self.columnas_output):
                if len(matriz_neuronas[i][j])!=0:
                    resultados[i][j] = int(max(set(matriz_neuronas[i][j]), key=matriz_neuronas[i][j].count))
                else:
                    resultados[i][j] = 0

        colores = ['w', 'green', 'blue', 'm', 'cyan', 'steelblue', 'gray', 'indigo', 'lightpink', 'red', 'yellow']
        mapa_colores = colors.ListedColormap(colores)
        plt.matshow(resultados, cmap=mapa_colores, norm=colors.BoundaryNorm(range(len(colores)), mapa_colores.N))
        plt.colorbar()
        #plt.show()
        plt.savefig(out_file_name)

    def salvar_matriz_pesos(self, file_name):
        np.savetxt(file_name, self.W, delimiter=",")
        return

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