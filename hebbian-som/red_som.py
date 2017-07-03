from math import e, log
import numpy as np
from numpy import linalg as LA
import red_hebbian as hebbian
import pandas as pd
import matplotlib.pyplot as plt

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
        coord_fila = int(np.ceil(min_index/self.columnas_output))-1
        return [coord_fila, coord_col]

    def cordenada_minima_distancia(self, x):
        min_index = self.minimo_indice(x)
        coordenada_minima_distancia = self.coordena_ganadora(min_index)
        return coordenada_minima_distancia

    def vecino(self, position, sigma):
        filas = self.filas_output
        columnas = self.columnas_output
        ret_matriz = np.matrix(np.zeros(filas*columnas))
        ret_matriz = ret_matriz.reshape(filas, columnas)
        for i in range(filas):
            for j in range(columnas):
                dist_ij2posicion = (i-position[0])**2 + (j-position[1])**2
                if sigma !=0:
                    dist_ij2posicion = dist_ij2posicion * 0.5 *(1/sigma)
                else:
                    raise Exception('SIGMA', ' ES 0')
                ret_matriz[i,j] = np.e**(-dist_ij2posicion)
        return ret_matriz.reshape(1, self.filas_output*self.columnas_output)

    def correccion(self, x, eta, sigma):
        x = np.matrix(x)
        min_index = self.minimo_indice(x)
        cordenada_ganadora = self.coordena_ganadora(min_index)
        vecindad = self.vecino(cordenada_ganadora, sigma)
        y = x - self.W
        delta = np.dot(flatten(vecindad),y)
        self.W += eta * delta
        return self
            
    def train_som(self, data_set, eta=0.1, sigma=5, epocas=1000):
        it = 0
        eta_inicial = eta
        sigma_inicial = sigma
        t1_sigma = float(epocas) / float(log(sigma_inicial, 2))
        t2_eta = float(epocas)
        for epoca in range(epocas):
            print('ENTRENANDO -> dataset: %d -- epoca: %d/%d -- eta: %f  -- inicial_eta: %f -- sigma: %f -- inical_sigma:%f -- filas: %d -- columnas %d' %
                (len(data_set), epoca, epocas, eta, eta_inicial, sigma, sigma_inicial, self.filas_output, self.columnas_output))

            for x in data_set:
                self.correccion(x, eta, sigma)

            it+=1
            if it < (epocas * 3 / 4):
                sigma, eta = self.variar_sigma_eta(it, sigma_inicial, t1_sigma, eta_inicial, t2_eta)

        return self

    def variar_sigma_eta(self, iteration_number, sigma_0, t1_sigma, eta_0, t2_eta):
        new_sigma = sigma_0 * (np.e ** (-(iteration_number / t1_sigma)))
        new_eta = eta_0 * (np.e ** (-(iteration_number / t2_eta)))
        return new_sigma, new_eta

    def calcular_y_graficar(self, data_set, categorias):
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

        cmap = plt.get_cmap('jet', 9)
        plt.matshow(resultados, cmap=cmap)
        plt.colorbar()
        plt.show()

    def salvar_matriz_pesos(self, file_name):
        print("FUNCIONALIDAD PARA SALVAR PESOS EN ARCHIVO NO IMPLEMENTADA")
        return

    def setear_matriz_pesos(self, file_name):
        print("FUNCIONALIDAD PARA LEER PESOS DE ARCHIVO NO IMPLEMENTADA")
        return
