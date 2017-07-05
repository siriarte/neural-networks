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
        #print(min_index)
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
        it = 0
        eta_inicial = eta
        sigma_inicial = sigma
        t1_sigma = float(epocas) / float(log(sigma_inicial, 2))
        t2_eta = float(epocas)
        for epoca in range(epocas):
            print('ENTRENANDO -> dataset: %d -- epoca: %d/%d -- eta: %f  -- inicial_eta: %f -- sigma: %f -- inical_sigma:%f -- filas: %d -- columnas %d' %
                (len(data_set), epoca, epocas, eta, eta_inicial, sigma, sigma_inicial, self.filas_output, self.columnas_output))

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
                        y = np.subtract(x, self.W[i+j])
                        delta = fx_vecindad * y
                        self.W[i+j] += eta * delta


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

        print(matriz_neuronas)

        for i in range(cantidad_entradas):
            x = np.matrix(data_set[i])
            min_index = self.minimo_indice(x)
            print(min_index)
            coordenada = self.coordena_ganadora(min_index)
            matriz_neuronas[coordenada[0]][coordenada[1]].append(categorias[i])

        print(matriz_neuronas)
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
