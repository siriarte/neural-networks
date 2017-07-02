import tp2 as hebbian
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def graficar(datos, columna_clases):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = []
    Y = []
    Z = []
    for elem in datos:
        X.append(elem[2])
        Y.append(elem[1])
        Z.append(elem[0])

    # load some test data for demonstration and plot a wireframe
    #ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    ax.scatter(X, Y, Z, c=columna_clases)
    # rotate the axes and update
    #for angle in range(0, 360):
    ax.view_init(30, 0)
    plt.draw()
    plt.show()

def partir_datos_en_tres(datos):
    primera = []
    segunda = []
    tercera = []
    for elem in datos:
        primera.append(elem[0:3])
        segunda.append(elem[3:6])
        tercera.append(elem[6:9])

    return primera, segunda, tercera

categorias, dataset = hebbian.parsear_dataset()
data_centrado = hebbian.centrar_matriz(dataset)
red = hebbian.Hebbian(len(dataset[0]), 3)
red.setear_matriz_pesos('matrices_de_pesos/m_sanger_sanger_100_0001_3.csv')
resultados = red.testear_red(data_centrado, categorias)
graficar(resultados, categorias)
#p,s,t = partir_datos_en_tres(resultados)
#graficar(p, categorias)
#graficar(s, categorias)
#graficar(t, categorias)
