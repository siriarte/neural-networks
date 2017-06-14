import math
import numpy as np
import matplotlib.pyplot as plt

###########################################
# Funciones de activacion y sus derivadas #
###########################################


# Funcion logistica
def fx_logistica(x):
    return 1.0 / (1.0 + math.exp(-x))


# Derivada de la funcion logistica
def fx_logistica_prima(x):
    return x * (1.0 - x)


# Funcion tangente hiperbolica
def fx_tangente(x):
    return np.tanh(x)


# Derivada de la funcion tangente hiperbolica
def fx_tangente_prima(x):
    return 1.0 - (x ** 2)


# Guardar log en archivo
def guardar_log_en_archivo(nombre_archivo, error_por_epoca):
    archivo_salida = open(nombre_archivo, 'w')
    x = []
    y = []
    for elem in error_por_epoca:

        s = '%d\t%4f\n' % (elem[0], elem[1])
        archivo_salida.write(s)

        x.append(elem[0])
        y.append(elem[1])

    archivo_salida.close()

    plt.plot(x, y)
    plt.plot(x, [0.5 for i in range(len(x))])
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR')
    plt.grid(True)
    plt.savefig("test.png")
    plt.show()


def fx_umbral(umbral,valor):
    if(valor>umbral):
        return 1
    else:
        return 0