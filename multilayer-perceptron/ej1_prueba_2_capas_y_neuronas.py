import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def variacion_capas_ocultas():
    seed(1)
    # Parseo datos
    datos_ejercicio = parser_entrada.datos_ejercicio_1()
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada']) * 0.8)
    datos_entrenamiento = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]
    tamanio_entrada = len(datos_ejercicio['entrada'][0])
    tamanio_salida = 1
    datos_validacion = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Entreno red 1 capa 10 neuronas
    red_1 = rn.RedNeuronal(tamanio_entrada, [10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 2 capas 10 neuronas
    red_2 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 3 capas 10 neuronas
    red_3 = rn.RedNeuronal(tamanio_entrada, [10, 10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 5 capas 10 neuronas
    red_4 = rn.RedNeuronal(tamanio_entrada, [10, 10, 10, 10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Ploteo entrenamiento
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()

    plt.plot(x1, y1, label='1 Capa oculta')
    plt.plot(x2, y2, label='2 Capas ocultas')
    plt.plot(x3, y3, label='3 Capas ocultas')
    plt.plot(x4, y4, label='5 Capas ocultas')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_2_capas_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    plt.cla()
    plt.clf()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()

    plt.plot(x1, y1, label='1 Capa oculta')
    plt.plot(x2, y2, label='2 Capas ocultas')
    plt.plot(x3, y3, label='3 Capas ocultas')
    plt.plot(x4, y4, label='5 Capas ocultas')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_2_capas_validacion.png")
    # plt.show()

def variacion_neuronas_ocultas():
    seed(1)
    # Parseo datos
    datos_ejercicio = parser_entrada.datos_ejercicio_1()
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada']) * 0.8)
    datos_entrenamiento = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]
    tamanio_entrada = len(datos_ejercicio['entrada'][0])
    tamanio_salida = 1
    datos_validacion = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Entreno red 1 capa 10 neuronas
    red_1 = rn.RedNeuronal(tamanio_entrada, [2, 2], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 2 capas 10 neuronas
    red_2 = rn.RedNeuronal(tamanio_entrada, [5, 5], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 3 capas 10 neuronas
    red_3 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 5 capas 10 neuronas
    red_4 = rn.RedNeuronal(tamanio_entrada, [20, 20], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Ploteo entrenamiento
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()

    plt.plot(x1, y1, label='2 Neuronas')
    plt.plot(x2, y2, label='5 Neuronas')
    plt.plot(x3, y3, label='10 Neuronas')
    plt.plot(x4, y4, label='20 Neuronas')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_2_neuronas_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    plt.cla()
    plt.clf()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()

    plt.plot(x1, y1, label='2 Neuronas')
    plt.plot(x2, y2, label='5 Neuronas')
    plt.plot(x3, y3, label='10 Neuronas')
    plt.plot(x4, y4, label='20 Neuronas')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_2_neuronas_validacion.png")
    # plt.show()


##########################
#EJECUCION DE LAS PRUEBAS#
##########################
variacion_capas_ocultas()
variacion_neuronas_ocultas()