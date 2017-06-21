import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def variacion_eta_005_momentum():
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

    # Entreno red - Momentum = 0.1
    red_0 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.005, 300)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.1, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.005, 300)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.3, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.005, 300)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.5, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.005, 300)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.7, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.005, 300)

    # Entreno red - Momentum = 0.9
    red_5 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.9, 1, 0, 0, 0, '')
    red_5.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.005, 300)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()
    [x5, y5] = red_5.get_error_entrenamiento_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta005_momentum_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()
    [x5, y5] = red_5.get_error_validacion_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta005_momentum_validacion.png")
    # plt.show()

def variacion_eta_05_momentum():
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

    # Entreno red - Momentum = 0.1
    red_0 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.1, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.3, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.5, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.7, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.9
    red_5 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.9, 1, 0, 0, 0, '')
    red_5.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()
    [x5, y5] = red_5.get_error_entrenamiento_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta05_momentum_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    plt.cla()
    plt.clf()
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()
    [x5, y5] = red_5.get_error_validacion_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta05_momentum_validacion.png")
    # plt.show()


def variacion_eta_1_momentum():
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

    # Entreno red - Momentum = 0.1
    red_0 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.1, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.3, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.5, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.7, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Entreno red - Momentum = 0.9
    red_5 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.9, 1, 0, 0, 0, '')
    red_5.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()
    [x5, y5] = red_5.get_error_entrenamiento_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta1_momentum_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()
    [x5, y5] = red_5.get_error_validacion_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta1_momentum_validacion.png")
    # plt.show()

def variacion_eta_3_momentum():
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

    # Entreno red - Momentum = 0.1
    red_0 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 300)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.1, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 300)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.3, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 300)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.5, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 300)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.7, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 300)

    # Entreno red - Momentum = 0.9
    red_5 = rn.RedNeuronal(tamanio_entrada, [10, 10], tamanio_salida, 0.9, 1, 0, 0, 0, '')
    red_5.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 300)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()
    [x5, y5] = red_5.get_error_entrenamiento_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta3_momentum_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    plt.cla()
    plt.clf()
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()
    [x5, y5] = red_5.get_error_validacion_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Momentum = 0')
    plt.plot(x1, y1, label='Momentum = 0.1')
    plt.plot(x2, y2, label='Momentum = 0.3')
    plt.plot(x3, y3, label='Momentum = 0.5')
    plt.plot(x4, y4, label='Momentum = 0.7')
    plt.plot(x5, y5, label='Momentum = 0.9')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_7_eta3_momentum_validacion.png")
    # plt.show()


##########################
#EJECUCION DE LAS PRUEBAS#
##########################
variacion_eta_005_momentum()
#variacion_eta_05_momentum()
#variacion_eta_1_momentum()
#variacion_eta_3_momentum()