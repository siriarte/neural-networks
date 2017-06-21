import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def variacion_early_stopping():
    seed(1)
    # Parseo datos
    datos_ejercicio = parser_entrada.datos_ejercicio_1()
    cantidad_neuronas_entrada = 10
    cantidad_neurona_salida = 1

    # Tamaño de los sets, entrenamiento = 80% validacion = 10% test = 10%
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada']) * 0.8)
    tamanio_validacion = int((len(datos_ejercicio['entrada']) - tamanio_entrenamiento) / 2)

    # Entrenamiento
    datos_entrenamiento_entrada = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]

    # Validacion
    datos_validacion_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento :]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Test
    datos_testing_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento + tamanio_validacion:]
    datos_testing_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento + tamanio_validacion:]

    # Entreno red - Momentum = 0.1
    red_0 = rn.RedNeuronal(cantidad_neuronas_entrada, [10, 10], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(cantidad_neuronas_entrada, [10, 10], cantidad_neurona_salida, 0, 1, 0.01, 0, 0, '')
    red_1.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(cantidad_neuronas_entrada, [10, 10], cantidad_neurona_salida, 0, 1, 0.08, 0, 0, '')
    red_2.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(cantidad_neuronas_entrada, [10, 10], cantidad_neurona_salida, 0, 1, 0.1, 0, 0, '')
    red_3.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(cantidad_neuronas_entrada, [10, 10], cantidad_neurona_salida, 0, 1, 0.15, 0, 0, '')
    red_4.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='EarlyStopping = 0')
    plt.plot(x1, y1, label='EarlyStopping = 0.01')
    plt.plot(x2, y2, label='EarlyStopping = 0.08')
    plt.plot(x3, y3, label='EarlyStopping = 0.1')
    plt.plot(x4, y4, label='EarlyStopping = 0.15')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_5_earlystopping_momentum_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()

    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='EarlyStopping = 0')
    plt.plot(x1, y1, label='EarlyStopping = 0.01')
    plt.plot(x2, y2, label='EarlyStopping = 0.08')
    plt.plot(x3, y3, label='EarlyStopping = 0.1')
    plt.plot(x4, y4, label='EarlyStopping = 0.15')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej1_prueba_5_earlystopping_momentum_validacion.png")
    # plt.show()

    print('EarlyStopping = 0 ---> TEST = %f' % (red_0.test_ej1(datos_testing_entrada, datos_testing_resultados)))
    print('EarlyStopping = 0.05 ---> TEST = %f' % (red_1.test_ej1(datos_testing_entrada, datos_testing_resultados)))
    print('EarlyStopping = 0.1 ---> TEST = %f' % (red_2.test_ej1(datos_testing_entrada, datos_testing_resultados)))
    print('EarlyStopping = 0.15 ---> TEST = %f' % (red_3.test_ej1(datos_testing_entrada, datos_testing_resultados)))
    print('EarlyStopping = 0.2 ---> TEST = %f' % (red_4.test_ej1(datos_testing_entrada, datos_testing_resultados)))

##########################
#EJECUCION DE LAS PRUEBAS#
##########################
variacion_early_stopping()