import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def variacion_momentum():
    seed(1)
    # Parseo datos
    datos_ejercicio = parser_entrada.datos_ejercicio_2()
    cantidad_neuronas_entrada = 8
    cantidad_neuronas_salida = 2

    # Tamaño de los sets, entrenamiento = 80% validacion = 10% test = 10%
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada']) * 0.8)
    tamanio_validacion = int((len(datos_ejercicio['entrada']) - tamanio_entrenamiento) / 2)

    # Entrenamiento
    datos_entrenamiento_entrada = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]

    # Validacion
    datos_validacion_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Test
    datos_testing_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento + tamanio_validacion:]
    datos_testing_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento + tamanio_validacion:]

    # Entreno red - Momentum = 0.1
    red_0 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neuronas_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neuronas_salida, 0.1, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neuronas_salida, 0.3, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neuronas_salida, 0.5, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neuronas_salida, 0.7, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - Momentum = 0.9
    red_5 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neuronas_salida, 0.9, 1, 0, 0, 0, '')
    red_5.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()
    [x5, y5] = red_5.get_error_entrenamiento_por_epoca()

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
    plt.savefig("graficos\\ej2_prueba_4_momentum_entrenamiento.png")
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
    plt.savefig("graficos\\ej2_prueba_4_momentum_validacion.png")
    # plt.show()

    # Imprimo nivel de testing
    print('Momentum = 0  ---> TEST = %f' % (red_0.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Momentum = 0.1  ---> TEST = %f' % (red_1.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Momentum = 0.3  ---> TEST = %f' % (red_2.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Momentum = 0.7  ---> TEST = %f' % (red_3.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Momentum = 0.9  ---> TEST = %f' % (red_4.test_ej2(datos_testing_entrada, datos_testing_resultados)))


##########################
#EJECUCION DE LAS PRUEBAS#
##########################
variacion_momentum()