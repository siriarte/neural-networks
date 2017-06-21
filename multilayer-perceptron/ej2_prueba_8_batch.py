import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def variacion_batch():
    seed(1)
    # Parseo datos
    datos_ejercicio = parser_entrada.datos_ejercicio_2()
    cantidad_neuronas_entrada = 8
    cantidad_neurona_salida = 2

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
    red_0 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_0.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.1
    red_1 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neurona_salida, 0, 5, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.3
    red_2 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neurona_salida, 0, 10, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.5
    red_3 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neurona_salida, 0, 100, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Entreno red - Momentum = 0.7
    red_4 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neurona_salida, 0, 500, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 500)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()
    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Batch = 1')
    plt.plot(x1, y1, label='Batch = 5')
    plt.plot(x2, y2, label='Batch = 10')
    plt.plot(x3, y3, label='Batch = 100')
    plt.plot(x4, y4, label='Batch = 500')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej2_prueba_8_batch_entrenamiento.png")
    #plt.show()

    # Ploteo validacion
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    [x2, y2] = red_2.get_error_validacion_por_epoca()
    [x3, y3] = red_3.get_error_validacion_por_epoca()
    [x4, y4] = red_4.get_error_validacion_por_epoca()
    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Batch = 1')
    plt.plot(x1, y1, label='Batch = 5')
    plt.plot(x2, y2, label='Batch = 10')
    plt.plot(x3, y3, label='Batch = 100')
    plt.plot(x4, y4, label='Batch = 500')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej2_prueba_8_batch_validacion.png")
    #plt.show()


    print('Batch = 1 ---> TEST = %f' % (red_0.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Batch = 5 ---> TEST = %f' % (red_1.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Batch = 10 ---> TEST = %f' % (red_2.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Batch = 100 ---> TEST = %f' % (red_3.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Batch = 500 ---> TEST = %f' % (red_4.test_ej2(datos_testing_entrada, datos_testing_resultados)))

##########################
#EJECUCION DE LAS PRUEBAS#
##########################
variacion_batch()