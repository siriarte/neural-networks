import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def variacion_capas_ocultas():
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
    datos_validacion_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Test
    datos_testing_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento + tamanio_validacion:]
    datos_testing_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento + tamanio_validacion:]

    # Entreno red 1 capa 8 neuronas
    red_1 = rn.RedNeuronal(cantidad_neuronas_entrada, [8], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 2 capas 8 neuronas
    red_2 = rn.RedNeuronal(cantidad_neuronas_entrada, [8, 8], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 3 capas 8 neuronas
    red_3 = rn.RedNeuronal(cantidad_neuronas_entrada, [8, 8, 8], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 5 capas 8 neuronas
    red_4 = rn.RedNeuronal(cantidad_neuronas_entrada, [8, 8, 8, 8, 8], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, 0.05, 300)

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
    plt.savefig("graficos\\ej2_prueba_2_capas_entrenamiento.png")
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
    plt.savefig("graficos\\ej2_prueba_2_capas_validacion.png")
    # plt.show()

    # Imprimo nivel de testing
    print('1 capa oculta ---> TEST = %f' % (red_1.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('2 capa oculta  ---> TEST = %f' % (red_2.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('3 capa oculta  ---> TEST = %f' % (red_3.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('5 capa oculta  ---> TEST = %f' % (red_4.test_ej2(datos_testing_entrada, datos_testing_resultados)))


def variacion_neuronas_ocultas():
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
    datos_validacion = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Test
    datos_testing_entrada = datos_ejercicio['entrada'][tamanio_entrenamiento + tamanio_validacion:]
    datos_testing_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento + tamanio_validacion:]


    # Entreno red 1 capa 10 neuronas
    red_1 = rn.RedNeuronal(cantidad_neuronas_entrada, [2, 2], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 2 capas 10 neuronas
    red_2 = rn.RedNeuronal(cantidad_neuronas_entrada, [5, 5], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_2.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 3 capas 10 neuronas
    red_3 = rn.RedNeuronal(cantidad_neuronas_entrada, [8, 8], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_3.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Entreno red - 5 capas 10 neuronas
    red_4 = rn.RedNeuronal(cantidad_neuronas_entrada, [16, 16], cantidad_neurona_salida, 0, 1, 0, 0, 0, '')
    red_4.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.05, 300)

    # Ploteo entrenamiento
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    [x2, y2] = red_2.get_error_entrenamiento_por_epoca()
    [x3, y3] = red_3.get_error_entrenamiento_por_epoca()
    [x4, y4] = red_4.get_error_entrenamiento_por_epoca()

    plt.plot(x1, y1, label='2 Neuronas')
    plt.plot(x2, y2, label='5 Neuronas')
    plt.plot(x3, y3, label='8 Neuronas')
    plt.plot(x4, y4, label='16 Neuronas')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej2_prueba_2_neuronas_entrenamiento.png")
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
    plt.plot(x3, y3, label='8 Neuronas')
    plt.plot(x4, y4, label='16 Neuronas')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej2_prueba_2_neuronas_validacion.png")
    # plt.show()

    # Imprimo nivel de testing
    print('2 neuronas  ---> TEST = %f' % (red_1.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('5 neuronas  ---> TEST = %f' % (red_2.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('8 neuronas  ---> TEST = %f' % (red_3.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('16 neuronas  ---> TEST = %f' % (red_4.test_ej2(datos_testing_entrada, datos_testing_resultados)))

##########################
#EJECUCION DE LAS PRUEBAS#
##########################
#variacion_capas_ocultas()
variacion_neuronas_ocultas()