import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt

def logistica_vs_tangente():
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

    # Entreno red_0
    red_0 = rn.RedNeuronal(cantidad_neuronas_entrada, [8, 8], cantidad_neurona_salida, 0, 1, 0, 0, 0)
    red_0.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada,
                   datos_validacion_resultados, 0.1, 300)

    # Entreno red_1
    red_1 = rn.RedNeuronal(cantidad_neuronas_entrada, [8, 8], cantidad_neurona_salida, 0, 1, 0, 1, 0)
    red_1.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada,
                   datos_validacion_resultados, 0.1, 300)

    # Ploteo entrenamiento
    [x0, y0] = red_0.get_error_entrenamiento_por_epoca()
    [x1, y1] = red_1.get_error_entrenamiento_por_epoca()
    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Sin parametros adaptativos')
    plt.plot(x1, y1, label='Con parametros adaptativos')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR ENTRENAMIENTO')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej2_prueba_1_param_funcion_de_activacion.png")
    # plt.show()

    # Ploteo validacion
    [x0, y0] = red_0.get_error_validacion_por_epoca()
    [x1, y1] = red_1.get_error_validacion_por_epoca()
    plt.cla()
    plt.clf()
    plt.plot(x0, y0, label='Función tangencial')
    plt.plot(x1, y1, label='Función logistica')
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR VALIDACION')
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False)
    plt.savefig("graficos\\ej2_prueba_1_param_funcion_de_activacion.png")
    # plt.show()

    # Imprimo nivel de testing
    print('Tangencial ---> TEST = %f' % (red_0.test_ej2(datos_testing_entrada, datos_testing_resultados)))
    print('Logistica ---> TEST = %f' % (red_1.test_ej2(datos_testing_entrada, datos_testing_resultados)))

##########################
# EJECUCION DE LAS PRUEBAS#
##########################
logistica_vs_tangente()
