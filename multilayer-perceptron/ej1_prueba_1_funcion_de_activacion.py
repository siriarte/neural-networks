import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt


def logistica_vs_tangente():
    seed(1)

    # Parseo datos
    datos_ejercicio = parser_entrada.datos_ejercicio_1()
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada'])*0.5)
    datos_entrenamiento = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]
    tamanio_entrada = len(datos_ejercicio['entrada'][0])
    tamanio_salida = len(datos_ejercicio['resultado'][0])
    datos_validacion = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]

    # Entreno red para logistica
    red_1 = rn.RedNeuronal(tamanio_entrada, [10,10], tamanio_salida, 0, 1, 0, 0, 0, '')
    red_1.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Entreno red para tangente
    red_2 = rn.RedNeuronal(tamanio_entrada, [10,10], tamanio_salida, 0, 1, 0, 0, 1, '')
    red_2.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 300)

    # Ploteo
    [x1, y1] = red_1.get_error_por_epoca()
    [x2, y2] = red_2.get_error_por_epoca()
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlabel('EPOCAS')
    plt.ylabel('ERROR')
    plt.grid(True)
    plt.savefig("graficos\\ej1_prueba_1_funcion_de_activacion.png")
    #plt.show()


logistica_vs_tangente()