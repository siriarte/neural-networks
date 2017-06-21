import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import sys
import ast


def main():
    seed(1)
    if len(sys.argv) != 11:
        print("Se tienen que setear los 10 parametros necesarios")
        return 0

    ejercicio = int(sys.argv[1])
    momentum = float(sys.argv[2])
    batch = int(sys.argv[3])
    early_stopping = float(sys.argv[4])
    param_adaptativos = int(sys.argv[5])
    fx_activacion = int(sys.argv[6])
    distribucion_pesos = int(sys.argv[7])
    eta = float(sys.argv[8])
    capas = ast.literal_eval(sys.argv[9])
    epocas = int(sys.argv[10])

    if ejercicio == 1:
        datos_ejercicio = parser_entrada.datos_ejercicio_1()
        cantidad_neuronas_entrada = 10
        cantidad_neurona_salida = 1
    elif ejercicio == 2:
        datos_ejercicio = parser_entrada.datos_ejercicio_2()
        cantidad_neuronas_entrada = 8
        cantidad_neurona_salida = 2
    else:
        print("Indique correctamente el numero de ejercicio")
        return 0

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

    # Entreno red y ejecuto testing
    red = rn.RedNeuronal(cantidad_neuronas_entrada, capas, cantidad_neurona_salida, momentum, batch, early_stopping, param_adaptativos, fx_activacion, '', distribucion_pesos)
    red.entrenar(datos_entrenamiento_entrada, datos_entrenamiento_resultados, datos_validacion_entrada, datos_validacion_resultados, eta, epocas)

    print("\n\n")
    if ejercicio == 1:
        print("PERFOMANCE EN TESTING = %F" % (red.test_ej1(datos_testing_entrada, datos_testing_resultados)))
    else:
        print("PERFOMANCE EN TESTING = %F" % (red.test_ej2(datos_testing_entrada, datos_testing_resultados)))


# Ejecuto la funcion principal
main()