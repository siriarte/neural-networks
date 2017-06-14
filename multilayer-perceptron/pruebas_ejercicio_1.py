import red_neuronal as rn
import parser_entrada as parser_entrada
from random import seed
import matplotlib.pyplot as plt

def fx_umbral(umbral,valor):
    if(valor>umbral):
        return 1
    else:
        return 0

def entrenamiento_basico():
    seed(1)
    datos_ejercicio = parser_entrada.datos_ejercicio_1()
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada'])*0.5)
    datos_entrenamiento = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]
    tamanio_entrada = len(datos_ejercicio['entrada'][0])
    tamanio_salida = 1
    datos_validacion = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]
    red = rn.RedNeuronal(tamanio_entrada, [10,10], tamanio_salida, 0, 1, 0, 0, 0,  'test_1.out')

    red.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.1, 100)



    sum = 0
    for i in range(0, len(datos_validacion)):
        valor = red.forward_propagation(datos_validacion[i])
        valor_binarizado = fx_umbral(0.5,valor[0])
        valor_verif_binarizado = fx_umbral(0.5,datos_validacion_resultados[i][0])
        if(valor_binarizado==valor_verif_binarizado): sum+=1
        print('{}    {}'.format(fx_umbral(0.5,valor[0]), fx_umbral(0.5,datos_validacion_resultados[i][0])))


    print(sum/len(datos_validacion))


def entrenamiento_basico_2():
    seed(1)
    datos_ejercicio = parser_entrada.datos_ejercicio_2()
    tamanio_entrenamiento = int(len(datos_ejercicio['entrada'])*0.5)
    datos_entrenamiento = datos_ejercicio['entrada'][:tamanio_entrenamiento]
    datos_entrenamiento_resultados = datos_ejercicio['resultado'][:tamanio_entrenamiento]
    tamanio_entrada = len(datos_ejercicio['entrada'][0])
    tamanio_salida = 2
    datos_validacion = datos_ejercicio['entrada'][tamanio_entrenamiento:]
    datos_validacion_resultados = datos_ejercicio['resultado'][tamanio_entrenamiento:]
    red = rn.RedNeuronal(tamanio_entrada, [10,10], tamanio_salida,0,20,0,0,0)

    red.entrenar(datos_entrenamiento, datos_entrenamiento_resultados, datos_validacion, datos_validacion_resultados, 0.3, 250)




entrenamiento_basico_2()


