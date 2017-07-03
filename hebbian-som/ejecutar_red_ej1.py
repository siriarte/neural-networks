import red_hebbian as hebbian
from random import seed
import sys
import ast

def main():
    seed(1)
    if len(sys.argv)!=3 and len(sys.argv)!=5:
        print("Se tienen que setear los 2 o 5 parametros")
        return 0

    # Parseo argumentos
    if len(sys.argv)==3:
        archivo_pesos = str(sys.argv[1])
        cantidad_neuronas_salida = int(sys.argv[2])
    else:
        algoritmo = str(sys.argv[1])
        eta = float(sys.argv[2])
        epocas = int(sys.argv[3])
        cantidad_neuronas_salida = int(sys.argv[4])
        archivo_pesos = ''

    # Parseo dataset
    categorias, data_set_entrada = hebbian.parsear_dataset()

    # Tamaño de los sets, entrenamiento = 90% validacion = 10%
    tamanio_entrenamiento = int(len(data_set_entrada) * 0.9)

    # Entrenamiento
    datos_entrenamiento_entrada = data_set_entrada[:tamanio_entrenamiento]
    categorias_entrenamiento = categorias[:tamanio_entrenamiento]

    # Validacion
    datos_validacion_entrada = data_set_entrada[tamanio_entrenamiento:]
    categorias_validacion = categorias[tamanio_entrenamiento:]

    # Creo la red
    red = hebbian.Hebbian(len(data_set_entrada[0]), cantidad_neuronas_salida)

    # Entreno o cargo archivo de pesos
    if archivo_pesos=='':
        red.entrenar(datos_entrenamiento_entrada, [], epocas, eta, algoritmo)
    else:
        red.setear_matriz_pesos(archivo_pesos)

    # Grafico
    resultados_entrenamiento = red.testear_red(datos_entrenamiento_entrada, categorias_entrenamiento)
    hebbian.Hebbian.graficar(resultados_entrenamiento, categorias_entrenamiento)
    resultados_validacion = red.testear_red(datos_validacion_entrada, categorias_validacion)
    hebbian.Hebbian.graficar(resultados_validacion, categorias_validacion)

# Ejecuto la funcion principal
main()