import red_som as som
import red_hebbian as habbian
from random import seed
import sys
import ast

def main():
    seed(1)
    if len(sys.argv)!=4 and len(sys.argv)!=6:
        print("Se tienen que setear los 4 o 5 parametros")
        return 0

    # Parseo argumentos
    if len(sys.argv)==4:
        archivo_pesos = str(sys.argv[1])
        filas_salida = int(sys.argv[2])
        columnas_salida = int(sys.argv[3])
    else:
        filas_salida = int(sys.argv[1])
        columnas_salida = int(sys.argv[2])
        eta = float(sys.argv[3])
        sigma = float(sys.argv[4])
        epocas = int(sys.argv[5])
        archivo_pesos = ''

    # Parseo dataset
    categorias, data_set_entrada = habbian.parsear_dataset()

    # Tamaño de los sets, entrenamiento = 100% validacion = 0%
    tamanio_entrenamiento = int(len(data_set_entrada) * 0.4)

    # Entrenamiento
    datos_entrenamiento_entrada = data_set_entrada[:tamanio_entrenamiento]
    categorias_entrenamiento = categorias[:tamanio_entrenamiento]

    # Validacion
    datos_validacion_entrada = data_set_entrada[tamanio_entrenamiento:]
    categorias_validacion = categorias[tamanio_entrenamiento:]

    # Creo la red
    red = som.PerceptronSOM(len(datos_entrenamiento_entrada[0]), filas_salida, columnas_salida)

    # Entreno o cargo archivo de pesos
    if archivo_pesos == '':
        red.train_som(datos_entrenamiento_entrada, eta, sigma, epocas)
    else:
        red.setear_matriz_pesos(archivo_pesos)
        return

    # Grafico
    red.calcular_y_graficar(datos_entrenamiento_entrada, categorias_entrenamiento)
    #resultados_entrenamiento = red.testear_red(datos_entrenamiento_entrada, categorias_entrenamiento)
    #hebbian.Hebbian.graficar(resultados_entrenamiento, categorias_entrenamiento)
    #resultados_validacion = red.testear_red(datos_validacion_entrada, categorias_validacion)
    #hebbian.Hebbian.graficar(resultados_validacion, categorias_validacion)

# Ejecuto la funcion principal
main()