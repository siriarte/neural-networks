# Ejecuta todas las combinaciones de pruebas y genera los archivos CVS
# de la matriz de pesos de cada una
import red_hebbian as hebbian
import red_som as som

# Algoritmo de pruebas
def pruebas_red_hebbiana():
        for filas_salida in [7, 10]:
            for epocas in [1, 500,1000]:
                for eta in [0.001, 0.01, 0.1]:
                    categorias, dataset = hebbian.parsear_dataset()
                    data_centrado = hebbian.centrar_matriz(dataset)
                    red = som.PerceptronSOM(len(dataset[0]), filas_salida, filas_salida)
                    str_eta = str(eta)[2:]
                    red.train_som(dataset, eta, 3, epocas)
                    archivo_salida_cvs = r'matrices_de_pesos/m_som_%d_%d_%s_%d.csv' % (filas_salida, epocas, str_eta, 3)
                    red.salvar_matriz_pesos(archivo_salida_cvs)
                    red.calcular_y_graficar(dataset, categorias)


# Ejecutar pruebas
pruebas_red_hebbiana()
