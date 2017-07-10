# Ejecuta todas las combinaciones de pruebas y genera los archivos CVS
# de la matriz de pesos de cada una
import red_hebbian as hebbian
import red_som as som


# Algoritmo de pruebas
def pruebas_red_hebbiana():
        for filas_salida in [10]:
            for sigma in [5, 3]:
                for epocas in [100, 300]:
                    for eta in [0.5, 0.1]:
                        categorias, data_set_entrada = hebbian.parsear_dataset()
                        data_centrado = hebbian.centrar_matriz(data_set_entrada)
                        tamanio_entrenamiento = int(len(data_centrado) * 0.9)

                        datos_entrenamiento_entrada = data_centrado[:tamanio_entrenamiento]
                        datos_validacion_entrada = data_centrado[tamanio_entrenamiento:]
                        categorias_entrenamiento = categorias[:tamanio_entrenamiento]
                        categorias_validacion = categorias[tamanio_entrenamiento:]

                        red = som.PerceptronSOM(len(datos_entrenamiento_entrada[0]), filas_salida, filas_salida)
                        red.train_som(datos_entrenamiento_entrada, eta, sigma, epocas)

                        str_eta = str(eta)[2:]
                        archivo_salida_cvs = r'matrices_de_pesos/m_som_%d_%d_%s_%d.csv' % (filas_salida, epocas, str_eta, sigma)
                        archivo_salida_img_entrenamiento = r'imagenes/som_%d_%d_%s_%d_train.png' % (filas_salida, epocas, str_eta, sigma)
                        archivo_salida_img_validacion = r'imagenes/som_%d_%d_%s_%d_val.png' % (filas_salida, epocas, str_eta, sigma)

                        #red.setear_matriz_pesos(archivo_salida_cvs)
                        red.salvar_matriz_pesos(archivo_salida_cvs)
                        red.calcular_y_graficar(datos_entrenamiento_entrada, categorias_entrenamiento, archivo_salida_img_entrenamiento)
                        red.calcular_y_graficar(datos_validacion_entrada, categorias_validacion, archivo_salida_img_validacion)


# Ejecutar pruebas
pruebas_red_hebbiana()
