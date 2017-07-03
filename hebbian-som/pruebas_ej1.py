# Ejecuta todas las combinaciones de pruebas y genera los archivos CVS
# de la matriz de pesos de cada una
import red_hebbian as hebbian

# Algoritmo de pruebas
def pruebas_red_hebbiana():
    for algoritmo in ['oja', 'sanger']:
        for neuronas_salida in [3, 9]:
            for epocas in [100, 200, 300]:
                for eta in [0.0001, 0.001, 0.01]:
                    categorias, dataset = hebbian.parsear_dataset()
                    data_centrado = hebbian.centrar_matriz(dataset)
                    red = hebbian.Hebbian(len(dataset[0]), neuronas_salida)
                    str_eta = str(eta)[2:]
                    red.entrenar(data_centrado, [], epocas, eta, algoritmo)
                    archivo_salida_cvs = r'matrices_de_pesos/m_%s_%d_%s_%d.csv' % (algoritmo, epocas, str_eta, neuronas_salida)
                    red.salvar_matriz_pesos(archivo_salida_cvs)


# Ejecutar pruebas
pruebas_red_hebbiana()
