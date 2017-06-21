import capa_neuronal as capa
import fx_auxiliares as fx_auxiliares
import numpy as np


# Clase RedNeuronal
class RedNeuronal(object):

    # Constructor de la Red Neuronal
    def __init__(self, cantidad_neuronas_entrada, neuronas_de_capas_ocultas, cantidad_neuronas_salida,
                 momentum=0, tamano_batch=1, early_stop=0, parametros_adaptativos=0, fx_activacion = 0, log_en_archivo='', distribucion_pesos = 0):

        # Seteo de variables
        self.bias = 1.0
        self.momentum = momentum
        self.tamano_batch = tamano_batch
        self.cantidad_capas = len(neuronas_de_capas_ocultas) + 2
        self.early_stop = early_stop
        self.parametros_adaptativos = parametros_adaptativos
        self.log_en_archivo = log_en_archivo
        self.error_entrenamiento_por_epoca = []
        self.error_validacion_por_epoca = []
        self.distribucion_pesos = distribucion_pesos

        # Funcion de activacion
        if fx_activacion == 0:
            self.funcion_de_activacion = fx_auxiliares.fx_tangente
            self.funcion_de_activacion_prima = fx_auxiliares.fx_tangente_prima
        else:
            self.funcion_de_activacion = fx_auxiliares.fx_logistica
            self.funcion_de_activacion_prima = fx_auxiliares.fx_logistica_prima

        # Genero las capas
        self.capas = []

        # Primera capa oculta n=0
        self.capas.append(capa.CapaNeuronal(cantidad_neuronas_entrada, neuronas_de_capas_ocultas[0], self.distribucion_pesos))

        # Capas ocultas de n=1...k
        for i in range(1, len(neuronas_de_capas_ocultas)):
            self.capas.append(capa.CapaNeuronal(neuronas_de_capas_ocultas[i - 1], neuronas_de_capas_ocultas[i], self.distribucion_pesos))

        # Capa de salida n=k+1
        self.capas.append(capa.CapaNeuronal(neuronas_de_capas_ocultas[-1], cantidad_neuronas_salida, self.distribucion_pesos))

    # Calcula la cantidad de neuronas de la red
    def cantidad_axones_red(self):
        ret = 0
        for capa in self.capas:
            ret += capa.cantidad_neuronas * capa.cantidad_neuronas_capa_anterior
        return ret

    # Sumador
    def fx_sumatoria(self, pesos, valores_entrada):

        # Inicializo las variables
        ret = 0

        # Sumo pesos x valores
        for i in range(len(pesos) - 1):
            ret = ret + (pesos[i] * valores_entrada[i])

        # Bias
        ret += pesos[-1] * self.bias

        # Retorno la suma
        return ret

    # Algoritmo de progacion hacia adelante
    def forward_propagation(self, valores_entrada):

        # Seteo la entrada
        entrada = valores_entrada

        # Itero todas las capas
        for capa in self.capas:

            # Inicio [] los nuevos valores de entrada
            valores_entrada_nuevos = []

            # Feed fordward
            for j in range(0, capa.cantidad_neuronas):

                # Aplico la sumatoria de pesos x valores de entrada
                s = self.fx_sumatoria(capa.pesos[j], entrada)

                # Seteo la salida con la funcion de activacion al calculo anterior
                capa.salida[j] = self.funcion_de_activacion(s)

                # Almaceno el resultado en la lista
                valores_entrada_nuevos.append(capa.salida[j])

            # Nueva entrada para la siguiente capa a partir de lo calculado
            entrada = valores_entrada_nuevos

        # Retorno el valor en la capa de salida
        return entrada

    # Algoritmo de retro-propagacion
    def back_propagation(self, resultado_esperado):

        # Itero las capas desde la de salida a la de entrada
        iterador_capas = len(self.capas) - 1

        # Ciclo
        while iterador_capas >= 0:
            # Seteo variables
            lista_errores = []
            capa_actual = self.capas[iterador_capas]

            # Para toda neurona de la capa actual propago el error desde la siguiente
            if iterador_capas == len(self.capas) - 1:
                # Si es la ultima
                for j in range(capa_actual.cantidad_neuronas):
                    neurona_salida = capa_actual.salida[j]
                    lista_errores.append(resultado_esperado[j] - neurona_salida)
            else:
                # Si no es la ultima
                for j in range(capa_actual.cantidad_neuronas):
                    error = 0.0
                    capa_siguiente = self.capas[iterador_capas + 1]
                    for k in range(0, capa_siguiente.cantidad_neuronas):
                        peso_neurona = capa_siguiente.pesos[k][j]
                        delta_neurona = capa_siguiente.delta[k]
                        error += peso_neurona * delta_neurona
                    lista_errores.append(error)

            # Actualizo los deltaW
            for j in range(capa_actual.cantidad_neuronas):
                capa_actual.delta[j] = lista_errores[j] * self.funcion_de_activacion_prima(capa_actual.salida[j])

            # Paso a la siguiente a la capa anterior
            iterador_capas = iterador_capas - 1

    # Actualiza los pesos de la red
    def actualizar_pesos(self, valor_de_entrada, eta):
        for i in range(len(self.capas)):
            entrada = valor_de_entrada
            if i!=0:
                entrada = self.capas[i-1].salida
            for j in range(self.capas[i].cantidad_neuronas):
                for k in range(len(entrada)):
                    delta_nuevo = eta * self.capas[i].delta[j] * entrada[k]
                    self.capas[i].pesos[j][k] += delta_nuevo
                    if self.momentum:
                        delta_nuevo += self.capas[i].delta_anterior[j] * + self.momentum
                        self.capas[i].delta_anterior[j] = delta_nuevo
                self.capas[i].pesos[j][-1] += eta * self.capas[i].delta[j]

    # Se utiliza cuando no es estocastico
    def calcular_pesos(self, valor_de_entrada, eta):

        # Inicio la veriable a retornar
        pesos = []

        # Ciclo de calculo
        for i in range(len(self.capas)):
            entrada = valor_de_entrada
            if i != 0:
                entrada = self.capas[i - 1].salida
            for j in range(self.capas[i].cantidad_neuronas):
                for k in range(len(entrada)):
                    delta_nuevo = eta * self.capas[i].delta[j] * entrada[k]
                    pesos.append(delta_nuevo)
                    if self.momentum:
                        delta_nuevo += self.capas[i].delta_anterior[j] * self.momentum
                        self.capas[i].delta_anterior[j] = delta_nuevo
                pesos.append(eta * self.capas[i].delta[j])

        # Retorno la lista de pesos
        return pesos

    def calcular_error_validacion(self, set_validacion, salida_validacion):
        # Error cuadratico por epoca
        suma_errores = 0

        for i in range(len(set_validacion)):

            # Lista de errores
            error = []

            # Calculo el valor de salida de la red
            salida_red = self.forward_propagation(set_validacion[i])
            valor_esperado = salida_validacion[i]

            # Agrego el error a la lista
            if isinstance(salida_validacion[i], tuple):
                for i, esperada in enumerate(valor_esperado[i]):
                    error.append((esperada - salida_red[i]) ** 2)
            else:
                error.append((valor_esperado[0] - salida_red[0]) ** 2)

            # Sumo el error
            suma_errores = sum(error) + suma_errores

        error_final = suma_errores / 2 / len(set_validacion)
        return error_final


    # Entrenador de la Red Neuronal
    def entrenar(self, set_entrenamiento, salida_entramiento, set_validacion, salida_validacion, eta, numero_de_epocas):

        # Ciclos para parametros adapativos
        ciclos_parametros_adaptativos = 0

        # Guarda el error de entrenamiento por epoca
        self.error_entrenamiento_por_epoca = []

        # Guarda el error de validacion por epoca
        self.error_validacion_por_epoca = []

        for epoca in range(numero_de_epocas):

            # Error cuadratico por epoca
            delta_error = 0

            # Creo la lista de pesos historicos vacia
            pesos_historial = [[] for i in range(self.cantidad_axones_red())]

            # Ciclos para batch
            ciclos_batch = 0
            
            for i in range(len(set_entrenamiento)):

                # Lista de errores
                error = []

                # Calculo el valor de salida de la red
                salida_red = self.forward_propagation(set_entrenamiento[i])
                valor_esperado = salida_entramiento[i]

                # Agrego el error a la lista
                for k in range(len(valor_esperado)):
                    error.append((valor_esperado[k] - salida_red[k]) ** 2)

                # Retro-propagacion
                self.back_propagation(valor_esperado)

                # Si no es estocastico y no se cumplieron los ciclos necesario
                # me guardo los promedios de los pesos hasta actualizarlos
                if self.tamano_batch != 1:
                    pesos = self.calcular_pesos(set_entrenamiento[i], eta)
                    for i in range(len(pesos)):
                        pesos_historial[i].append(pesos[i])
                    ciclos_batch += 1

                    # Si no es estocastico y se cumplio la cantidad de ciclos del tamanio del batch
                    # actualizo con los pesos promedio
                    if ciclos_batch == self.tamano_batch:
                        # Calculo el promedio de los pesos
                        pesos_promedio = []
                        for lista_pesos in pesos_historial:
                            pesos_promedio.append(sum(lista_pesos)/len(lista_pesos))

                        #Actualizo pesos
                        numero_neurona = 0
                        for capa in self.capas:
                            for i in range(capa.cantidad_neuronas):
                                for j in range(len(capa.pesos[i])):
                                    capa.pesos[i][j] += pesos_promedio[numero_neurona]
                                    numero_neurona = numero_neurona + 1
                        ciclos_batch = 0
                # Si es estocastico actualizo los pesos
                else:
                    self.actualizar_pesos(set_entrenamiento[i], eta)

                # Sumo error
                delta_error += sum(error)

            # Promedio el error
            delta_error = delta_error / 2 / len(set_entrenamiento)

            # Error validacion
            validacion_error = self.calcular_error_validacion(set_validacion, salida_validacion)

            # Guardo el error de validacion x epoca
            self.error_validacion_por_epoca.append([epoca, validacion_error])

            # Imprimo resultado
            print('->EPOCA=%d, ETA=%.3f, ERROR=%.3f, ERROR_VAL=%0.3f, MOMENTUM=%.3f, BATCH=%d, EARLY-STOP=%.2f, PARAMETROS ADAPTATIVOS=%d'
                  % (epoca, eta, delta_error, validacion_error,self.momentum, self.tamano_batch, self.early_stop, self.parametros_adaptativos))

            # Si parametros adaptativos está activado y pasó la primera epoca
            if self.parametros_adaptativos and epoca > 0:
                if self.parametros_adaptativos == ciclos_parametros_adaptativos:
                    if delta_error < delta_error_anterior:
                        eta+=eta*0.01
                    else:
                        eta-=eta*0.5
                    ciclos_parametros_adaptativos = 0
                    delta_error_anterior = delta_error
                else:
                    ciclos_parametros_adaptativos+=1
                    delta_error_anterior = delta_error
            # Si es la primera epoca me guardo el error
            elif self.parametros_adaptativos and epoca == 0:
                delta_error_anterior = delta_error
                ciclos_parametros_adaptativos+=1

            # Si esta activado early_stop lo verifico
            if self.early_stop:
                # error = self.calcular_error_validacion(set_validacion, salida_validacion)
                # print ("ERROR VALIDACION   %F" % (error))
                if validacion_error <= self.early_stop:
                    print("--------EARLY STOP BREAK-------------")
                    break

            # Guardo en el registro
            self.error_entrenamiento_por_epoca.append([epoca, delta_error])

        # Si esta activada la opción dejo el resultado en un archivo
        if self.log_en_archivo!='':
            fx_auxiliares.guardar_log_en_archivo(self.log_en_archivo, self.error_entrenamiento_por_epoca)


    def get_error_entrenamiento_por_epoca(self):
        x = []
        y = []
        for elem in self.error_entrenamiento_por_epoca:
            x.append(elem[0])
            y.append(elem[1])

        return [x,y]

    def get_error_validacion_por_epoca(self):
        x = []
        y = []
        for elem in self.error_validacion_por_epoca:
            x.append(elem[0])
            y.append(elem[1])

        return [x,y]

    def test_ej1(self, datos_testing_entrada, datos_testing_resultado):
        resultados_correctos = 0

        for i in range(len(datos_testing_entrada)):

            # Calculo el valor de salida de la red
            salida_red = self.forward_propagation(datos_testing_entrada[i])
            valor_salida = fx_auxiliares.fx_umbral(0.5, salida_red)
            valor_esperado = datos_testing_resultado[i]

            if valor_esperado[0] == valor_salida:
                resultados_correctos+=1


        return resultados_correctos / len(datos_testing_entrada)

    def test_ej2(self, datos_testing_entrada, datos_testing_resultado):
        # Error cuadratico por epoca
        resultados_correctos = 0
        epsilon = 0.01

        for i in range(len(datos_testing_entrada)):

            # Calculo el valor de salida de la red
            salida_red = self.forward_propagation(datos_testing_entrada[i])
            valor_esperado = datos_testing_resultado[i]
            suma_error = []
            for i in range(len(salida_red)):
                suma_error.append((valor_esperado[0] - salida_red[0]) ** 2)

            if sum(suma_error)/2 <= epsilon:
                resultados_correctos+=1


        return resultados_correctos / len(datos_testing_entrada)


