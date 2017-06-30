# -*- coding: utf-8 -*-


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


class PerceptronNoSup:
    def __init__(self,neuronasPorCapa):#,fun_act,der_act):
        #self.funAct= np.vectorize(funciones) # Paso en una lista las funciones de activacion
        #self.DerFunAct = np.vectorize(derivadas)
        #self.derivadas = derivadas     # Paso en una lista las funciones de activacion   
        #self.cantCapas = CantCapas 
        CantCapas =  len(neuronasPorCapa)
        self.w_= []
        self.bias = []
        cant_filas_pesos = neuronasPorCapa[1]        
        cant_col_pesos = neuronasPorCapa[0]
        for i in range(1,CantCapas):
            #peso = np.random.rand(l_neuronasPorCapa[i],l_neuronasPorCapa[i-1])       
            peso = np.random.uniform(low=-0.6, high=0.6, size=(cant_filas_pesos,cant_col_pesos))
            peso = np.matrix(peso)
            self.w_.append(peso)

            
            
    def train_oja(self, trainX, eta=0.01, epochs=10000):
        self.eta = eta
        self.epochs = epochs
        for _ in range(self.epochs):
            for x in trainX:
                x_ = np.matrix(x).T
                y = self.w_[0]*x_
                y = np.matrix(y)
                a = x_ - self.w_[0].T * y
                delta = y*(a.T)
                self.w_[0] = self.w_[0] + eta*delta
                #eta = eta/2
        return self
    
#==============================================================================
#     def train_oja_ver2(self, trainX, eta=0.01, epochs=1000):
#         self.eta = eta
#         self.epochs = epochs
#         for _ in range (self.epochs):
#             for x in trainX:                    
#                 x_ = np.matrix(x).T
#                 y = self.w_[0]*x_
#                 segundo_termino = (y.T)*self.w_[0] # W tiene m filas (ooutputs) y p columnas (inputs)
#                 primer_termino = x_
#                 resta = primer_termino-segundo_termino
#                 delta =y*(resta)
#                 delta_eta = eta*delta
#                 self.w_[0] =  self.w_[0] + delta_eta
#         return self
#==============================================================================
            
        
    def train_sanger(self, trainX, eta=0.01, epochs=100):
        self.eta = eta
        self.epochs = epochs
        #self.errors_ = []
        for _ in range(self.epochs):
            for x in trainX:
                x_ = np.matrix(x).T
                y = self.w_[0]*x_
                y = np.matrix(y)
                cant_filas = y.shape[0]
                cant_col = x_.shape[0]
                for j in range(0,cant_filas):
                    for i in range(0,cant_col):
                        #eta = eta/2 
                        suma_i_j = self.w_[0][0:j+1,i]
                        suma_i_j = suma_i_j.T
                        y_j = y[0:j+1]
                        self.w_[0][j,i] = self.w_[0][j,i] + eta*(x_[i] - suma_i_j*y_j )
        return self
                
        # Empieza a calcular la salida del perceptron simple.
    def net_input(self, X):
        cantidad = X.shape[0]
        l = []
        print cantidad
        for i in range(cantidad-1,-1,-1):
            X_ = np.matrix(X.iloc[i]).T
            net_1 = self.w_[0]*X_
            net_1 = net_1.T
            net_1_lista =  net_1.tolist()
            l.append(net_1_lista)
#        net = self.funAct(net_1)
#        for i in range(1,cantidad-1):
#            net = self.w_[i]*net - self.bias[i].T
#            net = self.funAct(net)
        return l

    def net_input_lista(self, X):
        cantidad = len(X)
        l = []
        print cantidad
        for i in range(cantidad-1,-1,-1):
            X_ = np.matrix(X[i]).T
            net_1 = self.w_[0]*X_
            net_1 = net_1.T
            net_1_lista =  net_1.tolist()
            l.append(net_1_lista)
#        net = self.funAct(net_1)
#        for i in range(1,cantidad-1):
#            net = self.w_[i]*net - self.bias[i].T
#            net = self.funAct(net)
        return l
                
    def predict(self, X):
        #X = np.append(1,X)
        a = self.net_input(X)
        return a
        
    def predict_lista(self, X):
        #X = np.append(1,X)
        a = self.net_input_lista(X)
        return a
        


        
#%%
