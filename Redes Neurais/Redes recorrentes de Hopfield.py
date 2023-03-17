import pandas as pd
import numpy as np

# Criando a classe para representar a rede recorrente de hopfield
class RRH:        
    
    # Definindo a função para realizar o treinamento
    def train_ep(self, train_data):
        
        train_data_updated = []
        for i in range(len(train_data)):
            
            train_data_updated.append((np.atleast_2d(train_data[i])).astype('float'))
        
        n = train_data_updated[0].shape[0]*train_data_updated[0].shape[1]
        
        self.w = np.atleast_2d(np.zeros((n, n)))
        
        for i in range(len(train_data)):
            
            self.w = self.w + train_data_updated[i]@train_data_updated[i].T
        
        self.w = self.w/n
        
        self.w = self.w - (len(train_data_updated)/n)*np.identity(n)
    
    # Definindo a função para realizar a previsão à partir do modelo já treinado
    def predict(self, variables):
        
        variables_updated = []
        
        for i in range(len(variables)):
            
            variables_updated.append((np.atleast_2d(variables[i])).astype('float'))
            
            current_v = variables_updated[i]
            
            while True:
                
                previous_v = current_v
                
                u = np.atleast_2d(self.w@previous_v.T)
                
                for j in range(len(u)):
                        
                    current_v[0][j] = (1-np.exp((-100)*u[j]))/(1+np.exp((-100)*u[j]))
                
                if (all(previous_v[0] == current_v[0])):
                    break

        return current_v
