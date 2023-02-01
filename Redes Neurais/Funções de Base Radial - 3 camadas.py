import pandas as pd
import numpy as np

class rbf_3_layers:
    
    def __init__(self, n1, n2, n3):
        # Número de elementos em cada camada
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
    
    def __repr__(self):
    # Constrói e retorna uma string que representa a
    # arquitetura da rede neural
        return "NeuralNetwork: {}".format(
                "-".join([str(self.n1), str(self.n2), str(self.n3)]))

    
    def correction_1(self):
        
        current_groups = np.zeros(len(self.train_samples_x))
        
        while True:
            
            previous_groups = np.copy(current_groups)
        
            for i in range(len(self.train_samples_x)):

                euclidian_distance = np.full(self.n2, 0, dtype=float)

                for j in range(self.n2):

                    for k in range(self.n1):
                        
                        euclidian_distance[j] = euclidian_distance[j] + (self.train_samples_x[i][k] - self.w[0][j][k])**2

                    euclidian_distance[j] = euclidian_distance[j]**(0.5)

                current_groups[i] = np.where(euclidian_distance == min(euclidian_distance))[0][0]
            
            for i in range(self.n2):

                weight_update = np.zeros(self.n1)

                for j in range(len(current_groups)):

                    if (current_groups[j] == i):

                        weight_update = weight_update + self.train_samples_x[j]

                weight_update = weight_update/np.count_nonzero(current_groups == i)

                for j in range(self.n1):

                    self.w[0][i][j] = np.copy(weight_update[j])
            
            if ((previous_groups == current_groups).all()):
                break
        
        var_ativation = np.zeros(self.n2)
            
        for i in range(self.n2):
                
            for j in range(len(current_groups)):
                    
                if (current_groups[j] == i):
                    
                    for k in range(self.n1):

                        var_ativation[i] = var_ativation[i] + (self.train_samples_x[j][k] - self.w[0][i][k])**2
                            
            var_ativation[i] = var_ativation[i]/np.count_nonzero(current_groups == i)
            
        self.var_ativation = var_ativation

        
    def forward_1(self, train_samples_x):
        
        y1 = np.zeros(self.n2)
        
        for i in range(self.n2):
            
            for j in range(self.n1):
                
                y1[i] = y1[i] + (train_samples_x[j] - self.w[0][i][j])**2
            
            y1[i] = y1[i]/(2*(self.var_ativation[i]))
            
            y1[i] = np.exp(-y1[i])
        
        y1 = np.insert(y1, 0, -1, axis = 0)
        
        return y1

        
    def forward_2(self, y1):
        
        y2 = np.atleast_2d(self.w[1]@y1.T)
        
        return(y2)
    

    def correction_2(self, y1, y2, train_samples_y):
        
        y1 = np.atleast_2d(y1)
        
        grad2 = np.atleast_2d(train_samples_y - y2)
        
        self.w[1] = np.atleast_2d(self.w[1] + self.learn_rate*grad2@y1)

    
    def eqm(self):
        
        eqm = 0
        
        for i in range(len(self.train_samples_x)):
            
            y1 = self.forward_1(self.train_samples_x[i])
            
            y2 = self.forward_2(y1)
            
            eqm = eqm - (((self.train_samples_y[i] - y2)**2)/2).sum()
            
        eqm = eqm/len(self.train_samples_x)
        
        return eqm


    def train(self, train_samples_x, train_samples_y, learn_rate, precision):
        
        self.train_samples_x = np.copy(train_samples_x)
        self.train_samples_y = np.copy(train_samples_y)
        self.learn_rate = learn_rate
        self.precision = precision
        
        self.w = []
        self.w.append(np.copy(self.train_samples_x[:self.n2, :]))
        self.w.append(np.random.default_rng().uniform(-(2.4/self.n1), (2.4/self.n1),(self.n3, self.n2+1)))
        
        self.correction_1()
        
        self.epoch = 1
        
        print('The inicial weight matrices are:')
        print(self.w)
        print('\n')
        
        while True:
            
            if (self.epoch==1)|((self.epoch%1000)==0):
                print('\n')
                print('######################################')
                print('Starting a new epoch number %i' %(self.epoch))
                print('\n')

            previous_eqm = self.eqm()
            
            for i in range(len(self.train_samples_y)):
                
                y1 = self.forward_1(self.train_samples_x[i])
            
                y2 = self.forward_2(y1)
            
                self.correction_2(y1, y2, self.train_samples_y[i])
                
            current_eqm = self.eqm()
            
            self.current_eqm = current_eqm
            
            if (self.epoch==1)|((self.epoch%1000)==0):
                print('EQM difference:')
                print(current_eqm - previous_eqm)
                
            if (abs(current_eqm - previous_eqm) <= precision):
                print("Training finished in %s epochs" % (self.epoch))
                print('\n')
                print("The final eqm was %f" % (self.current_eqm))
                print('\n')
                print('The final weight matrices are:')
                print(self.w)
                break
            
            self.epoch = self.epoch + 1


    def predict(self, variables):
        
        resultados = np.zeros((len(variables),self.n3), dtype = float)
        
        for i in range(len(variables)):
            
            y1 = self.forward_1(variables[i])
            
            y2 = self.forward_2(y1)
        
            for j in range(len(y2)):
            
                resultados[i][j] = y2[j]
        
        return resultados
