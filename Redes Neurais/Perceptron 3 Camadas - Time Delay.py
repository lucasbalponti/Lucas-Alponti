import pandas as pd
import numpy as np

class pmc_3_layers:
    
    def __init__(self, n1, n2, n3):
        # Número de elementos em cada camada
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
        # Inicializando pesos com valores aleatórios (LeCun)
        self.w = []
        self.w.append(np.random.default_rng().uniform(-(2.4/n1), (2.4/n1),(n2, n1+1)))
        self.w.append(np.random.default_rng().uniform(-(2.4/n1), (2.4/n1),(n3, n2+1)))
    
    def forward(self, variables_updated):
        
        # Função sigmoide
        gfunc = np.vectorize(lambda a : 1/(1+np.exp(-a)))
        
        # Primeira camada
        i1 = np.atleast_2d(self.w[0]@variables_updated)
        
        y1 = np.atleast_2d(gfunc(i1))
        
        y1 = np.insert(y1, 0, -1, axis = 0)
        
        # Segunda camada
        i2 = np.atleast_2d(self.w[1]@y1) 

        y2 = np.atleast_2d(gfunc(i2))
        
        return i1, y1, i2, y2
        
    def backward(self, variable, classe, i1, y1, i2, y2):
        
        variable = np.atleast_2d(variable)
        classe = np.atleast_2d(classe)
        
        # Derivada da função sigmoide
        glinhafunc = np.vectorize(lambda a : np.exp(-a)/((1+np.exp(-a))**2))
        
        # Segunda camada
        glinha2 = np.atleast_2d(glinhafunc(i2))

        grad2 = np.atleast_2d((classe.T - y2)*glinha2)
        
        previous_w1 = self.w[1]
        
        self.w[1] = np.atleast_2d(self.w[1] + self.taxa_aprendizado*grad2@y1.T + self.momentum*(self.w[1] - self.previous_w[1]))
        
        self.previous_w[1] = previous_w1
        
        # Primeira camada
        glinha1 = np.atleast_2d(glinhafunc(i1))
        
        grad1 = np.atleast_2d((grad2.T@self.w[1][:, 1:])*glinha1.T)
        
        previous_w0 = self.w[0]
        
        self.w[0] = np.atleast_2d(self.w[0] + self.taxa_aprendizado*grad1.T@variable.T + self.momentum*(self.w[0] - self.previous_w[0]))
        
        self.previous_w[0] = previous_w0
        
    
    def eqm(self):
        
        eqm = 0
            
        for i in range(len(self.variables) - self.n1):
            
            self.variables_updated = np.insert(self.variables[i : (i+self.n1)], 0, -1, axis = 0)
            
            i1, y1, i2, y2 = self.forward(self.variables_updated)
        
            eqm = eqm - (((self.output[i] - y2)**2)/2).sum()
            
        eqm = eqm/len(self.variables_updated)
        
        return eqm
        
    def train(self, variables, taxa_aprendizado, precision, momentum):
        
        self.variables = variables
        self.output = self.variables[self.n1:]
        self.taxa_aprendizado = taxa_aprendizado
        self.precision = precision
        self.momentum = momentum
        
        self.epoch = 1
        
        print('The inicial weight matrices are:')
        print(self.w)
        print('\n')
        
        self.previous_w = []
        self.previous_w.append(np.zeros(self.w[0].shape))
        self.previous_w.append(np.zeros(self.w[1].shape))
        
        while True:
            
            if (self.epoch==1)|((self.epoch%100)==0):
                print('\n')
                print('######################################')
                print('Starting a new epoch number %i' %(self.epoch))
                print('\n')
            
            previous_eqm = self.eqm()
            
            for i in range(len(self.variables) - self.n1):
                
                self.variables_updated = np.insert(self.variables[i : (i+self.n1)], 0, -1, axis = 0)
                
                i1, y1, i2, y2 = self.forward(self.variables_updated)
                
                self.backward(self.variables_updated, self.output[i], i1, y1, i2, y2)
            
            current_eqm = self.eqm()
            
            self.current_eqm = current_eqm
            
            if (self.epoch==1)|((self.epoch%100)==0):
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
        
        resultados = np.atleast_2d(np.zeros((len(variables), self.n3)))
        
        variables = np.insert(variables, 0, self.variables[-self.n1 :], axis = 0)
        
        for i in range(len(resultados)):
            
            variables_updated = np.insert(variables[i : (i+self.n1)], 0, -1, axis = 0)

            i1, y1, i2, y2 = self.forward(variables_updated)
            
            for j in range(len(y2)):
                
                resultados[i][j] = y2[j]
        
        return resultados

