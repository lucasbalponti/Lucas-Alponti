import pandas as pd
import numpy as np

class neuron:
    
    def __init__(self, ident, xcord, ycord):
        self.ident = ident
        self.xcord = xcord
        self.ycord = ycord
    
    def add_neighbours(self, neighbours):
        self.neighbours = neighbours
    
    def add_initial_weight(self, initial_weight):
        self.weight = initial_weight
    
    def add_class(self, neuron_class):
        self.neuron_class = neuron_class

class konohen:
    
    def __init__(self, nx, ny, radius):
        self.nx = nx
        self.ny = ny
        self.radius = radius
        self.k_map = []
        
        k = 0
        
        for i in range(self.nx):
            for j in range(self.ny):
                self.k_map.append(neuron(k, i, j))
                k = k+1
        
        for i in range(len(self.k_map)):
            neighbours = []
            for j in range(len(self.k_map)):
                if (i != j):
                    dist = np.linalg.norm(np.array([self.k_map[i].xcord, self.k_map[i].ycord]) - np.array([self.k_map[j].xcord, self.k_map[j].ycord]))
                    if (dist <= self.radius):
                        neighbours.append(self.k_map[j].ident)
            self.k_map[i].add_neighbours(neighbours)
    
    def map_2d(self):
        for i in range(self.ny):
            line = []
            for j in range(self.nx):
                line.append(self.k_map[i*(self.nx) + j].ident)
            print(line)
    
    def neighbours(self):
        for i in range(len(self.k_map)):
            print('Neuron ' + str(i) + ': ' + ', '.join(map(str, self.k_map[i].neighbours)))
    
    def winner_correction(self, i, winner):
        
        self.k_map[winner].weight = self.k_map[winner].weight + self.learn_rate*(self.train_samples[i] - self.k_map[winner].weight)
        
        self.k_map[winner].weight = self.k_map[winner].weight/np.linalg.norm(self.k_map[winner].weight)
        
        self.current_weights[winner] = self.k_map[winner].weight
    
    def neighbours_corretion(self, i, winner):
        
        for j in range(len(self.k_map[winner].neighbours)):
            
            self.k_map[self.k_map[winner].neighbours[j]].weight = self.k_map[self.k_map[winner].neighbours[j]].weight + (self.learn_rate/2)*(self.train_samples[i] - self.k_map[self.k_map[winner].neighbours[j]].weight)
            
            self.k_map[self.k_map[winner].neighbours[j]].weight = self.k_map[self.k_map[winner].neighbours[j]].weight/np.linalg.norm(self.k_map[self.k_map[winner].neighbours[j]].weight)
            
            self.current_weights[self.k_map[winner].neighbours[j]] = self.k_map[self.k_map[winner].neighbours[j]].weight
    
    def train(self, train_samples, classes, learn_rate):
        
        self.train_samples = np.copy(train_samples)
        self.learn_rate = learn_rate
        self.train_classes = classes
        
        for i in range(len(self.train_samples)):
            norm = 0
            for j in range(len(self.train_samples[i])):
                norm = norm + self.train_samples[i][j]**2
            norm = norm**0.5
            self.train_samples[i] = self.train_samples[i]/norm
        
        self.current_weights = []
        
        for i in range(len(self.k_map)):
            self.k_map[i].add_initial_weight(self.train_samples[i])
            self.current_weights.append(self.k_map[i].weight)
        
        self.epoch = 0
        
        while True:
            
            if (((self.epoch+1)%100 == 0)|(self.epoch == 0)):
                print('Starting epoch number ' + str(self.epoch+1))
                print('#######################')
            
            self.previous_weights = self.current_weights.copy()
            
            winner_list = []
                       
            for i in range(len(self.train_samples)):
                
                dist = []
                
                for j in range(len(self.k_map)):
                
                    dist.append(np.linalg.norm(self.train_samples[i] - self.k_map[j].weight))
                
                winner = dist.index(min(dist))
                
                self.winner_correction(i, winner)
                
                self.neighbours_corretion(i, winner)
                
                winner_list.append(winner)
                
            self.epoch = self.epoch + 1
            
            if all([np.allclose(x, y) for x, y in zip(self.previous_weights, self.current_weights)]):
                self.test_winners = winner_list.copy()
                print('Training finished in '+str(self.epoch)+' epochs')
                break
                
        for i in range(len(self.k_map)):
            
            class_dict = {}
            
            for j in range(len(np.unique(self.train_classes))):
                
                count = 0
                
                for k in range(len(self.test_winners)):
                    
                    if (self.k_map[i].ident == self.test_winners[k])&(np.unique(self.train_classes)[j] == self.train_classes[k]):
                        
                        count = count + 1
                
                class_dict[np.unique(self.train_classes)[j]] = count
            
            self.k_map[i].add_class([k for k, v in class_dict.items() if v == max(class_dict.values())])
    
    def class_map_2d(self):
        for i in range(self.ny):
            line = []
            for j in range(self.nx):
                line.append('&'.join(self.k_map[i*(self.nx) + j].neuron_class))
            print(line)
