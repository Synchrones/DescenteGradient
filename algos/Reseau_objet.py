import random
from collections.abc import Callable

class Neurone:
    '''
    Definis une "neurone", objet qui représente la liaison entre deux noeuds du réseau, caractérisé
    par la liste des poids qui lui entrent dedans, un biais et une fonnction d'activation.
    '''

    poids : list
    biais : float

    def __init__(self, poids : list, biais : float):
        self.poids = poids
        self.biais = biais

    def changer_poids(self, nouveau_poids):
        self.poids = nouveau_poids
    
    def __str__(self):
        return f"(poids:{self.poids}, biais:{self.biais})"


class Reseau():
    '''
    Objet du réseau de neurones, reçoit un nombre d'entrées, de couches et de sorties, représente un réseau
    avec la liste des entrées, une liste aléatoire de sorties et la liste des neurones.
    '''
    couches : list
    
    def __init__(self, couches : list):
        '''
        Reseau de neuronnes, une matrice qui représente chaque noeuds, qui contient la liste des poids des
        neurones qui en sortent.
        parametres : 
        couches : liste des couches du réseau contient le nombre de noeuds pour la couche de l'indice
        '''
        self.entrees = [0]*couches[0]
        self.couche1 = [Neurone([random.uniform(-0.5, 0.5) for i in range(couches[0])], random.uniform(-0.5, 0.5)) for j in range(couches[1])]
        self.couche2 = [Neurone([random.uniform(-0.5, 0.5) for i in range(couches[1])], random.uniform(-0.5, 0.5)) for j in range(couches[2])]
    

    def __str__(self):
        string ='[[[entrées:'
        for entree in self.entrees:
            string += str(entree)
        string += ']'
        for Neurone in self.couche1:
            string += str(Neurone)
        string += ']'
        for Neurone in self.couche2:
            string += str(Neurone)
        string += ']]'
        return string

#scenario
if __name__ == '__main__':
    print(Reseau([484, 10, 10]))