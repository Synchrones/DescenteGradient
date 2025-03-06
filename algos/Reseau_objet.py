import random
from collections.abc import Callable

class Neurone:
    '''
    Definis une "neurone", objet qui représente la liaison entre deux noeuds du réseau, caractérisé
    par la liste des poids qui lui entrent dedans, un biais et une fonnction d'activation.
    Le poid d'indice 0 est le biais
    '''

    poids : list
    fonction_activation : Callable
    derive_fonction_activation : Callable

    def __init__(self, poids : list, fonction_activation : Callable, derive_fonction_activation : Callable):
        self.poids = poids
        self.fonction_activation = fonction_activation
        self.derive_fonction_activation = derive_fonction_activation


    def changer_poids(self, nouveau_poids):
        self.poids = nouveau_poids
    
    def __str__(self):
        return f"(poids:{self.poids})"


class Reseau():
    '''
    Objet du réseau de neurones, reçoit un nombre d'entrées, de couches et de sorties, représente un réseau
    avec la liste des entrées, une liste aléatoire de sorties et la liste des neurones.
    '''
    couches : list
    
    def __init__(self, entrees, couches : list):
        '''
        Reseau de neuronnes, une matrice qui représente chaque noeuds, qui contient la liste des poids des
        neurones qui en sortent.
        parametres : 
        couches : liste des couches du réseau contient le nombre de noeuds pour la couche de l'indice, la fonction
        d'activation et sa dérivée. Ex : [(10, fnc1, derive1), (6, fnc2, derive2), ...]
        '''
        self.couches =[]
        for i, couche in enumerate(couches):
            if i == 0:
                self.couches.append(
                    [Neurone([random.uniform(-0.5, 0.5) for _ in range(entrees + 1)], couche[1], couche[2]) for _ in
                    range(couche[0])])
            else:
                self.couches.append([Neurone([random.uniform(-0.5, 0.5) for _ in range(couches[i-1][0] + 1)], couche[1], couche[2]) for _ in range(couche[0])])


    def __str__(self):
        string ='['
        for couche in self.couches:
            string += '['
            for neurone in couche:
                string += str(neurone)
            string += ']'
        string += ']'
        return string

#scenario
if __name__ == '__main__':
    print(Reseau(2, [(1, lambda:True, lambda:False), (3, lambda:True, lambda:False)]))