import random
from collections.abc import Callable
import json

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
    nb_entrees : int
    def __init__(self, nb_entrees : int, couches : list):
        '''
        Reseau de neuronnes, une matrice qui représente chaque nœud, qui contient la liste des poids des
        neurones qui en sortent.
        Parametres :
        nb_entrees : le nombre d'entrées du réseau
        couches : liste des couches du réseau contient le nombre de noeuds pour la couche de l'indice, la fonction
        d'activation et sa dérivée. Ex : [(10, fnc1, derive1), (6, fnc2, derive2), ...]
        '''
        self.couches = []
        self.nb_entrees = nb_entrees
        for i, couche in enumerate(couches):
            if i == 0:
                self.couches.append(
                    [Neurone([0] + [random.uniform(-0.5, 0.5) for _ in range(nb_entrees)], couche[1], couche[2]) for _ in
                     range(couche[0])])
            else:
                self.couches.append([Neurone([0] + [random.uniform(-0.5, 0.5) for _ in range(couches[i-1][0])], couche[1], couche[2]) for _ in range(couche[0])])

    def exporter_poids(self, nom : str):
        """
        Permet d'exporter les poids du réseau dans un fichier json
        Parameters
        ----------
        nom : le nom du fichier enregistré (sans extension)
        """
        dict_reseau = {}
        dict_reseau["nb_entrees"] = self.nb_entrees
        dict_reseau["couches"] = [len(couche) for couche in self.couches]

        dict_reseau["poids"] = [[[poids for poids in neurone.poids] for neurone in couche] for couche in self.couches]
        with open(f"ReseauxExportes/{nom}.json", "w", encoding='utf-8') as f:
            json.dump(dict_reseau, f, ensure_ascii=False, indent=4)

    def importer_poids(self, nom : str):
        """
        Permet d'importer les poids depuis un fichier json et de les appliquer au réseau
        Parameters
        ----------
        nom : le nom du fichier (sans extension)
        """
        with (open(f"ReseauxExportes/{nom}.json", "r") as f):
            dict_reseau = json.load(f)
            if dict_reseau["nb_entrees"] == self.nb_entrees and all([dict_reseau["couches"][couche] == len(self.couches[couche])
                                                                     for couche in range(len(self.couches))]):
                for couche in range(len(dict_reseau["couches"])):
                    for neurone in range(dict_reseau["couches"][couche]):
                        self.couches[couche][neurone].changer_poids(dict_reseau["poids"][couche][neurone])
            else:
                print("Taille du réseau incompatible")


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