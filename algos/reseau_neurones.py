# reseau
import numpy as np
from random import randint

# creation d'un réseau avec des poids nuls
def reseau(couches : list[int]) -> list[np.array]:

    """
    Fonction qui créé un réseau de neurones, avec des poids aléatoires.
    IN : couches (list[int]) - représente le réseau voulu par exemple le réseau de 2 entrées vers 3 neurones 
    vers 2 sorties est représenté par [2, 3, 2]
    OUT : reseau (liste[np.array]) - le réseau en question, contient les poids des neurones dans les bonnes
    dimensions
    """

    # initialisatiton du reseau
    reseau = []
    for indice in range(len(couches) - 1): # le nombre de matrices dans le réseau
        ligne = [randint(1, 50)]*couches[indice + 1] # le nombre de colones dans la matrice
        matrix = []
        for i in range(couches[indice]): # le nombre de lignes dans la matrice
            matrix.append(ligne)
        reseau.append(np.array(matrix))
    return reseau


def biais(reseau):
    biais = []
    for i in range(len(reseau) - 1):
        biais.append(np.array([randint(-50, 50) for j in range(len(reseau[i + 1]))]))
    return  biais


def modif_poid(reseau : list[np.array], couche : int, ligne : int, colone : int, nouveau_poid : float) -> None:

    """
    Permet de modifier un poid d'un réseau, le poid se situe sur la couche, la ligne et la colone, utile lors de
    l'aprentissae du réseau et de l'optimisation des poids.
    IN : reseau (list[np.array]) - le reseau dont on veut modifier un des poids
    IN : couche (int) - la couche sur laquelle on ssouhaite modifier un des poids
    IN : ligne (int) - la ligne du poid à modifier
    IN : colone (int) - la colone du poid à modifier
    IN : nouveau_poid (float) - la nouvelle valeur du poid
    """

    reseau[couche][ligne, colone] = nouveau_poid


def calcul(entrees : np.array, reseau : list[np.array], func_activation, biais : list[np.array]) -> list:
    
    """
    Fait passer les entrées dans un réseau retourne les sorties.
    IN OUT : entrees (np.array) - en input les entrées du réseau : les données recueillies, en output les
    sorties du reseau, multiplication par les poids et composition par la fonction d'activation.
    IN : reseau (list[np.array]) - le réseau sur lequel on applique le calcul
    IN : func_activation (function) - la fonction d'activation.
    """

    for i, matrice in enumerate(reseau):
       entrees = func_activation(np.dot(np.add(entrees, matrice)) + biais[i])
    
    return entrees


def fonction_activation(x):
    return 1 / (1 + np.exp(x))


# scenario
if __name__ == '__main__':
    m = reseau([2, 3, 2])
    print(m)
    modif_poid(m, 0, 0, 0, 69)
    print(m)
    modif_poid(m, 1, 1, 1, 69)
    print(m)
    b = biais(m)
    print(b)