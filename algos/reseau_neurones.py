# numpy
import numpy as np


# creation d'un réseau avec des poids nuls
def reseau(couches : list[int]) -> list[np.array]:

    """
    Fonction qui créé un réseau de neurones, avec des poids nuls.
    IN : couches (list[int]) - représente le réseau voulu par exemple le réseau de 2 entrées vers 3 neurones 
    vers 2 sorties est représenté par [2, 3, 2]
    OUT : reseau (liste[np.array]) - le réseau en question, contient les poids des neurones dans les bonnes
    dimensions
    """

    # initialisatiton du reseau
    reseau = []
    for indice in range(len(couches) - 1): # le nombre de matrices dans le réseau
        ligne = [0]*couches[indice + 1] # le nombre de colones dans la matrice
        matrix = []
        for i in range(couches[indice]): # le nombre de lignes dans la matrice
            matrix.append(ligne)
        reseau.append(np.array(matrix))
    return reseau


def modif_poid(reseau : list, couche : int, ligne : int, colone : int, nouveau_poid : float) -> None:

    """
    Permet de modifier un poid d'un réseau, le poid se situe sur la couche, la ligne et la colone.
    """

    reseau[couche][ligne, colone] = nouveau_poid


def calcul(entrees : np.array, reseau : list, func_activation) -> list:
    
    """
    Fait passer les entrées dans un réseau retourne les sorties
    """

    for matrice in reseau:
       entrees = func_activation(np.dot(entrees, matrice))
    
    return entrees


# scenario
if __name__ == '__main__':
    m = reseau([2, 3, 2])
    print(m)
    modif_poid(m, 0, 0, 0, 69)
    print(m)
    modif_poid(m, 1, 1, 1, 69)
    print(m)