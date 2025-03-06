from Reseau_objet import *
import pandas as pd
import numpy as np

def passe_avant(reseau : Reseau, entrees : list) -> list:
    """
    calcul d'une image au travers d'un réseau de neurone.
    """
    preresultat1 = [] # les sorties de la première couche avant d'être passées dans la fonction d'activation 
    resultat1 = [] # les sorties de la première couche passées dans la fonction d'activation
    for neurone in reseau.couche1:
        sortie = 0
        for i, poid in enumerate(neurone.poids):
            sortie += poid * entrees[i]
        sortie = sortie + neurone.biais
        preresultat1.append(sortie)
    for preresultat in preresultat1: # la fonction d'activation
        resultat1.append(Relu(preresultat))
    
    preresultat2 = [] # les sorties de la deuxième couche avant le softmax
    resultat2 = [] # les sorties de la deuxième couche après le softmax
    for neurone in reseau.couche2:
        sortie = 0
        for i, poid in enumerate(neurone.poids):
            sortie += poid * resultat1[i]
        sortie = sortie + neurone.biais
        preresultat2.append(sortie)
    print(preresultat2)
    for preresultat in preresultat2: # le softmax
        resultat2.append(float(np.exp(preresultat)/sum(np.exp(preresultat2))))
        #print(preresultat, sum(preresultat2))

    return preresultat1, resultat1, preresultat2, resultat2

def Relu(x):
    """
    Une fonction d'activation, assez simple, renvoie x lorsque x est positif et 0 sinon.
    """
    return max(0, x)

def Deriv_Relu(x):
    return x > 0

def passe_arriere(reseau : Reseau, sorties_attendues : list, preresultat1, resultat1, preresultat2, resultat2):
    """
    Calcul des incréments sur les poids et les biais du réseau.
    """
    dZ2 = [] # grandeur intermédiaire au calcul des incréments sur les poids et les biais de la deuxième couche
    resultat_attendus = one_hot(sorties_attendues)
    for i in range(10):
        dZ2.append(resultat2[i] - resultat_attendus[i])
    dW2 = [] #les incréments sur les poids de la deuxième couche
    for i in range(10):
        dW2.append([])
        for j in range(10):
            dW2[i].append(dZ2[i] * resultat1[j])
    dB2 = [sum(dZ2)]*10 #les incréments sur les biais de la deuxième couche
    dZ1 = []

    dW1 = []
    dB1 = []
    
    

def one_hot(y) -> list[list]:
    """
    Traite la liste des sorties, les transforme en sortie de réseau, par exemple le résultat attendu est 1
    donc cette fonction doit retourner [0, 1, 0, 0, 0, 0 ,0, 0, 0, 0]
    """
    resultat = []
    if isinstance(y, list):
        n = 0
        for element in y :
            resultat.append([0]*10)
            resultat[n][element] = 1
            n += 1
    else :
        resultat = [0]*10
        resultat[y] = 1
    return resultat


#scenario
if __name__ == "__main__":
    #telechargement de la base de donnée
    TRAIN = pd.read_csv('train.csv')#, skiprows = 1)
    TEST = pd.read_csv("test.csv")#, skiprows = 1)
    X_TRAIN = TRAIN.copy()
    X_TEST = TEST.copy()
    label = TRAIN.label.tolist()
    del X_TRAIN['label']
    #test sur une donnée aléatoire
    R = Reseau([484, 10, 10])
    indice = random.randint(0, 40000)
    entrees = X_TRAIN.loc[indice].tolist()
    entrees__normalisees = []
    for element in entrees:
        entrees__normalisees.append(element/255)
    preresultat1, resultat1, preresultat2, resultat2 = passe_avant(R, entrees__normalisees)
    print(f"resultat1 = {resultat1}, resultat2 = {resultat2}")
    print(resultat2.index(max(resultat2)))
    print(label[indice])
    print(one_hot(label[indice]))