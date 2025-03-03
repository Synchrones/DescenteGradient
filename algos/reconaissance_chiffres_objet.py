from Reseau_objet import *
import pandas as pd
import numpy as np

def passe_avant(reseau : Reseau, entrees : list):
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

def passe_arriere(reseau, sorties_attendues):
    pass

#scenario
if __name__ == "__main__":
    #telechargement de la base de donnée
    TRAIN = pd.read_csv('train.csv')#, skiprows = 1)
    TEST = pd.read_csv("test.csv")#, skiprows = 1)
    X_TRAIN = TRAIN.copy()
    X_TEST = TEST.copy()
    y = TRAIN.label.tolist()
    del X_TRAIN['label']
    #test sur une donnée aléatoire
    R = Reseau([484, 10, 10])
    indice = random.randint(0, 40000)
    entrees = X_TRAIN.loc[indice].tolist()
    preresultat1, resultat1, preresultat2, resultat2 = passe_avant(R, entrees)
    print(f"resultat1 = {resultat1}, resultat2 = {resultat2}")
    print(y[indice])
