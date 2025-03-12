import random

from Reseau_objet import *
import pandas as pd
import numpy as np
from inspect import signature

def passe_avant(reseau : Reseau, entrees : list) -> list:
    """
    calcul d'une image au travers d'un réseau de neurone.
    """
    # liste comprenant les informations nécéssaires à la rétropropagation :
    # liste de tuples de listes, forme : (valeur avant fnc activation, valeur après), un tuple par couche
    # le premier tuple est celui des entrées
    sorties = [(None, entrees[:])]
    for i, couche in enumerate(reseau.couches):
        sorties.append(([], []))
        for neurone in couche:
            # somme pondérée par les poids
            sortie = sum([([1] + sorties[i][1])[j] * neurone.poids[j] for j in range(len(neurone.poids))])
            sorties[i+1][0].append(sortie)

        # il faut appliquer la fonction d'activation après le calcul des sommes pondérés à cause des fncs comme
        # softmax qui utilisent les résultats des autres neurones
        for j in range(len(couche)):
            # application de la fonction d'activation
            if len(signature(couche[j].fonction_activation).parameters) == 1:
                sortie = couche[j].fonction_activation(sorties[i+1][0][j])
            else:
                sortie = couche[j].fonction_activation(sorties[i + 1][0][j], sorties[i + 1][0])
            sorties[i + 1][1].append(sortie)
    return sorties

def Relu(x):
    """
    Une fonction d'activation, assez simple, renvoie x lorsque x est positif et 0 sinon.
    """
    return max(0, x)

def derive_Relu(x):
    return x > 0

def softmax(x, autres_sorties):
    return float(np.exp(x) / sum(np.exp(autres_sorties)))


def passe_arriere(reseau : Reseau, sorties_attendues : list, sorties : list, fac_apprentissage : float):
    """
    Application de l'algorithme de rétro propagation du gradient
    sorties_attendues : forme [(1, 2, 3, ...), (3, 2, 1, ...), ...]
    sorties : liste de sorties obtenues par passe_avant()
    """

    maj_poids = []
    # on itère sur le nombre d'exemples testés, i représente un exemple
    for i in range(len(sorties_attendues)):
            sorties_exemple = sorties[i]
            gradients_neurones = []

            # calcul des gradients d'erreur liés à la dernière couche de neurones
            gradients_neurones.append([])
            for neurone in range(len(sorties_attendues[i])):
                # simplifications grâce à l'utilisation combinée de softmax et cross entropy :
                # https://medium.com/towards-data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
                gradients_neurones[0].append(sorties_exemple[-1][1][neurone] - sorties_attendues[i][neurone])

            # calcul des gradients des neurones des couches cachées
            for couche in range(len(reseau.couches) - 1, 0, -1):
                gradients_neurones.insert(0, [])
                for neurone in range(len(reseau.couches[couche])):
                    # règle de la chaine : responsabilité de l'erreur sur neurones suivantes * dérivée de l'activation par
                    # rapport au résultat de la somme pondérée
                    derive_poids_suivants = sum([gradients_neurones[1][j] * reseau.couches[couche][neurone].poids[j-1]
                                                 for j in range(1, len(gradients_neurones[1]))])
                    derive_activation = reseau.couches[couche][neurone].derive_fonction_activation(sorties_exemple[couche][neurone][1])
                    gradients_neurones[0].append(derive_activation * derive_poids_suivants)


            maj_poids.append([])
            for couche in range(len(reseau.couches)):
                maj_poids[i].append([])
                for neurone in range(len(reseau.couches[couche])):
                    maj_poids[i][couche].append([])
                    # cas du biais :
                    maj_poids[i][couche][neurone].append(gradients_neurones[couche][neurone])
                    for poids in range(len(reseau.couches[couche][neurone].poids) - 1):
                        maj_poids[i][couche][neurone].append(sorties_exemple[couche][1][poids] * gradients_neurones[couche][neurone])
    for couche in range(len(maj_poids[0])):
        for neurone in range(len(maj_poids[0][couche])):
            nouveaux_poids = []
            for poids in range(len(maj_poids[0][couche][neurone])):
                # moyenne de la modification du poids sur les exemples traités
                nouveaux_poids.append(reseau.couches[couche][neurone].poids[poids]
                                      - fac_apprentissage * sum([maj_poids[i][couche][neurone][poids] for i in range(len(maj_poids))]) / len(maj_poids))
            reseau.couches[couche][neurone].changer_poids(nouveaux_poids)



def one_hot(y) -> list[list]:
    """
    Traite la liste des sorties, les transforme en sortie de réseau, par exemple le résultat attendu est 1
    donc cette fonction doit retourner [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
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

"""
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
"""

# dérivée softmax non nécessaire dû aux simplifications
test_reseau = Reseau(1, [(1, lambda x: x, lambda x:1)])
print(test_reseau)
print(passe_avant(test_reseau, [1]))

for i in range(100):
    entrees = [random.uniform(-10, 10) for _ in range(10)]
    sorties_attendues = [[3 + 10 * entrees[j]] for j in range(10)]
    sorties_reseau = [passe_avant(test_reseau, [entree]) for entree in entrees]
    passe_arriere(test_reseau,
                  sorties_attendues,
                  sorties_reseau,
                  0.05)
    erreur_moy = sum([(sorties_attendues[j][0] - sorties_reseau[j][-1][1][0])**2 for j in range(10)]) / 10
    print(erreur_moy)
print(test_reseau)

