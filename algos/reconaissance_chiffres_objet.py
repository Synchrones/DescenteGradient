import random
import time

from Reseau_objet import *
import pandas as pd
import numpy as np
from inspect import signature
import cProfile
from joblib import Parallel, delayed
import multiprocessing

# utilisé pour la parallélisation
nombre_coeurs = multiprocessing.cpu_count()

def passe_avant(reseau : Reseau, entrees : list) -> list:
    """
    calcul d'une image au travers d'un réseau de neurone.
    Parameters
    ----------
    reseau : le réseau auquel on applique la passe avant
    entrees : les entrées passées au réseau

    Returns
    -------
    Une liste contenant toutes les informations nécessaires à la passe arrière (cd en dessous)
    """

    # liste comprenant les informations nécessaires à la rétro propagation :
    # liste de tuples de listes, forme : (valeur avant fnc activation, valeur après), un tuple par couche
    # le premier tuple est celui des entrées
    sorties = [(None, entrees[:])]
    for i, couche in enumerate(reseau.couches):
        sorties.append(([], []))
        for neurone in couche:
            # somme pondérée par les poids
            # note : faire une boucle for est beaucoup plus rapide qu'utiliser sum() avec une liste par compréhension
            # 3 sec contre 4 sec pour 800 neurones environ (dont plus de deux secondes sur ouverture fichiers)
            # sortie = sum([([1] + sorties[i][1])[j] * neurone.poids[j] for j in range(len(neurone.poids))])
            sortie = 1 * neurone.poids[0]
            for poids in range(1, len(neurone.poids)):
                sortie += sorties[i][1][poids - 1] * neurone.poids[poids]
            sorties[i+1][0].append(sortie)

        # il faut appliquer la fonction d'activation après le calcul des sommes pondérés à cause des fncs comme
        # softmax qui utilisent les résultats des autres neurones
        for j in range(len(couche)):
            # application de la fonction d'activation
            if len(signature(couche[j].fonction_activation).parameters) == 1:
                sortie = couche[j].fonction_activation(sorties[i+1][0][j])
            else: # cas de softmax
                sortie = couche[j].fonction_activation(sorties[i + 1][0][j], sorties[i + 1][0])
            sorties[i + 1][1].append(sortie)
    return sorties

def sortie_reseau(reseau : Reseau, entrees : list):
    """
    Similaire à la fonction passe avant mais ne renvoie pas les calculs intermédiaires, seulement la sortie du réseau

    Parameters
    ----------
    reseau : le réseau dont on calcule la sortie
    entrees : les entrees passées au réseau

    Returns
    -------
    Les valeurs renvoyées par le réseau sous forme d'une liste
    """
    sorties = entrees[:]
    for i, couche in enumerate(reseau.couches):
        sorties_couche = []
        for neurone in couche:
            # somme pondérée par les poids
            # note : faire une boucle for est beaucoup plus rapide qu'utiliser sum() avec une liste par compréhension
            # 3 sec contre 4 sec pour 800 neurones environ (dont plus de deux secondes sur ouverture fichiers)
            # sortie = sum([([1] + sorties[i][1])[j] * neurone.poids[j] for j in range(len(neurone.poids))])
            sortie = 1 * neurone.poids[0]
            for poids in range(1, len(neurone.poids)):
                sortie += sorties[poids - 1] * neurone.poids[poids]

            sorties_couche.append(sortie)


        # il faut appliquer la fonction d'activation après le calcul des sommes pondérés à cause des fncs comme
        # softmax qui utilisent les résultats des autres neurones
        for j in range(len(couche)):
            # application de la fonction d'activation
            if len(signature(couche[j].fonction_activation).parameters) == 1:
                sortie = couche[j].fonction_activation(sorties_couche[j])
            else: # cas de softmax
                sortie = couche[j].fonction_activation(sorties_couche[j], sorties_couche)
            sorties_couche[j] = sortie
        sorties = sorties_couche[:]
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
    maj_poids = [[[0 for _ in neurone.poids] for neurone in couche] for couche in reseau.couches]

    # on itère sur le nombre d'exemples testés, i représente un exemple
    for i in range(len(sorties_attendues)):

        sorties_exemple = sorties[i]
        gradients_neurones = [[]]

        # calcul des gradients d'erreur liés à la dernière couche de neurones
        for neurone in range(len(sorties_attendues[i])):
            # simplifications grâce à l'utilisation combinée de softmax et cross entropy :
            # https://medium.com/towards-data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
            gradients_neurones[0].append(sorties_exemple[-1][1][neurone] - sorties_attendues[i][neurone])

        # calcul des gradients des neurones des couches cachées
        for couche in range(len(reseau.couches) - 2, -1, -1):
            gradients_neurones.insert(0, [])
            for neurone in range(len(reseau.couches[couche])):
                # règle de la chaine : responsabilité de l'erreur sur les neurones suivantes * dérivée de l'activation par
                # rapport au résultat de la somme pondérée
                derive_poids_suivants = 0
                for gradient in range(len(gradients_neurones[1])):
                    derive_poids_suivants += gradients_neurones[1][gradient] * reseau.couches[couche+1][gradient].poids[neurone+1]
                # derive_poids_suivants = sum([gradients_neurones[1][j] * reseau.couches[couche+1][j].poids[neurone+1]
                #                             for j in range(len(gradients_neurones[1]))])
                derive_activation = reseau.couches[couche][neurone].derive_fonction_activation(sorties_exemple[couche+1][0][neurone])
                gradients_neurones[0].append(derive_activation * derive_poids_suivants)

                # calcul des maj des poids
                # cas du biais :
                maj_poids[couche][neurone][0] += gradients_neurones[0][neurone]

                for poids in range(1, len(reseau.couches[couche][neurone].poids)):
                    maj_poids[couche][neurone][poids] += sorties_exemple[couche][1][poids-1] * gradients_neurones[0][neurone]


    for couche in range(len(maj_poids)):
        for neurone in range(len(maj_poids[couche])):
            nouveaux_poids = []
            for poids in range(len(maj_poids[couche][neurone])):
                # moyenne de la modification du poids sur les exemples traités
                nouveaux_poids.append(reseau.couches[couche][neurone].poids[poids] - fac_apprentissage + maj_poids[couche][neurone][poids] / len(maj_poids))
            reseau.couches[couche][neurone].changer_poids(nouveaux_poids)


def one_hot(y) -> list[list]:
    """
    Traite la liste des sorties, les transforme en sortie de réseau, par exemple si le résultat attendu est 1
    cette fonction doit retourner [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
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

def test_import_export():

    test = Reseau(1, [(10, Relu, derive_Relu), (5, Relu, derive_Relu)])
    test.exporter_poids("test")
    test2 = Reseau(1, [(10, Relu, derive_Relu), (5, Relu, derive_Relu)])
    test2.importer_poids("test")
    print(test)
    print(test2)

#scenario
def main():
    #telechargement de la base de donnée
    TRAIN = pd.read_csv('train.csv')#, skiprows = 1)
    TEST = pd.read_csv("test.csv")#, skiprows = 1)
    X_TRAIN = TRAIN.copy()
    X_TEST = TEST.copy()
    label = TRAIN.label.tolist()
    del X_TRAIN['label']

    reseau = Reseau(784, [(784, Relu, derive_Relu), (10, softmax, None)])
    """
    entree = [element/255 for element in X_TRAIN.loc[random.randint(0, 40000)].tolist()]
    sortie = passe_avant(reseau, entree)
    passe_arriere(reseau, [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [sortie], 0.1)
    """

    nb_iteration = 1
    nb_exemples = 100
    fac_apprentissage = 0.1
    for i in range(nb_iteration):

        print("passe avant")
        # récupération des images dans la base de données
        temps = time.time()
        indices = [random.randint(0, 40000) for _ in range(nb_exemples)]
        entrees = [[element/255 for element in X_TRAIN.loc[indice].tolist()] for indice in indices]
        sorties_attendues_chiffre = [label[indice] for indice in indices]
        sorties_attendues = [one_hot(label[indice]) for indice in indices]
        # passe avant exécutée en parallèle, deviens plus efficace pour un grand nombre d'exemples (2x plus rapide pour 100)
        sorties_reseau = Parallel(n_jobs=nombre_coeurs)(delayed(lambda reseau, entree : passe_avant(reseau, entree))(reseau, entree) for entree in entrees)
        print(time.time() - temps)

        print("passe arrière")
        passe_arriere(reseau, sorties_attendues, sorties_reseau, fac_apprentissage)
        erreur = sum([-np.log(sorties_reseau[j][-1][1][sorties_attendues_chiffre[j]]) for j in range(nb_exemples)]) / nb_exemples
        print(erreur)
    """
    # tests après entrainement
    accuracy = 0
    for i in range(100):
        indice = random.randint(0, 40000)
        entree = [element / 255 for element in X_TRAIN.loc[indice].tolist()]
        accuracy += passe_avant(reseau, entree)[-1][1][label[indice]]
        print(label[indice], passe_avant(reseau, entree)[-1][1])
    print(f"accuracy {accuracy/100}")
    enregistrement = input("Entrez le nom du fichier si vous souhaitez enregistrer les poids de ce réseau")
    if enregistrement != "":
        reseau.exporter_poids(enregistrement)
    """

cProfile.run("main()")

def test_reseau():
    # dérivée softmax non nécessaire dû aux simplifications
    test_reseau = Reseau(1, [(1, Relu, derive_Relu)])
    print(test_reseau)
    print(passe_avant(test_reseau, [2]))
    nb_iteration = 100
    nb_batch = 10
    for i in range(nb_iteration):
        entrees = [random.uniform(-2, 2) for _ in range(nb_batch)]
        sorties_attendues = [[entree**2] for entree in entrees]
        sorties_reseau = [passe_avant(test_reseau, [entree]) for entree in entrees]
        passe_arriere(test_reseau,
                      sorties_attendues,
                      sorties_reseau,
                      0.1)
        erreur_moy = sum([(sorties_attendues[j][0] - sorties_reseau[j][-1][1][0])**2 for j in range(nb_batch)]) / nb_batch
        print(erreur_moy)
    print(test_reseau)

