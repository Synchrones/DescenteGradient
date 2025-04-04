import random
import time

from Reseau_objet import *
import pandas as pd
import numpy as np
from inspect import signature
import cProfile
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt


# utilisé pour la parallélisation
nombre_coeurs = multiprocessing.cpu_count()

def passe_avant(reseau : Reseau, entrees : list) -> list:
    """
    Calcul d'une image au travers d'un réseau de neurone.
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
            # gain d'un peu moins d'une seconde pour 800 neurones environ
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
            # gain d'un peu moins d'une seconde pour 800 neurones environ
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


def passe_arriere(reseau : Reseau, sorties_attendues : list, sorties_reseau : list, fac_apprentissage : float):
    """
    Application de l'algorithme de rétro propagation du gradient
    sorties_attendues : forme [(1, 2, 3, ...), (3, 2, 1, ...), ...]
    sorties : liste de sorties obtenues par passe_avant()
    """

    # note : attention à ne pas passer par des floats numpy dans les calculs car les approximations ont l'air de faire apparaitre
    # des 0 qui cassent l'algorithme (à vérifier)

    # segmentations des sorties sur lesquels on réalise la retro propagation pour paralléliser l'algorithme
    # gain de 6 secondes sur 100 sorties avec un processeur 12 cœurs
    sorties_attendues_partielles = list(decouper_liste(sorties_attendues, nombre_coeurs))
    sorties_reseau_partielles = list(decouper_liste(sorties_reseau, nombre_coeurs))
    maj_poids = Parallel(n_jobs=nombre_coeurs)(delayed(calculer_maj_poids)(reseau, sorties_attendues_partielles[i], sorties_reseau_partielles[i]) for i in range(nombre_coeurs))
    for couche in range(len(maj_poids[0])):
        # somme_majs_poids_couche = np.sum([maj_poids[i][couche] for i in range(len(maj_poids))], axis=0)
        for neurone in range(len(maj_poids[0][couche])):
            nouveaux_poids = []
            for poids in range(len(maj_poids[0][couche][neurone])):
                somme_majs_poids = 0
                for sortie in range(len(maj_poids)):
                    somme_majs_poids += maj_poids[sortie][couche][neurone][poids]
                # moyenne de la modification du poids sur les exemples traités
                nouveaux_poids.append(reseau.couches[couche][neurone].poids[poids] - fac_apprentissage * somme_majs_poids / len(sorties_attendues))
            reseau.couches[couche][neurone].changer_poids(nouveaux_poids)


def decouper_liste(liste, nb_elements):
    """
    Découpe une liste en nb_element - 1 sous-listes de tailles égales + une dernière avec le reste

    Parameters
    ----------
    liste : la liste à découper
    nb_elements : le nombre de sous listes voulues

    Returns
    -------
    Un tuple contenant les sous listes obtenues par découpage de la liste d'entrée
    """
    k, m = divmod(len(liste), nb_elements)
    return (liste[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nb_elements))


def calculer_maj_poids(reseau, sorties_attendues, sorties_reseau):
    """
    Calcul les sommes des modifications à réaliser sur chaque poids pour une liste de sorties données

    Parameters
    ----------
    reseau : le réseau dont on calcule la sortie
    sorties_attendues : les sorties attendues du réseau
    sorties_reseau : les sorties prévues par le réseau (retournées par la fonction passe_avant)

    Returns
    -------
    Les sommes des modifications des poids
    """

    maj_poids = [[[0 for _ in neurone.poids] for neurone in couche] for couche in reseau.couches]

    # On itère sur les sorties données
    for i in range(len(sorties_attendues)):

        sorties_exemple = sorties_reseau[i]
        gradients_neurones = [[]]

        # calcul des gradients d'erreur liés à la dernière couche de neurones
        for neurone in range(len(sorties_attendues[i])):
            # simplifications grâce à l'utilisation combinée de softmax et cross entropy :
            # https://medium.com/towards-data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
            gradients_neurones[0].append(sorties_exemple[-1][1][neurone] - sorties_attendues[i][neurone])

            # calcul des maj des poids, on réalise des sommes pour ensuite calculer la moyenne des modifications
            # cas du biais :
            maj_poids[-1][neurone][0] += gradients_neurones[0][neurone]
            # autres poids
            for poids in range(1, len(reseau.couches[-1][neurone].poids)):
                maj_poids[-1][neurone][poids] += sorties_exemple[-2][1][poids - 1] * gradients_neurones[0][neurone]

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

                # calcul des maj des poids, on réalise des sommes pour ensuite calculer la moyenne des modifications
                # cas du biais :
                maj_poids[couche][neurone][0] += gradients_neurones[0][neurone]
                # autres poids
                for poids in range(1, len(reseau.couches[couche][neurone].poids)):
                    maj_poids[couche][neurone][poids] += sorties_exemple[couche][1][poids-1] * gradients_neurones[0][neurone]

    return maj_poids


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


def entrainement_chiffre():
    # chargement de la base de donnée
    TRAIN = pd.read_csv('train.csv')#, skiprows = 1)
    X_TRAIN = TRAIN.copy()[:30000]
    X_TEST = TRAIN.copy()[30000:35000]
    X_VALID = TRAIN.copy()[35000:]
    label_train = X_TRAIN.label.tolist()
    del X_TRAIN['label']
    label_test = X_TEST.label.tolist()
    del X_TEST['label']
    label_valid = X_VALID.label.tolist()
    del X_VALID['label']

    reseau = Reseau(784, [(784, Relu, derive_Relu), (10, softmax, None)])

    nb_iteration = 50
    nb_exemples = 30
    fac_apprentissage = 0.1
    temps_moyen_passe_avant = 0
    temps_moyen_passe_arriere = 0
    for i in range(nb_iteration):

        print("passe avant")
        # récupération des images dans la base de données
        temps = time.time()
        indices = [i * nb_exemples + j for j in range(nb_exemples)]
        entrees = [[element/255 for element in X_TRAIN.loc[indice].tolist()] for indice in indices]
        sorties_attendues_chiffre = [label_train[indice] for indice in indices]
        sorties_attendues = [one_hot(label_train[indice]) for indice in indices]
        # passe avant exécutée en parallèle, deviens plus efficace pour un grand nombre d'exemples (2x plus rapide pour 100)
        sorties_reseau = Parallel(n_jobs=nombre_coeurs)(delayed(lambda reseau, entree : passe_avant(reseau, entree))(reseau, entree) for entree in entrees)
        print(f"temps d'exécution de la passe avant {time.time() - temps}")
        temps_moyen_passe_avant += time.time() - temps
        print("passe arrière")
        temps = time.time()
        passe_arriere(reseau, sorties_attendues, sorties_reseau, fac_apprentissage)
        print(f"temps d'exécution de la passe arrière {time.time() - temps}")
        temps_moyen_passe_arriere += time.time() - temps

        erreur = sum([-np.log(sorties_reseau[j][-1][1][sorties_attendues_chiffre[j]]) for j in range(nb_exemples)]) / nb_exemples
        print(f"erreur du réseau : {erreur}")

    # tests après entrainement
    accuracy = 0
    for i in range(100):
        indice = random.randint(0, 5000)
        entree = [element / 255 for element in X_VALID.loc[indice+30000].tolist()]
        accuracy += sortie_reseau(reseau, entree)[label_valid[indice]]
        print(label_valid[indice], passe_avant(reseau, entree)[-1][1])
    print(f"précision : {accuracy/100}")
    print(f"temps moyen d'exécution de la passe avant : {temps_moyen_passe_avant/nb_iteration}")
    print(f"temps moyen d'exécution de la passe arrière : {temps_moyen_passe_arriere / nb_iteration}")
    enregistrement = input("Entrez le nom du fichier si vous souhaitez enregistrer les poids de ce réseau")
    if enregistrement != "":
        reseau.exporter_poids(enregistrement)

def parse_pixel_column(pixel_str):
    """Convertit une chaîne '[  0   0 255 255]' en un tableau numpy [0, 0, 255]"""
    try:
        # Supprimer les crochets et diviser en une liste de 4 entiers
        pixel_values = list(map(int, pixel_str.strip("[]").split()))

        # Vérifier que nous avons bien 4 valeurs (R, G, B, A)
        if len(pixel_values) != 4:
            raise ValueError(f"Pixel mal formé: {pixel_str}")

        return np.array(pixel_values[:-1], dtype=np.uint8)
    except Exception as e:
        print(f"Erreur de conversion du pixel: {pixel_str} - {e}")
    return np.array([0, 0, 0], dtype=np.uint8)  # Valeur par défaut (noir opaque)


def entrainement_bouteilles():
    temps_total = time.time()
    # Lecture de la base de données
    db = pd.read_csv("donnees_bouteilles/base_de_donnesV2.csv")
    # Mélange de la base de donnée
    db = db.sample(frac=1)
    # Séparation des labels et des pixels
    labels = db.iloc[:, 0].values.tolist()  # Colonne des labels
    pixels_data = db.iloc[:, 1:]  # Colonnes des pixels
    "conversion en liste utilisables par l'algorithme"
    images = np.array(pixels_data.map(parse_pixel_column).values.tolist()).tolist()
    print(images[0])

    # segmentation de la base de données + aplatissage des listes
    # images_train = [[(0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]) / 255 for pixel in image] for image in images[:4000]]
    images_train = [[color / 255 for pixel in image for color in pixel] for image in images[:4000]]
    label_train = labels[:4000]
    # images_test = [[(0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]) / 255 for pixel in image] for image in images[4000:5000]]
    images_test = [[color / 255 for pixel in image for color in pixel] for image in images[4000:5000]]
    label_test = labels[4000:5000]
    # images_valid = [[(0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]) / 255 for pixel in image] for image in images[5000:]]
    images_valid = [[color / 255 for pixel in image for color in pixel] for image in images[5000:]]
    label_valid = labels[5000:]
    print(images_test[0])
    reseau = Reseau(10800, [(100, Relu, derive_Relu), (2, softmax, None)])

    nb_iteration = 200
    nb_exemples = 30
    fac_apprentissage = 0.01

    temps_moyen_passe_avant = 0
    temps_moyen_passe_arriere = 0
    for i in range(nb_iteration):
        print(f"itération {i+1} / {nb_iteration}")
        print("passe avant")
        # récupération des images dans la base de données
        temps = time.time()
        indices = [(i * nb_exemples + j) % len(images_train) for j in range(nb_exemples)]
        entrees = [images_train[indice] for indice in indices]
        sorties_attendues_chiffre = [label_train[indice] for indice in indices]
        # forme [0, 1] ou [1, 0]
        sorties_attendues = [[1 - label_train[indice], label_train[indice]] for indice in indices]
        # passe avant exécutée en parallèle, deviens plus efficace pour un grand nombre d'exemples (2x plus rapide pour 100)
        sorties_reseau = Parallel(n_jobs=nombre_coeurs)(
            delayed(lambda reseau, entree: passe_avant(reseau, entree))(reseau, entree) for entree in entrees)
        print(f"temps d'exécution de la passe avant {time.time() - temps}")
        temps_moyen_passe_avant += time.time() - temps
        print("passe arrière")
        temps = time.time()
        passe_arriere(reseau, sorties_attendues, sorties_reseau, fac_apprentissage)
        print(f"temps d'exécution de la passe arrière {time.time() - temps}")
        temps_moyen_passe_arriere += time.time() - temps

        erreur = sum(
            [-np.log(sorties_reseau[j][-1][1][sorties_attendues_chiffre[j]]) for j in range(nb_exemples)]) / nb_exemples
        print(f"erreur du réseau : {erreur}")

    # tests après entrainement
    accuracy = 0
    for i in range(100):
        indice = i % len(label_valid)
        entree = images_valid[indice]
        sortie = sortie_reseau(reseau, entree)
        accuracy += sortie[label_valid[indice]] > sortie[1-label_valid[indice]]
        print(label_valid[indice], passe_avant(reseau, entree)[-1][1])
    print(f"précision : {accuracy / 100}")
    print(f"temps moyen d'exécution de la passe avant : {temps_moyen_passe_avant / nb_iteration}")
    print(f"temps moyen d'exécution de la passe arrière : {temps_moyen_passe_arriere / nb_iteration}")
    print(f"Temps total de l'exécution : {time.time() - temps_total}")
    enregistrement = input("Entrez le nom du fichier si vous souhaitez enregistrer les poids de ce réseau")
    if enregistrement != "":
        reseau.exporter_poids(enregistrement)


# test sur des regréssions linéaires
def test_reseau():
    # on n'utilise pas de fonction d'activation pour une simple regréssion linéaire
    test_reseau = Reseau(1, [(1, lambda x: x, lambda x : 1)])
    print(test_reseau)

    # définie la fonction que le réseau doit trouver/approximer
    fnc = lambda x : 2*x - 9
    nb_iteration = 20
    nb_batch = 10
    for i in range(nb_iteration):
        entrees = [random.uniform(-2, 2) for _ in range(nb_batch)]
        sorties_attendues = [[fnc(entree)] for entree in entrees]
        sorties_reseau = [passe_avant(test_reseau, [entree]) for entree in entrees]
        passe_arriere(test_reseau,
                      sorties_attendues,
                      sorties_reseau,
                      0.5)
        erreur_moy = sum([(sorties_attendues[j][0] - sorties_reseau[j][-1][1][0])**2 for j in range(nb_batch)]) / nb_batch
        print(f"erreur moyenne du réseau : {erreur_moy}")
    print(f"réseau final : {test_reseau}")

# entrainement_chiffre()
# cProfile.run("entrainement_chiffre()")
# test_reseau()
entrainement_bouteilles()


"""
# Charger le fichier CSV
df = pd.read_csv("donnees_bouteilles/base_de_donnes.csv")

# Séparer les labels et les pixels
labels = df.iloc[:, 0].values  # Colonne des labels
pixels_data = df.iloc[:, 1:]  # Colonnes des pixels




pixels_array = np.array(pixels_data.applymap(parse_pixel_column).values.tolist())

print(f"Forme après conversion : {pixels_array.shape}")  # Devrait être (1997, 3600, 4)

n_samples, n_pixels, n_channels = pixels_array.shape

# Vérifier que chaque pixel a bien 4 valeurs
if n_channels != 4:
    raise ValueError(f"Erreur: Chaque pixel doit avoir 4 valeurs, mais shape={pixels_array.shape}")

# Vérifier que le nombre total de pixels correspond à une image carrée
image_size = int(np.sqrt(n_pixels))
if image_size * image_size != n_pixels:
    raise ValueError("Les dimensions des images ne sont pas correctes !")

# Reshape pour obtenir (n_samples, 60, 60, 4)
images = pixels_array.reshape(-1, image_size, image_size, 4).astype("float32") / 255.0

print(f"Nouvelle forme des images : {images.shape}")  # Devrait être (1997, 60, 60, 4)




# Afficher une image (par exemple la première)
for i in range(10):
    c = random.randint(1, 1998)
    plt.imshow(images[c])

    plt.title(f"Label: {labels[c]}")
    plt.show()
"""
