from reseau_neurones import *

#données
import pandas as pd
pd.options.display.max_columns = None
import tqdm

#affichage
from matplotlib import pyplot as plt
import matplotlib


def Image_matrice(table : pd.DataFrame, ligne : int) -> pd.array:

    '''
    Retourne l'image de la ligne d'index 'ligne' dans le dataset 'table' sous forme d'une matrices 
    de taille 28 x 28.
    IN : table (DataFrame) - représente le fichier dans lequel on est
    IN : ligne (int) - le numéro de la ligne du chiffre que l'on cherhce a retourner
    '''

    return table.iloc[ligne, 0:].values.reshape(28,28)


def Image_ligne(table : pd.DataFrame, ligne : int) -> pd.Series:

    '''
    Retourne l'image de la ligne d'index 'ligne' dans le dataset 'table' sous forme d'une ligne (784 x 1).
    IN : table (DataFrame) - représente le fichier dans lequel on est
    IN : ligne (int) - le numéro de la ligne du chiffre que l'on cherhce a retourner
    '''

    return table.iloc[ligne, 0:]


# scenario
if __name__ == '__main__':
    #telechargement de la base de donnée
    TRAIN = pd.read_csv('train.csv')#, skiprows = 1)
    TEST = pd.read_csv("test.csv")#, skiprows = 1)
    X_TRAIN = TRAIN.copy()
    X_TEST = TEST.copy()
    y = TRAIN.label
    del X_TRAIN['label']
    taille = 0
    for key in X_TRAIN:
        taille += 1
    couche1 = 100
    nbsorties = 10
    R = reseau([taille, couche1, nbsorties])
    B = biais(R)
    sorties = np.empty((1, 1))
    for i in tqdm.tqdm(range(len(X_TRAIN))):
        entrees = X_TRAIN.loc[i].tolist()
        sorties = np.concatenate(sorties, calcul(entrees, R, lambda x : 1 / (1 + np.exp(x)), B))
        #print(sorties)
        #print(i)
    print('fini')