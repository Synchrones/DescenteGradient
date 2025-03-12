import random

import numpy as np


def descente_gradient(derives, depart, epsilon, pas, facteur_momentum):
    """
    Descente de gradient classique à n variables
    IN : derives (list(function)) - les dérivées selon chaque variable de la fonction
    IN : depart (list(float)) - le point de départ de la descente
    IN : epsilon (float) - la valeur de la dérivée à partir de laquelle on arrête la descente (on n'atteint pas forcément le 0)
    IN : pas (float) - la vitesse de convergence (ne doit pas être trop gros ou trop petit)
    IN : facteur_momentum (float) - le facteur de prise de momentum
    """
    iterations = 0
    terminer = False
    nouvelles_coords = [0 for _ in range(len(derives))]
    momentum = [0 for _ in range(len(derives))]
    while not terminer:
        terminer = True
        pentes = [derives[i](*depart) for i in range(len(derives))]
        for var in range(len(pentes)):
            nouvelles_coords[var] = depart[var] - pentes[var] * pas + facteur_momentum * momentum[var]
            momentum[var] = nouvelles_coords[var] - depart[var]
            depart[var] = nouvelles_coords[var]
            if abs(pentes[var]) > epsilon:
                terminer = False
        iterations += 1
        print(depart)
    return depart, iterations

# ex d'appel
print(descente_gradient([lambda x, y: 2*x, lambda x, y: 2*y],
                        [random.randint(-100, 100), random.randint(-100, 100)],
                        0.01,
                        0.4,
                        0.1))