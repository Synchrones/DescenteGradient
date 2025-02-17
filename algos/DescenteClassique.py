import random
import numpy as np


def descente_gradient(fnc, derives, depart, epsilon, pas):
    """
    Descente de gradient classique à n variables
    IN : fnc (function) - la fonction étudiée
    IN : derives (list(function)) - les dérivées selon chaque variable de la fonction
    IN : depart (list(float)) - le point de départ de la descente
    IN : epsilon (float) - la valeur à partir de laquelle on arrête la descente (on atteint pas forcément le 0)
    IN : pas (float) - la vitesse de convergence (ne doit pas etre trop gros ou trop petit)
    """
    iterations = 0
    while abs(fnc(*depart)) > epsilon:
        pentes = [derives[i](depart[i]) for i in range(len(derives))]
        for var in range(len(pentes)):
            depart[var] -= pentes[var] * pas
        iterations += 1
        print(depart)
    return depart, iterations

# ex d'appel
print(descente_gradient(lambda x, y: x**2 + y**2,
                        [lambda x: 2*x, lambda y: 2*y],
                        [random.randint(-100, 100), random.randint(-100, 100)],
                        0.1,
                        0.1))