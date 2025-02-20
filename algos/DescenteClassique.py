import random


def descente_gradient(derives, depart, epsilon, pas):
    """
    Descente de gradient classique à n variables
    IN : derives (list(function)) - les dérivées selon chaque variable de la fonction
    IN : depart (list(float)) - le point de départ de la descente
    IN : epsilon (float) - la valeur de la dérivée à partir de laquelle on arrête la descente (on n'atteint pas forcément le 0)
    IN : pas (float) - la vitesse de convergence (ne doit pas être trop gros ou trop petit)
    """
    iterations = 0
    terminer = False
    while not terminer:
        terminer = True
        pentes = [derives[i](depart[i]) for i in range(len(derives))]
        for var in range(len(pentes)):
            depart[var] -= pentes[var] * pas
            if abs(pentes[var]) > epsilon:
                terminer = False
        iterations += 1
        print(depart)
    return depart, iterations

# ex d'appel
print(descente_gradient([lambda x: 2*x, lambda y: 2*y],
                        [random.randint(-100, 100), random.randint(-100, 100)],
                        0.1,
                        0.4))