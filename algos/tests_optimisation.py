import multiprocessing
import time

from joblib import Parallel, delayed

nombre_coeurs = multiprocessing.cpu_count()


def test_parallelisation():
    temps1 = time.time()
    test1 = [i**2 for i in range(1000)]
    temps1 = time.time() - temps1
    temps2 = time.time()
    test2 = Parallel(n_jobs=nombre_coeurs)(delayed(lambda x : x**2)(i) for i in range(1000))
    temps2 = time.time() - temps2
    print(temps1, temps2)


def test_liste_append():
    moyenne = 0
    for _ in range(10):
        temps1 = time.time()
        test1 = [0] * 1000000
        for i in range(1000000):
            test1[i] = i
        temps1 = time.time() - temps1
        temps2 = time.time()
        test2 = []
        for i in range(1000000):
            test2.append(i)
        temps2 = time.time() - temps2
        print(temps1, temps2)
        moyenne += temps2 / temps1
    print(f"Append est {moyenne/10} fois plus lent")

test_liste_append()
