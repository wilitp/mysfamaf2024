
from random import random
from typing import Any, List


def discreta_general(p: List[float], x: List[Any]):
    U = random()

    i, F = 0, p[0]

    # F es la acumulada

    while U >= F:
        i += 1
        F = p[i]
    
    return x[i]

def discreta_uniforme(m: int, n:int):
    """
    Genera un valor entero en [m, n] con igual probabilidad

    es lo mismo que randint(m, n)
    """

    assert m <= n

    j = random()

    return m + int(j * (n - m + 1))

def permutacion(a: List[Any]):
    """
    Permuta in place de manera aleatoria
    """

    # Estrategia:
    # Recorre `a` e intercambia cada a[j] con algún índice posterior
    # Se puede probar que este método genera todas las permutaciones y lo hace con igual probabilidad

    N = len(a)


    for j in range(0, N):

        index = discreta_uniforme(j, N-1)

        a[j], a[index] = a[index], a[j]

def subconjunto(a: List[Any], r):
    acopy = [*a]

    permutacion(acopy)

    return acopy[0:r]