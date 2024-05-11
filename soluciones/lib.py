
from random import random
from typing import Any, Callable, List, Optional

def urna(px: Callable[[Any], float], x_vals: List[Any]):
    # 10 decimales de definicion maxima
    MAX_DEFINITION = 10

    def def_for_prob(p):
        if p > 1:
            raise Exception("Probabilidad > 1")
        elif p == 0:
            return 0
        else:
            return len(str(p)) - 2 # le sacamos el cero y el punto
    

    definition = min(MAX_DEFINITION, max(def_for_prob(px(x)) for x in x_vals))


    # Genero la urna
    urn = []

    for x in x_vals:
        urn += [x] * int( 10**definition * px(x))
    
    U = discreta_uniforme(0, len(urn)- 1)


    return urn[U]


    


def sim_esp_var(sim, n):
    esp  = sum(sim() for _ in range(0, n))/n
    esp2 = sum(sim()**2 for _ in range(0, n))/n

    var = esp2 - esp**2

    return esp, var

def trans_inversa_general_optimizada(p: Callable[[Any], float], x_vals: List[Any]):
    aux = list(zip(x_vals, (p(x) for x in x_vals)))

    aux = sorted(aux, key = lambda x : x[1], reverse=True)

    lookup_table = {}

    for x, prob in aux:
        lookup_table[x] = prob
    
    new_px = lambda x : lookup_table.get(x, 0)

    new_x_vals = [x for x, _ in aux]

    return trans_inversa_general(new_px, new_x_vals)

def trans_inversa_general(p: Callable[[Any], float], x_vals: List[Any]):
    U = random()

    # Empezamos a acumular la probabilidad desde el primer elemento

    r = x_vals[0]
    i, F = 0, p(r)

    # F es la acumulada
    while U >= F:
        i += 1
        F += p(x_vals[i])
    
    return x_vals[i]

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


def rechazo(simy, px, py, c: Optional[float] = None, x_vals: Optional[List] = None):

    if c is None and x_vals is None:
        raise Exception("c y x_vals no pueden ser al mismo tiempo")
    

    if c is None:
        assert x_vals is not None

        c = max(px(x) / py(x) for x in x_vals)
    
    while True:
        y = simy()
        U = random()

        if U < px(y) / (c * py(y)):
            return y

def aprox_sum(a, b, term, n):

    return sum(term((a + (b - a) * random())) for _ in range(n)) * (1/n)* (b-a+1)


def trans_discreta_general_infinita(px):
    U = random()

    i, F = 0, px(0)

    while U >= F:
        i += 1
        F += px(i)

    return i

def trans_discreta_general_infinita_optimizada(px, mu):

    F = px(0)

    for j in range(1, int(mu) + 1):
        F += px(j) 

    U = random()
    
    if U >= F:
        j = int(mu) + 1

        while U >= F:
            F += px(j)
            j += 1
        
        return j - 1
    else:
        j = int(mu)
        while U < F:
            F -= px(j)
            j -= 1

        return j+1

