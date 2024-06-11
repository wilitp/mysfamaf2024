
from random import random
from typing import Any, Callable, List, Optional, Tuple
import math
from functools import reduce

import numpy as np


def montecarlo_multi(g, intervals: List[Tuple[float, float]], n: int):
    """
    Calcula una integral múltiple
    """

    assert g.__code__.co_argcount == len(intervals)

    # obtener a partir de los x_i los u_i a partir de cambios de variable
    # el cambio de variable depende de si los limites es a, b ambos finitos
    # o si son a = 0 y b = infinito
    def aux(): 

        factors = []
        args = []

        for a, b in intervals:
            x = random()
            u = (x * (b - a) + a) if b is not math.inf else 1/x - 1
            p = (b - a) if b is not math.inf else 1/x**2
            args.append(u)
            factors.append(p)
        
        return [reversed(args), factors]

    # generador de samples para obtener el promedio
    def gen_func():
        for _ in range(0, n):

            args, prods = aux()

            prod = reduce(lambda x, y : x*y, prods)

            yield g(*args) * prod

    gen = gen_func()

    # devolver el promedio de la muestra generada
    return (sum(gen))/n

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

def sim_esp_var(sim: Callable[[], float], n: int):
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
    while U >= F and i < len(x_vals):
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

ProbFunc = Callable[[float], float]

def rechazo(simy, px: ProbFunc, py: ProbFunc, c: Optional[float] = None, x_vals: Optional[List[float | int]] = None):

    if c is None and x_vals is None:
        raise Exception("c y x_vals no pueden ser al mismo tiempo")
    

    if c is None:

        assert x_vals is not None and len(x_vals) != 0

        c = max(px(x) / py(x) for x in x_vals)

        c = max(c, 1)
    
    while True:
        y = simy()
        U = random()

        if U < px(y) / (c * py(y)):
            return y

def aprox_sum(a, b, term, n):

    return sum(term((a + (b - a) * random())) for _ in range(n)) * (1/n)* (b-a+1)

def trans_discreta_general_infinita_new(px, mu: float = 0, rec: Optional[Callable[[float], float]]=None ):
    # mu es el punto de partida para la busqueda ascendente

    prob = px(0)
    F = prob

    for j in range(1, int(mu) + 1):
        # si tenemos una funcion recursiva aprovechamos el valor anterior
        if rec is not None:
            F += rec(prob)
        else:
            F += px(j) 

    U = random()
    
    if U >= F:
        j = int(mu) + 1


        while U >= F:
            # si tenemos una funcion recursiva aprovechamos el valor anterior
            if rec is not None:
                F += rec(prob)
            else:
                F += px(j) 
            j += 1
        
        return j - 1
    else:
        j = int(mu)
        while U < F:
            F -= px(j)
            j -= 1

        return j+1

def trans_discreta_general_infinita(px):
    U = random()
    
    i, F = 0, px(0)

    while F <= U:
        i += 1
        F += px(i)

    return i

def trans_discreta_general_infinita_optimizada(px, mu):
    # mu es el punto de partida para la busqueda

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

def composicion(alpha, simx, simy):
    U = random()

    if U <= alpha:
        return simx()
    else:
        return simy()

def composicion_new(ps: List[float], sims: List[Callable[[], float]]):

    assert len(ps) == len(sims) != 0
    assert sum(ps) == 1
    acum = []

    u = random()

    i, acc = 0, ps[0]

    while u >= acc:
        i += 1
        acc += ps[i]
    
    return sims[i]()




def trans_inversa_con_funcion(F_1: ProbFunc):
    u = random()
    return F_1(u)

class RecurrentStats:
    var: float | None = None
    mean: float | None = None
    n: int = 0


    def add(self, x: float):

        if self.n == 0:


            self.n = 1
            self.var = 1
            self.mean = x
            return
        

        assert(self.mean is not None)
        assert(self.var is not None)
        # Aplicar formulas recursivas
        new_mean = self.mean + (x - self.mean)/(self.n+1)

        new_var = (1- 1/self.n) * self.var + (self.n+1) * (new_mean - self.mean)**2

        self.mean = new_mean
        self.var = new_var
        self.n += 1
    
    @property
    def get_mean(self):
        if self.n == 0:
            raise Exception("No data")

        assert(self.mean is not None)

        return self.mean

    @property
    def get_var(self):
        if self.n == 0:
            raise Exception("No data")
        assert(self.var is not None)

        return self.var
    

def estadistico_prueba(ps, Ns):
    # Calcula el estadistico de prueba para hacer el test de chi-cuadrado de Pearson
    T = sum((N - sum(Ns)*ps[i])**2 / (sum(Ns)*ps[i]) for (i, N) in enumerate(Ns))
    return T

def estadistico_prueba_sim(ps: List[float], n: int):
    # Calcula el estadistico de prueba para hacer el test de chi-cuadrado de Pearson
    # a partir de una muestra simulada

    sim = lambda : trans_inversa_general_optimizada(lambda i : ps[i], list(range(len(ps))))
    Ns = np.zeros(len(ps))

    for _ in range(n):
        Ns[sim()] += 1
    
    return estadistico_prueba(ps, Ns)
    
def estadistico_prueba_sim_binomiales(ps: List[float], m: int):
    # Calcula el estadistico de prueba para hacer el test de chi-cuadrado de Pearson
    # a partir de una muestra simulada
    # Usar este cuando n sea muy grande(es mas barato computacionalmente)

    n = m

    Ns = [0] * len(ps)

    p_accum = 0 # probabilidad acumulada de los valores anteriores

    for i, pi in enumerate(ps):
        Ns[i] = np.random.binomial(n, pi/(1-p_accum))
    
    return estadistico_prueba(ps, Ns)

def estadistico_smirnov(sample: List[float], F: Callable[[float], float]):
    sample = sorted(sample)

    n = len(sample)

    d = max(max((i+1)/n - F(x), F(x) - i/n) for i, x in enumerate(sample))

    return d

def estadistico_smirnov_sim(n):
    sample = sorted([random() for _ in range(n)])

    d = estadistico_smirnov(sample, lambda x : x)

    return d
