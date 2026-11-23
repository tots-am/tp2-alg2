"""Heurísticas construtivas e algoritmos aproximados, instrumentados por sub-rotina.

Cada função recebe a matriz de distâncias D (int64, n x n) — e coords quando
precisa de geometria — e um Profiler. O Profiler registra o tempo de cada
sub-rotina (perf_counter) e, se ligado com memory=True, o pico de memória
alocada (tracemalloc). tracemalloc deixa o código 2-3x mais lento, por isso as
medições de tempo e de memória são feitas em passadas separadas.

Gargalos esperados (verificados no profiling):
  * Twice-around-the-tree: MST (Prim denso O(n^2), vetorizado em NumPy).
  * Christofides: emparelhamento perfeito de peso mínimo (blossom do NetworkX,
    O(k^3) em Python puro, k = nº de vértices de grau ímpar).
"""
from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager

import networkx as nx
import numpy as np


class Profiler:
    def __init__(self, memory: bool = False):
        self.memory = memory
        self.times: dict[str, float] = defaultdict(float)
        self.peaks: dict[str, int] = defaultdict(int)

    @contextmanager
    def section(self, name: str):
        if self.memory:
            tracemalloc.start()
            tracemalloc.reset_peak()
        t = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - t
            if self.memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.peaks[name] = max(self.peaks[name], peak)


def _cost(D, tour) -> int:
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())


# ----------------------------------------------------------------------------
# 1. Trivial (cold start)
# ----------------------------------------------------------------------------
def trivial(D, coords=None, prof: Profiler | None = None):
    prof = prof or Profiler()
    with prof.section("build"):
        tour = list(range(len(D)))
    return tour


# ----------------------------------------------------------------------------
# 2. Vizinho mais próximo, a partir do vértice 0.  O(n^2)
# ----------------------------------------------------------------------------
def nearest_neighbor(D, coords=None, prof: Profiler | None = None, start: int = 0):
    prof = prof or Profiler()
    with prof.section("nn_walk"):
        n = len(D)
        visited = np.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True
        cur = start
        big = np.iinfo(np.int64).max
        for _ in range(n - 1):
            row = np.where(visited, big, D[cur])     # máscara vetorizada: O(n) por passo
            cur = int(row.argmin())
            visited[cur] = True
            tour.append(cur)
    return tour


# ----------------------------------------------------------------------------
# 3. Fecho convexo + inserção mais barata.  O(n^3) no total (O(n^2) por passo)
# ----------------------------------------------------------------------------
def convex_hull(coords) -> list[int]:
    """Andrew's monotone chain, O(n log n). Retorna índices em sentido anti-horário."""
    idx = sorted(range(len(coords)), key=lambda i: (coords[i][0], coords[i][1]))

    def cross(o, a, b):
        return ((coords[a][0] - coords[o][0]) * (coords[b][1] - coords[o][1])
                - (coords[a][1] - coords[o][1]) * (coords[b][0] - coords[o][0]))

    lower, upper = [], []
    for i in idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def hull_insertion(D, coords, prof: Profiler | None = None):
    prof = prof or Profiler()
    n = len(D)
    with prof.section("hull"):
        tour = convex_hull(coords)
        if len(tour) < 3:                       # pontos colineares
            tour = list(range(min(3, n)))
    with prof.section("insertion"):
        in_tour = np.zeros(n, dtype=bool)
        in_tour[tour] = True
        rest = np.flatnonzero(~in_tour)
        while rest.size:
            t = np.asarray(tour)
            nxt = np.roll(t, -1)
            # custo de inserir cada r entre (t_i, t_{i+1}): |R| x |T|
            delta = D[t][:, rest].T + D[nxt][:, rest].T - D[t, nxt][None, :]
            flat = int(delta.argmin())
            ri, ei = divmod(flat, len(t))
            tour.insert(ei + 1, int(rest[ri]))
            rest = np.delete(rest, ri)
    # rotaciona para começar em 0 (conveniência para o B&B)
    z = tour.index(0)
    return tour[z:] + tour[:z]


# ----------------------------------------------------------------------------
# Sub-rotinas compartilhadas: MST (Prim denso), Euler, atalhos
# ----------------------------------------------------------------------------
def mst_prim(D) -> list[tuple[int, int]]:
    """Prim O(n^2) sobre matriz densa — ótimo para grafo completo
    (heap daria O(n^2 log n) pois m = n^2)."""
    n = len(D)
    in_t = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1)
    key[0] = 0
    for _ in range(n):
        u = int(np.where(in_t, np.inf, key).argmin())
        in_t[u] = True
        row = D[u]
        upd = (~in_t) & (row < key)
        key[upd] = row[upd]
        parent[upd] = u
    return [(int(parent[v]), v) for v in range(n) if parent[v] >= 0]


def preorder(n, edges, root=0) -> list[int]:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, order, stack = [False] * n, [], [root]
    while stack:
        v = stack.pop()
        if seen[v]:
            continue
        seen[v] = True
        order.append(v)
        stack.extend(reversed(adj[v]))
    return order


def euler_circuit(n, edges, start=0) -> list[int]:
    """Hierholzer iterativo sobre multigrafo (lista de arestas). O(m)."""
    adj = [[] for _ in range(n)]
    for eid, (a, b) in enumerate(edges):
        adj[a].append((b, eid))
        adj[b].append((a, eid))
    used = [False] * len(edges)
    ptr = [0] * n
    stack, circuit = [start], []
    while stack:
        v = stack[-1]
        while ptr[v] < len(adj[v]) and used[adj[v][ptr[v]][1]]:
            ptr[v] += 1
        if ptr[v] == len(adj[v]):
            circuit.append(stack.pop())
        else:
            u, eid = adj[v][ptr[v]]
            used[eid] = True
            stack.append(u)
    return circuit[::-1]


def shortcut(walk) -> list[int]:
    seen, tour = set(), []
    for v in walk:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    return tour


# ----------------------------------------------------------------------------
# 4. Twice-around-the-tree (2-aproximação)
# ----------------------------------------------------------------------------
def twice_around_tree(D, coords=None, prof: Profiler | None = None):
    prof = prof or Profiler()
    with prof.section("mst"):
        edges = mst_prim(D)
    with prof.section("preorder"):
        tour = preorder(len(D), edges, 0)      # = circuito euleriano + atalhos
    return tour


# ----------------------------------------------------------------------------
# 5. Christofides (1.5-aproximação)
# ----------------------------------------------------------------------------
def christofides(D, coords=None, prof: Profiler | None = None):
    prof = prof or Profiler()
    n = len(D)
    with prof.section("mst"):
        edges = mst_prim(D)
    with prof.section("odd_vertices"):
        deg = np.zeros(n, dtype=int)
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        odd = np.flatnonzero(deg % 2 == 1)
    with prof.section("matching_build_graph"):
        G = nx.Graph()
        sub = D[np.ix_(odd, odd)]
        for i in range(len(odd)):
            for j in range(i + 1, len(odd)):
                G.add_edge(int(odd[i]), int(odd[j]), weight=int(sub[i, j]))
    with prof.section("matching"):
        M = nx.min_weight_matching(G)          # blossom de Edmonds, O(k^3)
    with prof.section("euler"):
        walk = euler_circuit(n, edges + [tuple(e) for e in M], 0)
    with prof.section("shortcut"):
        tour = shortcut(walk)
    return tour


STRATEGIES = {
    "trivial": trivial,
    "nn": nearest_neighbor,
    "hull": hull_insertion,
    "tat": twice_around_tree,
    "christofides": christofides,
}


def run_strategy(name, inst, memory=False):
    """Executa uma estratégia e devolve (tour, custo, tempo_total, profiler)."""
    prof = Profiler(memory=memory)
    fn = STRATEGIES[name]
    t = time.perf_counter()
    tour = fn(inst.D, inst.coords, prof)
    total = time.perf_counter() - t
    assert sorted(tour) == list(range(inst.n)), f"{name}: tour inválido"
    return tour, _cost(inst.D, tour), total, prof
