"""Features geométricas/topológicas das instâncias (análise de sensibilidade)."""
from __future__ import annotations

import math

import numpy as np

from .heuristics import convex_hull, mst_prim


def _polygon_area(P) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extract(inst) -> dict:
    c = inst.coords * inst.scale
    D = inst.D.astype(float)
    n = inst.n
    f = {"instance": inst.name, "n": n, "orig_type": inst.orig_type}

    # Bounding box: razão de aspecto (>= 1; 1 = quadrado)
    w, h = np.ptp(c[:, 0]), np.ptp(c[:, 1])
    f["aspect_ratio"] = max(w, h) / max(min(w, h), 1e-9)

    # Fecho convexo
    hull = convex_hull(c)
    f["hull_ratio"] = len(hull) / n
    area = _polygon_area(c[hull]) if len(hull) >= 3 else w * h
    f["hull_area_frac_bbox"] = area / max(w * h, 1e-9)

    # Vizinho mais próximo: heterogeneidade local e índice de Clark-Evans
    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    nnd = Dn.min(axis=1)
    f["nn_cv"] = float(nnd.std() / max(nnd.mean(), 1e-9))
    # R = (média NN observada) / (esperada sob Poisson); R<1 => clusters, R>1 => regular.
    # Esperada com correção de borda de Donnelly (1978): 0.5*sqrt(A/n) + (0.0514 + 0.041/sqrt(n))*P/n
    # (sem ela, para N pequeno pontos uniformes dão R ~ 1.7 por efeito de borda).
    H = c[hull] if len(hull) >= 3 else c
    perim = float(np.linalg.norm(H - np.roll(H, -1, axis=0), axis=1).sum())
    exp_nn = 0.5 * math.sqrt(max(area, 1e-9) / n) + (0.0514 + 0.041 / math.sqrt(n)) * perim / n
    f["clark_evans_R"] = float(nnd.mean() / exp_nn)

    # Entropia espacial em grade g x g (normalizada em [0,1]; 1 = uniforme)
    g = max(2, int(round(math.sqrt(n / 2))))
    gx = np.clip(((c[:, 0] - c[:, 0].min()) / max(w, 1e-9) * g).astype(int), 0, g - 1)
    gy = np.clip(((c[:, 1] - c[:, 1].min()) / max(h, 1e-9) * g).astype(int), 0, g - 1)
    counts = np.bincount(gx * g + gy, minlength=g * g)
    p = counts[counts > 0] / n
    f["spatial_entropy"] = float(-(p * np.log(p)).sum() / math.log(g * g))
    f["grid_occupancy"] = float((counts > 0).mean())

    # Árvore geradora mínima
    edges = mst_prim(inst.D)
    deg = np.zeros(n, dtype=int)
    lens = np.array([D[a, b] for a, b in edges])
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    f["mst_max_degree"] = int(deg.max())
    f["mst_leaf_frac"] = float((deg == 1).mean())
    f["mst_odd_frac"] = float((deg % 2 == 1).mean())   # tamanho do matching do Christofides
    f["mst_edge_cv"] = float(lens.std() / lens.mean())
    f["mst_len_norm"] = float(lens.sum() / math.sqrt(n * max(area, 1e-9)))

    # Distribuição das distâncias
    iu = np.triu_indices(n, 1)
    dd = D[iu]
    f["dist_cv"] = float(dd.std() / dd.mean())
    return f
