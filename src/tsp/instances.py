"""Leitura de instâncias TSPLIB e montagem do conjunto de teste.

Todas as instâncias são tratadas como EUC_2D: d(i,j) = nint(||p_i - p_j||).
A TSPLIB não tem nenhuma instância EUC_2D "oficial" com N <= 40, então:
  * instâncias GEO/ATT/EXPLICIT com coordenadas (NODE_COORD ou DISPLAY_DATA)
    têm suas distâncias RECALCULADAS como EUC_2D sobre as coordenadas;
    coordenadas GEO (graus) são multiplicadas por 100 antes do arredondamento,
    senão nint() destruiria a precisão (distâncias da ordem de 1-10);
  * sub-instâncias "nome_kK" = primeiros K pontos de uma instância EUC_2D maior,
    para cobrir a faixa N <= 40 com geometrias variadas (uniforme, clusters,
    grade, alongada).
Os ótimos publicados na TSPLIB só valem para instâncias que já eram EUC_2D
(eil51, st70); para as demais o ótimo vem do próprio B&B.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSPLIB_DIR = os.path.join(ROOT, "data", "tsplib")

# Ótimos publicados (EUC_2D original). Demais: determinados pelo B&B.
KNOWN_OPT = {"eil51": 426, "berlin52": 7542, "st70": 675}


@dataclass
class Instance:
    name: str
    coords: np.ndarray            # (n, 2) float
    source: str                   # arquivo TSPLIB de origem
    orig_type: str                # EDGE_WEIGHT_TYPE original
    scale: float = 1.0
    known_opt: int | None = None
    _D: np.ndarray | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.coords)

    @property
    def D(self) -> np.ndarray:
        """Matriz de distâncias inteiras (int64), calculada uma vez e cacheada."""
        if self._D is None:
            c = self.coords * self.scale
            diff = c[:, None, :] - c[None, :, :]
            self._D = np.floor(np.sqrt((diff ** 2).sum(-1)) + 0.5).astype(np.int64)
        return self._D


def parse_tsplib(path: str) -> tuple[dict, np.ndarray]:
    """Lê o cabeçalho e a seção de coordenadas (NODE_COORD ou DISPLAY_DATA)."""
    header, coords = {}, []
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue
            if line.endswith("SECTION"):
                section = line
                continue
            if line[0].isalpha():          # linha de cabeçalho "CHAVE : valor"
                section = None
                if ":" in line:
                    k, v = line.split(":", 1)
                    header[k.strip()] = v.strip()
                continue
            if section in ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION"):
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
            # EDGE_WEIGHT_SECTION é ignorada: recalculamos tudo como EUC_2D
    n = int(header["DIMENSION"])
    arr = np.array(coords, dtype=float)
    if len(arr) < n:
        raise ValueError(f"{path}: sem coordenadas suficientes ({len(arr)}/{n})")
    return header, arr[:n]


def load(name: str, k: int | None = None) -> Instance:
    path = os.path.join(TSPLIB_DIR, name + ".tsp")
    header, coords = parse_tsplib(path)
    etype = header.get("EDGE_WEIGHT_TYPE", "?")
    scale = 100.0 if etype == "GEO" else 1.0
    if k is not None:
        return Instance(f"{name}_k{k}", coords[:k].copy(), name, etype, scale)
    opt = KNOWN_OPT.get(name) if etype == "EUC_2D" else None
    return Instance(name, coords, name, etype, scale, opt)


# Conjunto de teste: 15 instâncias, 12 (80%) com N <= 40, 3 de estresse em [45, 75].
TEST_SET: list[tuple[str, int | None]] = [
    ("kroA100", 12),     # uniforme
    ("burma14", None),   # GEO -> EUC_2D
    ("ulysses16", None), # GEO -> EUC_2D
    ("berlin52", 20),
    ("ulysses22", None), # GEO -> EUC_2D
    ("kroB100", 25),     # uniforme
    ("bayg29", None),    # EXPLICIT com DISPLAY_DATA
    ("bier127", 30),     # clusters (biergärten de Augsburg)
    ("a280", 33),        # placa de circuito: pontos contíguos/alinhados
    ("pr76", 35),
    ("lin105", 38),      # estrutura em grade
    ("eil101", 40),
    ("att48", None),     # ATT -> EUC_2D  (estresse)
    ("eil51", None),     # EUC_2D         (estresse)
    ("st70", None),      # EUC_2D         (estresse)
]


def test_set() -> list[Instance]:
    return [load(n, k) for n, k in TEST_SET]


def scaling_set(base: str = "a280", sizes=None) -> list[Instance]:
    """Sub-instâncias de tamanho crescente para regressão de complexidade das
    heurísticas (que rodam em segundos mesmo para N grande)."""
    sizes = sizes or list(range(10, 281, 10))
    return [load(base, k) for k in sizes]


def tour_cost(D: np.ndarray, tour) -> int:
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())
