"""Wrapper Python para o núcleo C++ do Branch-and-Bound (src/bnb/bnb)."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from .instances import ROOT

BIN = os.path.join(ROOT, "src", "bnb", "bnb")


def solve(D: np.ndarray, ub: int, tour, bound: str = "hk", time_limit: float = 1800.0,
          cpu: int | None = None) -> dict:
    """Roda o B&B com warm-start (ub, tour). Timeout é tratado DENTRO do binário
    (preserva contadores de nós); o timeout do subprocess é só uma rede de segurança."""
    n = len(D)
    buf = [str(n)]
    buf.extend(" ".join(map(str, row)) for row in D.tolist())
    buf.append(str(int(ub)))
    buf.append(" ".join(map(str, tour)))
    cmd = [BIN, "--bound", bound, "--time-limit", str(time_limit)]
    if cpu is not None:
        cmd = ["taskset", "-c", str(cpu)] + cmd
    p = subprocess.run(cmd, input="\n".join(buf) + "\n", capture_output=True, text=True,
                       timeout=time_limit + 300)
    if p.returncode != 0:
        raise RuntimeError(f"bnb falhou ({p.returncode}): {p.stderr[:500]}")
    return json.loads(p.stdout)


def physical_cpus() -> list[int]:
    """Uma CPU lógica por núcleo físico (evita dois workers no mesmo núcleo via
    hyperthreading, o que distorceria as medições de tempo)."""
    seen, cpus = set(), []
    base = "/sys/devices/system/cpu"
    try:
        for name in sorted(os.listdir(base), key=lambda s: (len(s), s)):
            if not (name.startswith("cpu") and name[3:].isdigit()):
                continue
            topo = os.path.join(base, name, "topology")
            with open(os.path.join(topo, "core_id")) as f:
                core = f.read().strip()
            with open(os.path.join(topo, "physical_package_id")) as f:
                pkg = f.read().strip()
            if (pkg, core) not in seen:
                seen.add((pkg, core))
                cpus.append(int(name[3:]))
    except OSError:
        pass
    return cpus or list(range(os.cpu_count() or 1))
