import gc
import numpy as np
from .tsp import heuristics as H, instances as I

for n in (200, 400):
    inst = I.load("pr1002", n)
    for modo in ("gc ligado", "gc desligado"):
        ts = []
        for _ in range(3):
            gc.collect()
            gc.disable() if modo == "gc desligado" else gc.enable()
            p = H.Profiler()
            H.christofides(inst.D, inst.coords, p)
            gc.enable()
            ts.append(p.times["matching"])
        print(n, modo, round(float(np.median(ts)), 3), "s")